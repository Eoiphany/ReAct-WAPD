"""注释
命令:
1. `python3 -m Autobs.train_ppo -v single -r coverage --iters 30`

# 训练综合得分的PPO模型，使用数据集512张图，输出checkpoint到指定目录
# 旧方案：展平 256×256 → MLP
# 新方法：展平 256×256 → reshape 成 1×256×256 → CNN backbone → shared feature → actor/value head
python -m Autobs.train_ppo \
  -v single \
  -r score \
  --dataset_path /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight \
  --dataset_limit 512 \
  --iters 200 \
  --module-state-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/pretrained_policy_cnn_rmnet/best_module_state.pt \
  --network-type rmnet \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/checkpoints/rmnet_radiomap3dseer.pt \
  --heuristic-targets-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/heuristic_targets.json \
  --checkpoint_dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/checkpoints_score_v2_cnn_rmnet


python -m Autobs.train_ppo \
  -v single \
  -r score \
  --dataset_path /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight \
  --dataset_limit 512 \
  --iters 200 \
  --network-type rmnet \
  --module-state-path   --module-state-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/pretrained_policy/best_module_state.pt \
  --model-path /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/checkpoints/rmnet_radiomap3dseer.pt \
  --heuristic-targets-path /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/outputs/heuristic_targets.json \
  --checkpoint_dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/checkpoints_score_v2_pretrained

之前没有network-type时默认使用PMNet，可能导致和最终对比的结果一样，学不到东西呢。
你预训练只初始化了 actor，不是 critic,critic 还是随机的，第一次 advantage 估计会很噪，容易把 actor 推坏
这次训用的pretrained_policy，不是pretrained_policy_val


python -m Autobs.train_ppo -v single -r score --dataset_path /Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight --iters 200 --dataset_limit 512 --checkpoint_dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/checkpoints_zonghe

python -m Autobs.train_ppo -v single -r score --dataset_path /root/autodl-tmp/coding/test/buildingsWHeight --iters 200 --dataset_limit 512 --checkpoint_dir /root/autodl-tmp/code/Autobs/checkpoints_zonghe

3. `python3 -m Autobs.train_ppo -v multi -r score --city_map_path /abs/path/to/maps --dataset_limit 64 --iters 50`
4. `python3 -m Autobs.train_ppo --resume /abs/path/to/checkpoint --checkpoint_dir /abs/path/to/output`
5. `python3 -m Autobs.train_ppo --module-state-path /abs/path/to/pretrained_module_state.pt --checkpoint_dir /abs/path/to/output`

参数含义:
- `-v, --version`: 训练模式，`single` 表示单站点，`multi` 表示多站点。
- `-r, --reward_type`: 训练奖励类型，当前仅保留统一后的 `score`。
- `--city_map_path`: 城市地图文件或目录路径；默认使用 `paper_experiment/dataset/png/buildingsWHeight`。
- `--dataset_path`: `--city_map_path` 的直观别名，推荐用于显式指定数据集路径。
- `--dataset_limit`: 仅使用数据集前 N 张图进行训练；默认不截断，使用整个目录。
- `--dataset_offset`: 从排序后的数据集第几个样本开始取。
- `--dataset_stride`: 从排序后的数据集按步长抽样，便于构造稀疏子集。
- `--iters`: PPO 训练迭代次数。
- `--checkpoint_dir`: checkpoint 输出目录；默认写入 `Autobs/checkpoints`。
- `--resume`: 从已有 RLlib checkpoint 恢复训练。
- `--module-state-path`: 从离线监督预训练得到的策略模块权重初始化 PPO；与 `--resume` 二选一。
- `--network-type`: 训练奖励使用的 surrogate 类型；需与 `--model-path` 对应。
- `--metrics_path`: 奖励历史输出文件；不传则默认写入 checkpoint 目录下的 `reward_history.jsonl`。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

from Autobs.paths import CHECKPOINT_DIR, DEFAULT_DATASET_MAP_DIR, ensure_runtime_dirs

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


ENV_MODULES = {
    "single": "Autobs.env.single_site",
    "multi": "Autobs.env.multi_site",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", choices=sorted(ENV_MODULES), default="single")
    parser.add_argument(
        "-r",
        "--reward_type",
        choices=["score"],
        default="score",
    )
    parser.add_argument("--city_map_path", "--dataset_path", dest="city_map_path", default=str(DEFAULT_DATASET_MAP_DIR))
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--dataset_offset", type=int, default=0)
    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--checkpoint_dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--module-state-path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--network-type", default="pmnet", choices=["pmnet", "pmnet_v3", "rmnet", "rmnet_v3"])
    parser.add_argument("--heuristic-targets-path", default=None)
    return parser


def load_environment_class(version: str):
    module_name = ENV_MODULES[version]
    module = importlib.import_module(module_name)
    return module.BaseEnvironment


def load_reward_history(history_path: str | Path) -> list[dict[str, Any]]:
    path = Path(history_path).expanduser().resolve()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def render_training_curves(history: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("Training history is empty; cannot render curves")

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [int(item["iter"]) for item in history]
    reward_means = [float(item["reward_mean"]) for item in history]
    timesteps = [float(item["timesteps"]) if item.get("timesteps") is not None else float("nan") for item in history]

    reward_curve_png = output_dir / "reward_curve.png"
    timesteps_curve_png = output_dir / "timesteps_curve.png"
    training_summary_json = output_dir / "training_summary.json"

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(iterations, reward_means, color="#0b6e4f", linewidth=2.0)
    ax.set_title("PPO Reward Mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward Mean")
    ax.grid(alpha=0.25)
    fig.savefig(reward_curve_png, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(iterations, timesteps, color="#1f4e79", linewidth=2.0)
    ax.set_title("PPO Sampled Timesteps")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Timesteps")
    ax.grid(alpha=0.25)
    fig.savefig(timesteps_curve_png, dpi=200)
    plt.close(fig)

    summary = {
        "num_points": len(history),
        "best_reward_mean": max(reward_means),
        "final_reward_mean": reward_means[-1],
        "final_timesteps": timesteps[-1],
    }
    training_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "reward_curve_png": str(reward_curve_png),
        "timesteps_curve_png": str(timesteps_curve_png),
        "training_summary_json": str(training_summary_json),
    }


def build_agent(args: argparse.Namespace):
    environment_class = load_environment_class(args.version)
    env_overrides = {
        "reward_type": args.reward_type,
        "city_map_path": args.city_map_path,
        "dataset_limit": args.dataset_limit,
        "dataset_offset": args.dataset_offset,
        "dataset_stride": args.dataset_stride,
        "model_path": args.model_path,
        "network_type": args.network_type,
        "heuristic_targets_path": args.heuristic_targets_path,
    }
    if args.resume:
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.util.annotations import RayDeprecationWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=RayDeprecationWarning)
            agent = Algorithm.from_checkpoint(str(Path(args.resume).expanduser().resolve()))
            agent.config.env_config.update(env_overrides)
    else:
        from Autobs.ppo_config import get_ppo_config

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            agent = get_ppo_config(environment_class, env_config_overrides=env_overrides)
    return agent


def load_module_state(module_state_path: str | Path) -> dict[str, Any]:
    import torch

    path = Path(module_state_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported module-state payload type: {type(payload)!r}")
    return payload


def apply_module_state(agent, module_state: dict[str, Any]) -> None:
    get_module = getattr(agent, "get_module", None)
    if not callable(get_module):
        raise AttributeError("Agent does not expose get_module(); cannot apply pretrained module state")
    module = get_module()
    current_state = module.state_dict()
    merged_state = dict(current_state)

    for key, value in module_state.items():
        mapped_key = key
        if key.startswith("encoder.encoder.net.mlp."):
            suffix = key.removeprefix("encoder.encoder.net.mlp.")
            candidate = f"encoder.actor_encoder.net.mlp.{suffix}"
            if candidate in current_state:
                mapped_key = candidate
        if mapped_key not in current_state:
            continue
        if hasattr(current_state[mapped_key], "shape") and hasattr(value, "shape"):
            if tuple(current_state[mapped_key].shape) != tuple(value.shape):
                continue
        merged_state[mapped_key] = value

    module.load_state_dict(merged_state, strict=True)


def run_training(args: argparse.Namespace) -> None:
    ensure_runtime_dirs()
    if args.resume and args.module_state_path:
        raise ValueError("--resume and --module-state-path are mutually exclusive")
    if args.model_path:
        os.environ["AUTOBS_SURROGATE_NETWORK_TYPE"] = args.network_type
        resolved_model_path = str(Path(args.model_path).expanduser().resolve())
        if args.network_type.startswith("pmnet"):
            os.environ["AUTOBS_PMNET_WEIGHTS"] = resolved_model_path
        else:
            os.environ["AUTOBS_RMNET_WEIGHTS"] = resolved_model_path
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path).expanduser().resolve() if args.metrics_path else checkpoint_dir / "reward_history.jsonl"

    agent = build_agent(args)
    if args.module_state_path and not args.resume:
        module_state = load_module_state(args.module_state_path)
        apply_module_state(agent, module_state)
        print(f"loaded_module_state={Path(args.module_state_path).expanduser().resolve()}")
    for iteration in range(1, args.iters + 1):
        result = agent.train()
        reward_mean = result.get("episode_reward_mean")
        if reward_mean is None:
            reward_mean = result.get("sampler_results", {}).get("episode_reward_mean")
        if reward_mean is None:
            reward_mean = result.get("env_runners", {}).get("episode_return_mean")
        timesteps = result.get("timesteps_total") or result.get("num_env_steps_sampled")
        print(f"iter={iteration} reward_mean={reward_mean} timesteps={timesteps}")
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "iter": iteration,
                        "reward_mean": reward_mean,
                        "timesteps": timesteps,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        checkpoint = agent.save(str(checkpoint_dir))
        print(f"iter={iteration} checkpoint={checkpoint}")

    history = load_reward_history(metrics_path)
    if history:
        artifacts = render_training_curves(history, checkpoint_dir)
        print(f"reward_curve={artifacts['reward_curve_png']}")
        print(f"timesteps_curve={artifacts['timesteps_curve_png']}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
