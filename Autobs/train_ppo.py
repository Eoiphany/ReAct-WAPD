"""注释
命令:
1. `python3 -m Autobs.train_ppo -v single -r coverage --iters 30`

python -m Autobs.train_ppo -v single --dataset_limit 512 --dataset_offset 0 --dataset_stride 1 --checkpoint_dir /Users/epiphanyer/Desktop/coding/paper_experiment/Autobs/checkpoint_test

3. `python3 -m Autobs.train_ppo -v multi -r score --city_map_path /abs/path/to/maps --dataset_limit 64 --iters 50`
4. `python3 -m Autobs.train_ppo --resume /abs/path/to/checkpoint --checkpoint_dir /abs/path/to/output`

参数含义:
- `-v, --version`: 训练模式，`single` 表示单站点，`multi` 表示多站点。
- `-r, --reward_type`: 奖励类型，支持 `coverage`、`spectral_efficiency`、`score`，`capacity` 作为兼容别名映射到 `spectral_efficiency`。
- `--city_map_path`: 城市地图文件或目录路径；默认使用 `paper_experiment/dataset/png/buildingsWHeight`。
- `--dataset_limit`: 仅使用数据集前 N 张图进行训练；默认不截断，使用整个目录。
- `--dataset_offset`: 从排序后的数据集第几个样本开始取。
- `--dataset_stride`: 从排序后的数据集按步长抽样，便于构造稀疏子集。
- `--iters`: PPO 训练迭代次数。
- `--checkpoint_dir`: checkpoint 输出目录；默认写入 `Autobs/checkpoints`。
- `--resume`: 从已有 RLlib checkpoint 恢复训练。
- `--metrics_path`: 奖励历史输出文件；不传则默认写入 checkpoint 目录下的 `reward_history.jsonl`。
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from Autobs.paths import CHECKPOINT_DIR, DEFAULT_DATASET_MAP_DIR, ensure_runtime_dirs


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
        choices=["coverage", "capacity", "spectral_efficiency", "channel_capacity", "score"],
        default="coverage",
    )
    parser.add_argument("--city_map_path", default=str(DEFAULT_DATASET_MAP_DIR))
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--dataset_offset", type=int, default=0)
    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--checkpoint_dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--metrics_path", default=None)
    return parser


def load_environment_class(version: str):
    module_name = ENV_MODULES[version]
    module = importlib.import_module(module_name)
    return module.BaseEnvironment


def build_agent(args: argparse.Namespace):
    environment_class = load_environment_class(args.version)
    if args.resume:
        from ray.rllib.algorithms.algorithm import Algorithm

        agent = Algorithm.from_checkpoint(str(Path(args.resume).expanduser().resolve()))
    else:
        from Autobs.ppo_config import get_ppo_config

        agent = get_ppo_config(environment_class)

    agent.config.env_config.update(
        {
            "reward_type": args.reward_type,
            "city_map_path": args.city_map_path,
            "dataset_limit": args.dataset_limit,
            "dataset_offset": args.dataset_offset,
            "dataset_stride": args.dataset_stride,
        }
    )
    return agent


def run_training(args: argparse.Namespace) -> None:
    ensure_runtime_dirs()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path).expanduser().resolve() if args.metrics_path else checkpoint_dir / "reward_history.jsonl"

    agent = build_agent(args)
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
