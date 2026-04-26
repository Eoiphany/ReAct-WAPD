"""注释
命令示例:
1. 按老仓库 `radiomap_explain_weighted.json` 的生成方式构建当前子项目版本:
python -m ReAct.build_explain_weighted_dataset \
  --traj-dir /Users/epiphanyer/Desktop/coding/ReAct/trajs_heuristic \
  --output /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/train_data/radiomap_explain_weighted.json

2. 合并多个轨迹目录后再构建一份数据集:!!!用这个
python -m ReAct.build_explain_weighted_dataset \
  --traj-dir /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/trajs_llamafactory_v4 \
  --traj-dir /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/outputs/trajs_heuristic_sa \
  --output /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/train_data/radiomap_explain_weighted_merged.json

3. 只抽取前 100 条样本做快速检查:
python -m ReAct.build_explain_weighted_dataset \
  --traj-dir /Users/epiphanyer/Desktop/coding/ReAct/trajs_llamafactory_v4 \
  --output /Users/epiphanyer/Desktop/coding/paper_experiment/ReAct/train_data/radiomap_explain_weighted_smoke.json \
  --max-samples 100

参数含义:
- --traj-dir: 轨迹目录，可重复传入多次；脚本会顺序读取其中的 `*.json` 轨迹文件。
- --prompt-path: prompt JSON 路径；默认读取当前 ReAct 子项目的 `prompts/radiomap.json`。
- --prompt-key: prompt 键名；默认使用 `react_radiomap_explain_weighted_json`。
- --output: 输出数据路径；支持 `.json` 或 `.jsonl`。
- --output-mode: `rationale` 表示只学习原因；`rationale_weights` 表示学习原因+权重。
- --max-trajs: 最多读取多少条轨迹；0 表示不限制。
- --max-steps: 每条轨迹最多取多少步；0 表示不限制。
- --max-samples: 全局最多输出多少条样本；0 表示不限制。
- --no-clean-nan: 不把 observation 中的 `NaN/Infinity` 清洗为 `null`。

脚本逻辑说明:
该脚本同时兼容两条旧数据链路：
1. `radiomap_interpret.json` 风格：把当前 `chosen_action` 写回 observation，只学习 `{"rationale": "..."}`。
2. `radiomap_explain_weighted.json` 风格：基于当前 observation 重新计算 `weights`，输出 `{"rationale": "...", "weights": {...}}`。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = ROOT_DIR / "prompts" / "radiomap.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "train_data" / "radiomap_explain_weighted.json"


def load_prompt(path: str | Path, key: str) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if key not in data:
        raise KeyError(f"prompt key not found: {key}")
    return str(data[key])


def clean_nan(text: str) -> str:
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\bInfinity\b", "null", text)
    text = re.sub(r"\b-Infinity\b", "null", text)
    return text


def _extract_balanced_json(text: str, start: int) -> str:
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _parse_decide_payload(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    raw = _extract_balanced_json(text, start)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_selected_action(action_text: str) -> Dict[str, Any] | None:
    payload = _parse_decide_payload(action_text)
    if not isinstance(payload, dict):
        return None
    selected = payload.get("selected_action")
    return selected if isinstance(selected, dict) else None


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w_cov = float(weights.get("w_cov", 1.0))
    w_cap = float(weights.get("w_cap", 0.2))
    w_sites = float(weights.get("w_sites", 0.0))
    total = w_cov + w_cap + w_sites
    if total <= 0:
        return {"w_cov": 1.0, "w_cap": 0.2, "w_sites": 0.0}
    return {
        "w_cov": w_cov / total,
        "w_cap": w_cap / total,
        "w_sites": w_sites / total,
    }


def _weights_from_observation(obs_json: str) -> Dict[str, float]:
    # 这里保持与老仓库 build_finetune_data.py 的动态权重规则一致。
    try:
        obs = json.loads(obs_json)
    except Exception:
        return _normalize_weights({"w_cov": 1.0, "w_cap": 0.2, "w_sites": 0.0})

    goal = obs.get("goal", {}) or {}
    constraints = obs.get("constraints", {}) or {}
    state = obs.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}

    cov = metrics.get("coverage")
    cap = metrics.get("capacity")
    site_count = int(state.get("site_count") or 0)

    targets = goal.get("targets", {}) or {}
    cov_t = targets.get("coverage_pct")
    cap_t = targets.get("capacity")
    site_limit = constraints.get("site_limit")

    miss_cov = cov_t is not None and cov is not None and float(cov) < float(cov_t)
    miss_cap = cap_t is not None and cap is not None and float(cap) < float(cap_t)
    over_site = site_limit is not None and site_count > int(site_limit)

    w_cov = 1.0
    w_cap = 0.2
    w_sites = 0.0

    if miss_cov:
        w_cov = max(w_cov, 1.2)
        w_cap = max(w_cap, 0.3)
    if miss_cap:
        w_cap = max(w_cap, 1.0)
        w_cov = max(w_cov, 0.3)
    if over_site:
        w_sites = max(w_sites, 0.2)
    if not miss_cov and not miss_cap:
        w_sites = max(w_sites, 0.05)

    return _normalize_weights({"w_cov": w_cov, "w_cap": w_cap, "w_sites": w_sites})


def build_history(observations: List[str], actions: List[str], step_idx: int) -> str:
    observation = observations[step_idx]
    if step_idx > 0 and (step_idx - 1) < len(actions):
        observation = observation.rstrip("\n")
        observation += f"\nChosenAction: {actions[step_idx - 1]}\n"
    return observation


def build_reason_input(observations: List[str], actions: List[str], step_idx: int) -> str:
    obs_text = observations[step_idx]
    try:
        obs_payload = json.loads(obs_text)
    except Exception:
        selected_action = _extract_selected_action(actions[step_idx]) if step_idx < len(actions) else None
        if selected_action is None:
            return obs_text
        return f"ChosenAction: {json.dumps(selected_action, ensure_ascii=True)}\n{obs_text}"

    selected_action = _extract_selected_action(actions[step_idx]) if step_idx < len(actions) else None
    if selected_action is not None:
        obs_payload["chosen_action"] = selected_action
    return json.dumps(obs_payload, ensure_ascii=True)


def iter_trajs(traj_dirs: List[Path]) -> Iterable[Dict[str, List[str]]]:
    for traj_dir in traj_dirs:
        for path in sorted(traj_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "observations" in item and "actions" in item:
                        yield item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", action="append", required=True)
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--prompt-key", default="react_radiomap_explain_weighted_json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-mode", choices=["rationale", "rationale_weights"], default="rationale_weights")
    parser.add_argument("--max-trajs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-clean-nan", action="store_true")
    return parser


def build_dataset(
    traj_dirs: List[str | Path],
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    prompt_key: str = "react_radiomap_explain_weighted_json",
    output: str | Path = DEFAULT_OUTPUT_PATH,
    output_mode: str = "rationale_weights",
    max_trajs: int = 0,
    max_steps: int = 0,
    max_samples: int = 0,
    no_clean_nan: bool = False,
) -> Path:
    traj_dirs = [Path(item).expanduser().resolve() for item in traj_dirs]
    for traj_dir in traj_dirs:
        if not traj_dir.exists():
            raise FileNotFoundError(f"traj_dir not found: {traj_dir}")

    prompt_text = load_prompt(prompt_path, prompt_key)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl = output_path.suffix == ".jsonl"
    if not write_jsonl and output_path.suffix != ".json":
        raise SystemExit("output must end with .json or .jsonl")

    total = 0
    records: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for traj_idx, traj in enumerate(iter_trajs(traj_dirs)):
            if max_trajs and traj_idx >= max_trajs:
                break

            observations = traj.get("observations", [])
            actions = traj.get("actions", [])
            rationales = traj.get("rationales", []) or []
            if not no_clean_nan:
                observations = [clean_nan(obs) for obs in observations]

            steps = min(len(actions), len(observations) - 1, len(rationales))
            if max_steps:
                steps = min(steps, max_steps)

            for step_idx in range(steps):
                if output_mode == "rationale":
                    history = build_reason_input(observations, actions, step_idx)
                    output_text = json.dumps({"rationale": rationales[step_idx]}, ensure_ascii=True)
                else:
                    history = build_history(observations, actions, step_idx)
                    output_text = json.dumps(
                        {
                            "rationale": rationales[step_idx],
                            "weights": _weights_from_observation(observations[step_idx]),
                        },
                        ensure_ascii=True,
                    )
                record = {
                    "instruction": prompt_text,
                    "input": history,
                    "output": output_text,
                }
                if write_jsonl:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    records.append(record)
                total += 1
                if max_samples and total >= max_samples:
                    break
            if max_samples and total >= max_samples:
                break

        if not write_jsonl:
            handle.seek(0)
            handle.write(json.dumps(records, ensure_ascii=False, indent=2))
            handle.truncate()

    print(f"Wrote {total} samples to {output_path}")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    build_dataset(
        traj_dirs=args.traj_dir,
        prompt_path=args.prompt_path,
        prompt_key=args.prompt_key,
        output=args.output,
        output_mode=args.output_mode,
        max_trajs=args.max_trajs,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        no_clean_nan=args.no_clean_nan,
    )


if __name__ == "__main__":
    main()
