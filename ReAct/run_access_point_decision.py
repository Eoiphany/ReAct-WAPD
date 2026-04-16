"""
用途:
  无线接入点决策主入口。读取建筑高度图和自然语言需求，使用 heuristic、random、openai、qwen 或本地微调规划器逐步输出站点动作，并把轨迹写到相对路径目录。

示例命令:
  启发式决策:
    python ReAct/run_access_point_decision.py \
      --city-map-path ../test/dataset/png/buildingsWHeight/0.png \
      --user-request "Improve coverage to 95% with at most 20 sites." \
      --planner heuristic
  OpenAI 决策:
    python ReAct/run_access_point_decision.py \
      --city-map-path ../test/dataset/png/buildingsWHeight/0.png \
      --user-request-path requests/task1.txt \
      --planner openai \
      --openai-api-key $OPENAI_API_KEY \
      --openai-model gpt-4o-mini
  本地 Qwen 决策:
    python ReAct/run_access_point_decision.py \
      --city-map-path ../test/dataset/png/buildingsWHeight/0.png \
      --user-request-path requests/task1.txt \
      --planner qwen \
      --qwen-model-path Qwen/Qwen2.5-7B

参数说明:
  --city-map-path: 输入建筑高度图路径。
  --user-request: 直接传入自然语言需求。
  --user-request-path: 从文本文件读取自然语言需求；优先级高于 --user-request。
  --planner: 决策器类型，heuristic、heuristic_greedy、heuristic_sa、heuristic_ga、heuristic_pso、heuristic_bruteforce、heuristic_full_enum、random、openai、qwen、llamafactory。
  --prompt-path: prompt 配置 JSON 路径。
  --prompt-key: prompt JSON 中的键名；当 --llm-decision-mode=explain_weighted 且沿用默认值时，会自动切到 explain_weighted prompt。
  --traj-dir: 轨迹输出目录。
  --traj-id: 可选轨迹文件名，不含后缀。
  --max-steps: 最大决策步数。
  --auto-steps: 根据需求文本自动推断步数上限。
  --candidate-sample: 环境 observation 中保留的候选点数量。
  --llm-top-k-candidates: 给 OpenAI 压缩 observation 时保留的候选点数量。
  --llm-decision-mode: LLM 决策模式，decide 为直接输出动作，explain_weighted 为输出解释+权重后由程序选动作。
  --heuristic-max-evals: 外部启发式优化脚本允许的最大评估次数。
  --heuristic-candidate-stride: 候选站点采样步长，供 greedy / bruteforce 使用。
  --heuristic-candidate-limit: 候选站点上限，供 greedy / bruteforce 使用。
  --eval-model: 评估模型，pmnet、rmnet 或 proxy。
  --init-mode: 初始站点模式，none、random、greedy、ppo。
  --init-k: random 初始化时采样几个站点。
  --seed: 随机种子。
  --openai-api-key: OpenAI API Key；也可用环境变量 OPENAI_API_KEY。
  --openai-model: OpenAI 模型名。
  --openai-base-url: OpenAI 接口地址。
  --openai-response-format: none 或 json_object。
  --qwen-model-path: 本地 Qwen 模型目录。
  --qwen-device: 本地 Qwen 推理设备，auto/cpu/cuda/mps。
  --qwen-dtype: 本地 Qwen 推理精度，auto/float16/bfloat16/float32。
  --qwen-max-new-tokens: 本地 Qwen 最大生成 token 数。
  --print-step: 打印每一步动作和指标。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from decision_core import (
    build_openai_messages,
    call_openai_chat,
    compact_obs_for_llm_decide,
    compute_dynamic_metric_weights,
    extract_rationale,
    extract_rationale_weights_fallback,
    extract_selected_action,
    extract_weights,
    infer_max_steps,
    infer_request_overrides,
    init_locs_greedy,
    init_locs_from_ppo,
    init_locs_random,
    load_prompt,
    normalize_metric_weights,
    parse_decide_payload,
    plan_action_heuristic as plan_action_react_heuristic,
    plan_action_random,
    recover_direct_action_from_text,
    repair_action_with_candidates,
    select_best_candidate_with_weights,
    score_candidates,
    validate_action,
)
from env_utils import default_redundancy_target
from qwen_adapter import call_qwen_chat
from radiomap_env import RadioMapEnv, build_candidates
from surrogate_adapter import infer_surrogate
from wrappers import HistoryWrapper, LoggingWrapper
from heuristic.run_heuristic_baseline import plan_action_heuristic
from heuristic.optimizer_bridge import next_action_from_target_layout, solve_target_layout


ROOT_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT_DIR / "base_config.yaml").read_text(encoding="utf-8")) or {}
PATH_CFG = CONFIG.get("paths", {}) if isinstance(CONFIG, dict) else {}
DEFAULT_PROMPT_PATH = (ROOT_DIR / PATH_CFG.get("prompt_path", "prompts/radiomap.json")).resolve()
DEFAULT_TRAJ_DIR = (ROOT_DIR / PATH_CFG.get("traj_dir", "outputs/trajs")).resolve()
PPO_CFG = CONFIG.get("ppo", {}) if isinstance(CONFIG, dict) else {}
LLAMAFACTORY_CFG = CONFIG.get("llamafactory", {}) if isinstance(CONFIG, dict) else {}
QWEN_CFG = CONFIG.get("qwen", {}) if isinstance(CONFIG, dict) else {}


_llamafactory_chat_model = None


def _load_llamafactory_chat_model(args: argparse.Namespace):
    global _llamafactory_chat_model
    if _llamafactory_chat_model is not None:
        return _llamafactory_chat_model
    if args.llamafactory_root:
        lf_root = os.path.abspath(args.llamafactory_root)
        lf_src = os.path.join(lf_root, "src")
        if lf_src not in os.sys.path:
            os.sys.path.append(lf_src)
    try:
        from llamafactory.chat import ChatModel
    except ImportError as exc:
        raise ImportError(
            "Failed to import LLaMA-Factory. Set --llamafactory_root to its repo root or ensure it is installed."
        ) from exc
    lf_args = {
        "model_name_or_path": args.llamafactory_model,
        "adapter_name_or_path": args.llamafactory_adapter,
        "finetuning_type": "lora",
        "template": args.llamafactory_template,
        "infer_backend": args.llamafactory_backend,
        "infer_dtype": args.llamafactory_dtype,
    }
    _llamafactory_chat_model = ChatModel(lf_args)
    return _llamafactory_chat_model


def _sanitize_id(text: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())
    return cleaned.strip("-") or "unknown"


def _build_file_id(args: argparse.Namespace, city_map_path: str, user_request_path: Optional[str]) -> str:
    if args.traj_id:
        return _sanitize_id(args.traj_id)
    map_id = _sanitize_id(Path(city_map_path).stem)
    req_id = _sanitize_id(Path(user_request_path).stem) if user_request_path else "inline"
    return f"{map_id}__{req_id}"


def _make_action_json(action: Dict[str, Any], rationale: str, parsed_request: Dict[str, Any]) -> str:
    payload = {
        "parsed_request": parsed_request,
        "rationale": rationale,
        "selected_action": action,
    }
    return f"DECIDE[{json.dumps(payload, ensure_ascii=True)}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-map-path", required=True)
    parser.add_argument("--user-request", default="")
    parser.add_argument("--user-request-path", default="")
    parser.add_argument(
        "--planner",
        choices=[
            "heuristic",
            "heuristic_greedy",
            "heuristic_sa",
            "heuristic_ga",
            "heuristic_pso",
            "heuristic_bruteforce",
            "heuristic_full_enum",
            "random",
            "openai",
            "qwen",
            "llamafactory",
        ],
        default="heuristic",
    )
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--prompt-key", default="react_radiomap_decide")
    parser.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR))
    parser.add_argument("--traj-id", default="")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--auto-steps", action="store_true")
    parser.add_argument("--candidate-sample", type=int, default=64)
    parser.add_argument("--llm-top-k-candidates", type=int, default=16)
    parser.add_argument("--llm-decision-mode", choices=["decide", "explain_weighted"], default="decide")
    parser.add_argument("--heuristic-max-evals", type=int, default=200)
    parser.add_argument("--heuristic-candidate-stride", type=int, default=8)
    parser.add_argument("--heuristic-candidate-limit", type=int, default=500)
    parser.add_argument("--eval-model", choices=["pmnet", "rmnet", "proxy"], default="pmnet")
    parser.add_argument("--init-mode", choices=["none", "random", "greedy", "ppo"], default="none")
    parser.add_argument("--init-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-base-url", default="https://api.openai.com")
    parser.add_argument("--openai-response-format", choices=["none", "json_object"], default="none")
    parser.add_argument("--qwen-model-path", default=str((ROOT_DIR.parent / QWEN_CFG.get("model_path", "Qwen/Qwen2.5-7B")).resolve()))
    parser.add_argument("--qwen-device", choices=["auto", "cpu", "cuda", "mps"], default=str(QWEN_CFG.get("device", "auto")))
    parser.add_argument("--qwen-dtype", choices=["auto", "float16", "bfloat16", "float32"], default=str(QWEN_CFG.get("dtype", "auto")))
    parser.add_argument("--qwen-max-new-tokens", type=int, default=int(QWEN_CFG.get("max_new_tokens", 320)))
    parser.add_argument("--llamafactory-root", default=str(LLAMAFACTORY_CFG.get("root", "")))
    parser.add_argument("--llamafactory-model", default=str(LLAMAFACTORY_CFG.get("model", "")))
    parser.add_argument("--llamafactory-adapter", default=str(LLAMAFACTORY_CFG.get("adapter", "")))
    parser.add_argument("--llamafactory-template", default=str(LLAMAFACTORY_CFG.get("template", "qwen")))
    parser.add_argument("--llamafactory-backend", default=str(LLAMAFACTORY_CFG.get("backend", "huggingface")))
    parser.add_argument("--llamafactory-dtype", default=str(LLAMAFACTORY_CFG.get("dtype", "auto")))
    parser.add_argument("--ppo-checkpoint", default=str((ROOT_DIR / PPO_CFG.get("checkpoint_path", "checkpoints")).resolve()))
    parser.add_argument("--ppo-version", choices=["single", "multi"], default=str(PPO_CFG.get("version", "single")))
    parser.add_argument("--ppo-init-k", type=int, default=int(PPO_CFG.get("init_k", 1)))
    parser.add_argument("--print-step", action="store_true")
    args = parser.parse_args()
    if args.llm_decision_mode == "explain_weighted" and args.prompt_key == "react_radiomap_decide":
        args.prompt_key = "react_radiomap_explain_weighted"

    user_request_path = args.user_request_path or None
    user_request = args.user_request
    if user_request_path:
        user_request = Path(user_request_path).read_text(encoding="utf-8").strip()

    goal = {"primary": "maximize_coverage", "targets": {"coverage_pct": 0.95, "redundancy_rate": default_redundancy_target()}}
    constraints = {"site_limit": 20}
    inferred_goal, inferred_constraints, objective = infer_request_overrides(user_request)
    goal["primary"] = inferred_goal.get("primary", goal["primary"])
    goal_targets = inferred_goal.get("targets", {})
    if isinstance(goal_targets, dict):
        goal["targets"].update(goal_targets)
    constraints.update(inferred_constraints)

    if args.auto_steps:
        args.max_steps = infer_max_steps(user_request, base=args.max_steps)

    pmnet_fn = None if args.eval_model == "pmnet" else (lambda inputs: infer_surrogate(inputs, model_type=args.eval_model))
    init_locs = []
    if args.init_mode == "random":
        init_locs = init_locs_random(args.city_map_path, args.seed, k=args.init_k)
    elif args.init_mode == "greedy":
        init_locs = init_locs_greedy(
            city_map_path=args.city_map_path,
            goal=goal,
            constraints=constraints,
            user_request=user_request,
            candidate_sample=args.candidate_sample,
            seed=args.seed,
            objective=objective,
            pmnet=pmnet_fn,
        )
    elif args.init_mode == "ppo":
        init_locs = init_locs_from_ppo(
            city_map_path=args.city_map_path,
            checkpoint=args.ppo_checkpoint,
            version=args.ppo_version,
            top_k=args.ppo_init_k,
        )

    base_env = RadioMapEnv(
        city_map_path=args.city_map_path,
        goal=goal,
        constraints=constraints,
        user_request=user_request,
        init_locs=init_locs,
        pmnet=pmnet_fn,
        candidate_sample=args.candidate_sample,
        seed=args.seed,
    )
    prompt_text = load_prompt(args.prompt_path, args.prompt_key)
    file_id = _build_file_id(args, args.city_map_path, user_request_path)
    env = HistoryWrapper(
        LoggingWrapper(base_env, folder=args.traj_dir, file_id=file_id),
        obs_format="history",
        prompt=prompt_text,
    )

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    candidates = build_candidates(base_env.pixel_map)
    rng = np.random.default_rng(args.seed)
    heuristic_target_layout = None
    heuristic_target_metrics = None

    for step_idx in range(args.max_steps):
        obs_json = json.loads(env.env.traj["observations"][-1])
        scored = score_candidates(base_env, candidates, objective)
        scored_sorted = sorted(scored, key=lambda c: float(c.get("score", -1e9)), reverse=True)
        obs_json["candidates"] = scored_sorted[: max(1, int(args.llm_top_k_candidates))]
        env.env.traj["observations"][-1] = json.dumps(obs_json, ensure_ascii=True)
        obs = env.observation(env.env.traj["observations"][-1])

        parsed_request = {"goal": goal, "constraints": constraints, "evaluation_criteria": []}
        rationale = ""
        if args.planner == "heuristic":
            selected_action = plan_action_heuristic(
                base_env,
                candidates,
                args.candidate_sample,
                args.seed + step_idx,
                objective=objective,
            )
            rationale = "Heuristic planner selected the locally best legal action."
        elif args.planner.startswith("heuristic_"):
            if heuristic_target_layout is None:
                cache_dir = ROOT_DIR / "outputs" / "heuristic_cache" / f"{file_id}__{args.planner}"
                heuristic_target_layout, heuristic_target_metrics = solve_target_layout(
                    planner_name=args.planner,
                    height_map_path=args.city_map_path,
                    goal=goal,
                    constraints=constraints,
                    eval_model=args.eval_model,
                    output_dir=cache_dir,
                    fallback_k=max(1, args.max_steps),
                    max_evals=args.heuristic_max_evals,
                    candidate_stride=args.heuristic_candidate_stride,
                    candidate_limit=args.heuristic_candidate_limit,
                )
            selected_action = next_action_from_target_layout(base_env, heuristic_target_layout)
            rationale = f"{args.planner} planner converted the cached target layout into the next closed-loop action."
        elif args.planner == "random":
            selected_action = plan_action_random(base_env, candidates, rng)
            rationale = "Random planner selected a legal exploratory action."
        elif args.planner == "openai":
            llm_obs = (
                compact_obs_for_llm_decide(env.env.traj["observations"][-1], max_candidates=args.llm_top_k_candidates)
                if args.llm_decision_mode == "decide"
                else env.env.traj["observations"][-1]
            )
            messages = build_openai_messages(prompt_text, llm_obs)
            llm_text = call_openai_chat(
                api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
                model=args.openai_model,
                messages=messages,
                base_url=args.openai_base_url,
                response_format=args.openai_response_format,
            )
            try:
                payload = parse_decide_payload(llm_text)
            except Exception:
                if args.llm_decision_mode != "decide":
                    raise
                recovered = recover_direct_action_from_text(llm_text)
                if recovered is None:
                    raise
                payload = {"selected_action": recovered}
            parsed_request = payload.get("parsed_request", parsed_request)
            rationale = extract_rationale(payload)
            if args.llm_decision_mode == "explain_weighted":
                weights = extract_weights(payload)
                normalized, ok = normalize_metric_weights(weights, compute_dynamic_metric_weights(obs_json))
                if not ok or normalized is None:
                    rationale_fb, weights_fb = extract_rationale_weights_fallback(llm_text)
                    if rationale_fb and not rationale:
                        rationale = rationale_fb
                    normalized, ok = normalize_metric_weights(weights_fb, compute_dynamic_metric_weights(obs_json))
                if not ok or normalized is None:
                    normalized = compute_dynamic_metric_weights(obs_json)
                site_limit = base_env.constraints.get("site_limit")
                if site_limit is not None and len(base_env.tx_locs) >= int(site_limit):
                    selected_action = plan_action_react_heuristic(base_env, candidates, args.candidate_sample, rng, objective=objective)
                else:
                    best, _ = select_best_candidate_with_weights(
                        env=base_env,
                        candidates=obs_json.get("candidates", candidates),
                        weights=normalized,
                        redundancy_target=goal.get("targets", {}).get("redundancy_rate"),
                    )
                    selected_action = {
                        "name": "Propose",
                        "args": {"sites": [{"row": int(best["row"]), "col": int(best["col"]), "z_m": float(best.get("z_m", 3.0))}], "mode": "add"},
                    }
            else:
                selected_action = extract_selected_action(payload)
                selected_action = repair_action_with_candidates(selected_action, obs_json)
        elif args.planner == "qwen":
            if not args.qwen_model_path:
                raise ValueError("--qwen-model-path required when planner=qwen")
            llm_obs = (
                compact_obs_for_llm_decide(env.env.traj["observations"][-1], max_candidates=args.llm_top_k_candidates)
                if args.llm_decision_mode == "decide"
                else env.env.traj["observations"][-1]
            )
            messages = build_openai_messages(prompt_text, llm_obs)
            llm_text = call_qwen_chat(
                model_path=args.qwen_model_path,
                messages=messages,
                device=args.qwen_device,
                dtype=args.qwen_dtype,
                max_new_tokens=args.qwen_max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            try:
                payload = parse_decide_payload(llm_text)
            except Exception:
                if args.llm_decision_mode != "decide":
                    raise
                recovered = recover_direct_action_from_text(llm_text)
                if recovered is None:
                    raise
                payload = {"selected_action": recovered}
            parsed_request = payload.get("parsed_request", parsed_request)
            rationale = extract_rationale(payload)
            if args.llm_decision_mode == "explain_weighted":
                weights = extract_weights(payload)
                normalized, ok = normalize_metric_weights(weights, compute_dynamic_metric_weights(obs_json))
                if not ok or normalized is None:
                    rationale_fb, weights_fb = extract_rationale_weights_fallback(llm_text)
                    if rationale_fb and not rationale:
                        rationale = rationale_fb
                    normalized, ok = normalize_metric_weights(weights_fb, compute_dynamic_metric_weights(obs_json))
                if not ok or normalized is None:
                    normalized = compute_dynamic_metric_weights(obs_json)
                site_limit = base_env.constraints.get("site_limit")
                if site_limit is not None and len(base_env.tx_locs) >= int(site_limit):
                    selected_action = plan_action_react_heuristic(base_env, candidates, args.candidate_sample, rng, objective=objective)
                else:
                    best, _ = select_best_candidate_with_weights(
                        env=base_env,
                        candidates=obs_json.get("candidates", candidates),
                        weights=normalized,
                        redundancy_target=goal.get("targets", {}).get("redundancy_rate"),
                    )
                    selected_action = {
                        "name": "Propose",
                        "args": {"sites": [{"row": int(best["row"]), "col": int(best["col"]), "z_m": float(best.get("z_m", 3.0))}], "mode": "add"},
                    }
            else:
                selected_action = extract_selected_action(payload)
                selected_action = repair_action_with_candidates(selected_action, obs_json)
        else:
            if not args.llamafactory_model:
                raise ValueError("--llamafactory-model required when planner=llamafactory")
            llm_obs = (
                compact_obs_for_llm_decide(env.env.traj["observations"][-1], max_candidates=args.llm_top_k_candidates)
                if args.llm_decision_mode == "decide"
                else env.env.traj["observations"][-1]
            )
            chat_model = _load_llamafactory_chat_model(args)
            messages = build_openai_messages(prompt_text, llm_obs)
            responses = chat_model.chat(messages[1:], system=messages[0]["content"], do_sample=False, temperature=0.0, top_p=1.0, top_k=1, max_new_tokens=320)
            if not responses:
                raise RuntimeError("LLaMAFactory returned empty response")
            llm_text = responses[0].response_text
            try:
                payload = parse_decide_payload(llm_text)
            except Exception:
                if args.llm_decision_mode != "decide":
                    raise
                recovered = recover_direct_action_from_text(llm_text)
                if recovered is None:
                    raise
                payload = {"selected_action": recovered}
            parsed_request = payload.get("parsed_request", parsed_request)
            rationale = extract_rationale(payload)
            if args.llm_decision_mode == "explain_weighted":
                weights = extract_weights(payload)
                normalized, ok = normalize_metric_weights(weights, compute_dynamic_metric_weights(obs_json))
                if not ok or normalized is None:
                    rationale_fb, weights_fb = extract_rationale_weights_fallback(llm_text)
                    if rationale_fb and not rationale:
                        rationale = rationale_fb
                    normalized, ok = normalize_metric_weights(weights_fb, compute_dynamic_metric_weights(obs_json))
                if not ok or normalized is None:
                    normalized = compute_dynamic_metric_weights(obs_json)
                site_limit = base_env.constraints.get("site_limit")
                if site_limit is not None and len(base_env.tx_locs) >= int(site_limit):
                    selected_action = plan_action_react_heuristic(base_env, candidates, args.candidate_sample, rng, objective=objective)
                else:
                    best, _ = select_best_candidate_with_weights(
                        env=base_env,
                        candidates=obs_json.get("candidates", candidates),
                        weights=normalized,
                        redundancy_target=goal.get("targets", {}).get("redundancy_rate"),
                    )
                    selected_action = {
                        "name": "Propose",
                        "args": {"sites": [{"row": int(best["row"]), "col": int(best["col"]), "z_m": float(best.get("z_m", 3.0))}], "mode": "add"},
                    }
            else:
                selected_action = extract_selected_action(payload)
                selected_action = repair_action_with_candidates(selected_action, obs_json)

        ok, reasons = validate_action(base_env, selected_action, obs_json)
        if not ok:
            raise RuntimeError(f"Invalid selected action: {reasons}")

        action_text = _make_action_json(selected_action, rationale, parsed_request)
        env.env.last_rationale = rationale
        obs, reward, terminated, truncated, info = env.step(action_text)

        if args.print_step:
            metrics = base_env.last_metrics
            print(
                json.dumps(
                    {
                        "step": step_idx,
                        "action": selected_action,
                        "coverage": None if metrics is None else float(metrics.coverage),
                        "capacity": None if metrics is None else float(metrics.capacity),
                        "redundancy_rate": None if metrics is None else float(metrics.redundancy_rate),
                        "done": bool(terminated or truncated),
                    },
                    ensure_ascii=False,
                )
            )

        if terminated or truncated:
            break

    env.env.write()


if __name__ == "__main__":
    main()
