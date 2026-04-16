"""
用途:
  无线接入点决策的核心逻辑：请求解析、候选点评分、启发式规划、OpenAI 调用、动作解析与合法性检查。

示例命令:
  无。该文件是公共模块，供主入口导入。

参数说明:
  infer_request_overrides(user_request): 从自然语言提取 goal / constraints / objective。
  plan_action_heuristic(...): 根据当前环境和候选点输出单步动作。
  plan_action_random(...): 输出随机合法动作。
  call_openai_chat(...): 调用 OpenAI Chat Completions。
  parse_decide_payload(text): 从 LLM 输出中抽取 DECIDE JSON。
  validate_action(...): 检查动作是否合法并返回原因。
"""

from __future__ import annotations

import ast
import copy
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from env_utils import (
    calc_action_mask,
    calc_upsampling_loc,
    default_redundancy_target,
    load_map_normalized,
    map_size,
    normalize_redundancy_target,
    redundancy_balance_score,
)
from radiomap_env import RadioMapEnv, build_candidates, height_from_gray


def load_prompt(path: str, key: str) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if key not in data:
        raise KeyError(f"prompt key not found: {key}")
    return data[key]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except Exception:
        return default
    if np.isnan(num) or np.isinf(num):
        return default
    return num


def _sites_from_locs(env: RadioMapEnv, locs: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    sites = []
    for r, c in locs:
        sites.append({"row": int(r), "col": int(c), "z_m": float(height_from_gray(env.pixel_map[r, c]))})
    return sites


def _objective_score(
    objective: Dict[str, float],
    cov: float,
    cap: float,
    site_count: int,
    redundancy_rate: float = 0.0,
    redundancy_target: Optional[Dict[str, float]] = None,
) -> float:
    return (
        float(objective.get("w_cov", 1.0)) * cov
        + float(objective.get("w_cap", 0.2)) * cap
        + float(objective.get("w_red", 0.0)) * redundancy_balance_score(redundancy_rate, redundancy_target)
        - float(objective.get("w_sites", 0.0)) * site_count
    )


def normalize_metric_weights(weights: Optional[Dict[str, Any]], fallback: Optional[Dict[str, float]] = None) -> Tuple[Optional[Dict[str, float]], bool]:
    fallback = fallback or {"w_cov": 1.0, "w_cap": 0.2, "w_red": 0.2, "w_sites": 0.0}
    if not isinstance(weights, dict):
        return None, False
    try:
        w_cov = float(weights.get("w_cov", fallback.get("w_cov", 1.0)))
        w_cap = float(weights.get("w_cap", fallback.get("w_cap", 0.2)))
        w_red = float(weights.get("w_red", fallback.get("w_red", 0.2)))
        w_sites = float(weights.get("w_sites", fallback.get("w_sites", 0.0)))
    except Exception:
        return None, False
    if min(w_cov, w_cap, w_red, w_sites) < 0:
        return None, False
    total = w_cov + w_cap + w_red + w_sites
    if total <= 0:
        return None, False
    return {
        "w_cov": w_cov / total,
        "w_cap": w_cap / total,
        "w_red": w_red / total,
        "w_sites": w_sites / total,
    }, True


def compute_dynamic_metric_weights(observation: Any) -> Dict[str, float]:
    if isinstance(observation, str):
        try:
            data = json.loads(observation)
        except Exception:
            data = {}
    elif isinstance(observation, dict):
        data = observation
    else:
        data = {}

    goal = data.get("goal", {}) or {}
    constraints = data.get("constraints", {}) or {}
    state = data.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}
    diagnosis = data.get("diagnosis", {}) or {}
    margins = diagnosis.get("margins", {}) or {}

    target_cov = goal.get("targets", {}).get("coverage_pct")
    target_cap = goal.get("targets", {}).get("capacity")
    target_red = normalize_redundancy_target(goal.get("targets", {}).get("redundancy_rate"))

    cov = _safe_float(metrics.get("coverage"), 0.0)
    cap = _safe_float(metrics.get("capacity"), 0.0)
    red = _safe_float(metrics.get("redundancy_rate"), target_red["ideal"])
    site_count = int(state.get("site_count") or 0)
    site_limit = constraints.get("site_limit")

    cov_gap = _safe_float(margins.get("coverage_gap"), max(0.0, float(target_cov) - cov) if target_cov is not None else 0.0)
    cap_gap = _safe_float(margins.get("capacity_gap"), max(0.0, float(target_cap) - cap) if target_cap is not None else 0.0)
    red_gap = _safe_float(
        margins.get("redundancy_gap"),
        max(0.0, abs(red - float(target_red["ideal"])) - float(target_red["tolerance"])),
    )
    site_over = max(0, int(margins.get("site_over") or 0))
    if site_limit is not None:
        site_over = max(site_over, max(0, site_count - int(site_limit)))

    raw = {
        "w_cov": 1.0,
        "w_cap": 0.2,
        "w_red": 0.2,
        "w_sites": 0.0,
    }
    if cov_gap > 0:
        raw["w_cov"] = max(raw["w_cov"], 1.0 + cov_gap * 4.0)
    if cap_gap > 0:
        raw["w_cap"] = max(raw["w_cap"], 0.5 + cap_gap * 2.5)
    if red_gap > 0:
        raw["w_red"] = max(raw["w_red"], 0.6 + red_gap * 4.0)
    if site_over > 0:
        raw["w_sites"] = max(raw["w_sites"], 0.15 + 0.05 * site_over)
    if cov_gap <= 0 and cap_gap <= 0 and red_gap <= 0:
        raw["w_sites"] = max(raw["w_sites"], 0.05)

    weights, ok = normalize_metric_weights(raw)
    if ok and weights is not None:
        return weights
    return {"w_cov": 0.6, "w_cap": 0.15, "w_red": 0.15, "w_sites": 0.10}


def infer_request_overrides(user_request: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    text = user_request or ""
    lower = text.lower()
    goal: Dict[str, Any] = {"primary": "maximize_coverage", "targets": {}}
    constraints: Dict[str, Any] = {}
    objective = {"w_cov": 1.0, "w_cap": 0.2, "w_red": 0.0, "w_sites": 0.0}
    fixed_site_count: Optional[int] = None
    goal["targets"]["redundancy_rate"] = default_redundancy_target()

    if ("capacity" in lower) or ("容量" in text):
        goal["primary"] = "maximize_capacity"
        objective["w_cap"] = 0.8
        objective["w_cov"] = 0.4

    if ("coverage" in lower) or ("覆盖" in text):
        if ("capacity" not in lower) and ("容量" not in text):
            goal["primary"] = "maximize_coverage"
            objective["w_cov"] = 1.0
            objective["w_cap"] = 0.2

    if ("cost" in lower) or ("成本" in text) or ("energy" in lower) or ("能耗" in text):
        objective["w_sites"] = max(objective["w_sites"], 0.05)

    if ("redundancy" in lower) or ("冗余" in text):
        goal["primary"] = "balance_redundancy"
        objective["w_red"] = max(objective["w_red"], 0.8)
        objective["w_cov"] = max(objective["w_cov"], 0.4)
        objective["w_cap"] = max(objective["w_cap"], 0.3)

    fixed_patterns = [
        r"(?:fixed|exactly|equal to|set to)\s*([0-9]+)\s*(?:sites|site|aps|ap)",
        r"(?:固定|恰好|正好|限定|给定)\s*([0-9]+)\s*(?:个)?(?:站点|基站|站|小区)",
    ]
    for pattern in fixed_patterns:
        match = re.search(pattern, lower if "site" in pattern else text)
        if match:
            fixed_site_count = int(match.group(1))
            break

    coverage_vals: List[float] = []
    coverage_vals += [
        float(v) / 100.0
        for v in re.findall(r"(?:coverage|覆盖).*?(?:>=|>|at least|至少|不低于|不少于)\s*([0-9]+(?:\.[0-9]+)?)\s*%", lower)
    ]
    if coverage_vals:
        goal["targets"]["coverage_pct"] = max(coverage_vals)
    elif fixed_site_count is not None:
        goal["targets"]["coverage_pct"] = None

    match = re.search(r"(?:capacity|容量).*?(?:>=|>|at least|至少|不低于|不少于)\s*([0-9]+(?:\.[0-9]+)?)", lower)
    if match:
        goal["targets"]["capacity"] = float(match.group(1))
    elif fixed_site_count is not None:
        goal["targets"]["capacity"] = None

    if fixed_site_count is not None:
        constraints["site_exact"] = int(fixed_site_count)
        constraints["site_limit"] = int(fixed_site_count)
    else:
        match = re.search(r"(?:no more than|at most|<=|不超过|上限|最多)\s*([0-9]+)", lower)
        if match and (("site" in lower) or ("站点" in text) or ("cell" in lower)):
            constraints["site_limit"] = int(match.group(1))
        else:
            match = re.search(r"([0-9]+)\s*(?:sites|site|站|小区)", lower)
            if match:
                constraints["site_limit"] = int(match.group(1))
    return goal, constraints, objective


def infer_max_steps(user_request: str, base: int = 5) -> int:
    text = (user_request or "").lower()
    steps = base
    if "capacity" in text and "coverage" in text:
        steps += 3
    if any(k in text for k in ["budget", "cost", "energy", "fairness", "公平", "能耗", "成本"]):
        steps += 2
    return max(3, min(15, steps))


_PPO_AGENT_CACHE: Dict[str, Any] = {}


def _load_ppo_agent(checkpoint: str):
    if checkpoint in _PPO_AGENT_CACHE:
        return _PPO_AGENT_CACHE[checkpoint]
    try:
        from ray.rllib.algorithms.algorithm import Algorithm
    except Exception as exc:
        raise RuntimeError("PPO 初始化需要 ray[rllib] 环境，请先安装依赖。") from exc
    agent = Algorithm.from_checkpoint(str(Path(checkpoint).resolve()))
    _PPO_AGENT_CACHE[checkpoint] = agent
    return agent


def _compute_rllib_logits(agent, observation: Dict[str, Any]) -> np.ndarray:
    try:
        module = agent.get_module()
    except Exception:
        module = agent.get_module("default_policy")
    action_dist_class = module.get_inference_action_dist_cls()
    import torch

    obs = observation["observations"]
    mask = observation["action_mask"]
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
    fwd_ins = {"obs": {"observations": obs_tensor, "action_mask": mask_tensor}}
    fwd_out = module.forward_inference(fwd_ins)
    logits = fwd_out["action_dist_inputs"]
    if hasattr(logits, "detach"):
        return logits[0].detach().cpu().numpy()
    action_dist = action_dist_class.from_logits(logits)
    return action_dist.logits[0].detach().cpu().numpy()


def _resize_map(pixel_map: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if pixel_map.shape == size:
        return pixel_map
    from PIL import Image

    crop_u8 = (pixel_map * 255).astype(np.uint8)
    img = Image.fromarray(crop_u8).resize(size, resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def init_locs_random(city_map_path: str, seed: int, k: int = 1) -> List[Tuple[int, int]]:
    pixel_map = load_map_normalized(city_map_path)
    pixel_map = _resize_map(pixel_map, (map_size, map_size))
    mask = calc_action_mask(pixel_map)
    valid_ids = np.where(mask > 0.5)[0]
    if len(valid_ids) == 0:
        return []
    kk = max(1, min(int(k), int(len(valid_ids))))
    rng = np.random.default_rng(seed)
    chosen = np.atleast_1d(rng.choice(valid_ids, size=kk, replace=False))
    out = []
    for action_id in chosen:
        row, col = calc_upsampling_loc(int(action_id))
        out.append((int(row), int(col)))
    return out


def init_locs_from_ppo(city_map_path: str, checkpoint: str, version: str = "single", top_k: int = 1) -> List[Tuple[int, int]]:
    agent = _load_ppo_agent(checkpoint)
    pixel_map = load_map_normalized(city_map_path)
    pixel_map = _resize_map(pixel_map, (map_size, map_size))
    mask = calc_action_mask(pixel_map)
    flat = pixel_map.reshape(-1).astype(np.float32)
    obs = np.tile(flat, 2) if version == "multi" else flat
    observation = {"observations": obs, "action_mask": mask}
    logits = _compute_rllib_logits(agent, observation)
    logits = np.array(logits, dtype=np.float32)
    logits[mask <= 0.5] = -1e9
    k = min(max(int(top_k), 1), int(np.sum(mask > 0.5)))
    if k <= 0:
        return []
    top_ids = np.argpartition(-logits, k - 1)[:k]
    top_ids = top_ids[np.argsort(-logits[top_ids])]
    locs = []
    for action_id in top_ids:
        row, col = calc_upsampling_loc(int(action_id))
        locs.append((int(row), int(col)))
    return locs


def _extract_first_site_from_action(action: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if not isinstance(action, dict) or action.get("name") != "Propose":
        return None
    args = action.get("args", {}) or {}
    sites = args.get("sites") or []
    if isinstance(sites, list) and sites:
        site = sites[0]
        if isinstance(site, dict) and "row" in site and "col" in site:
            return int(site["row"]), int(site["col"])
    return None


def init_locs_greedy(
    city_map_path: str,
    goal: Dict[str, Any],
    constraints: Dict[str, Any],
    user_request: str,
    candidate_sample: int,
    seed: int,
    objective: Dict[str, float],
    pmnet=None,
) -> List[Tuple[int, int]]:
    env = RadioMapEnv(
        city_map_path=city_map_path,
        goal=goal,
        constraints=constraints,
        user_request=user_request,
        init_locs=[],
        pmnet=pmnet,
        candidate_sample=max(1, int(candidate_sample)),
        seed=seed,
    )
    candidates = build_candidates(env.pixel_map)
    rng = np.random.default_rng(seed)
    action = plan_action_heuristic(env, candidates, max(1, int(candidate_sample)), rng, objective)
    loc = _extract_first_site_from_action(action)
    return [loc] if loc is not None else []


def plan_action_heuristic(
    env: RadioMapEnv,
    candidates: List[Dict[str, Any]],
    sample_k: int,
    rng: np.random.Generator,
    objective: Dict[str, float],
) -> Dict[str, Any]:
    if not candidates:
        return {"name": "Propose", "args": {"sites": [{"row": 0, "col": 0, "z_m": 3.0}], "mode": "add"}}

    if env.last_metrics is None:
        env._evaluate()
    current_cov = _safe_float(env.last_metrics.coverage if env.last_metrics else 0.0)
    current_cap = _safe_float(env.last_metrics.capacity if env.last_metrics else 0.0)
    target_cov = env.goal.get("targets", {}).get("coverage_pct")
    target_cap = env.goal.get("targets", {}).get("capacity")
    site_limit = env.constraints.get("site_limit")
    site_exact = env.constraints.get("site_exact")
    current_locs = list(env.tx_locs)
    site_count = len(current_locs)
    site_limit = None if site_limit is None else int(site_limit)

    def sample_pool():
        if sample_k > 0 and sample_k < len(candidates):
            idx = rng.choice(len(candidates), size=sample_k, replace=False)
            return [candidates[int(i)] for i in idx]
        return candidates

    def score(cov: float, cap: float, count: int, red: float) -> float:
        return _objective_score(
            objective,
            cov,
            cap,
            count,
            redundancy_rate=red,
            redundancy_target=env.goal.get("targets", {}).get("redundancy_rate"),
        )

    current_red = _safe_float(env.last_metrics.redundancy_rate if env.last_metrics else 0.0)
    base_score = score(current_cov, current_cap, len(current_locs), current_red)
    sampled = sample_pool()

    best_add = None
    best_add_score = -1e9
    for cand in sampled:
        env.tx_locs = current_locs + [(cand["row"], cand["col"])]
        metrics = env._evaluate()
        cand_score = score(
            _safe_float(metrics.coverage),
            _safe_float(metrics.capacity),
            len(env.tx_locs),
            _safe_float(metrics.redundancy_rate),
        )
        if cand_score > best_add_score:
            best_add_score = cand_score
            best_add = cand
    env.tx_locs = current_locs
    add_improve = best_add_score - base_score

    best_move = None
    best_move_score = -1e9
    if current_locs:
        for site_idx in range(len(current_locs)):
            for cand in sampled:
                env.tx_locs = list(current_locs)
                env.tx_locs[site_idx] = (cand["row"], cand["col"])
                metrics = env._evaluate()
                cand_score = score(
                    _safe_float(metrics.coverage),
                    _safe_float(metrics.capacity),
                    len(env.tx_locs),
                    _safe_float(metrics.redundancy_rate),
                )
                if cand_score > best_move_score:
                    best_move_score = cand_score
                    best_move = (site_idx, cand)
        env.tx_locs = current_locs

    best_remove_idx = None
    best_remove_score = -1e9
    if len(current_locs) > 1:
        for site_idx in range(len(current_locs)):
            env.tx_locs = [loc for i, loc in enumerate(current_locs) if i != site_idx]
            metrics = env._evaluate()
            cand_score = score(
                _safe_float(metrics.coverage),
                _safe_float(metrics.capacity),
                len(env.tx_locs),
                _safe_float(metrics.redundancy_rate),
            )
            if cand_score > best_remove_score:
                best_remove_score = cand_score
                best_remove_idx = site_idx
        env.tx_locs = current_locs

    cov_ok = target_cov is None or current_cov >= float(target_cov)
    cap_ok = target_cap is None or current_cap >= float(target_cap)

    if cov_ok and cap_ok and add_improve < 0.001 and best_move_score <= base_score:
        metrics = env._evaluate()
        return {
            "name": "Finish",
            "args": {
                "final_site_set": _sites_from_locs(env, current_locs),
                "metrics": {
                    "coverage": float(metrics.coverage),
                    "capacity": float(metrics.capacity),
                    "redundancy_rate": float(metrics.redundancy_rate),
                },
            },
        }

    if site_exact is not None and site_count < int(site_exact) and best_add is not None:
        return {"name": "Propose", "args": {"sites": [{"row": best_add["row"], "col": best_add["col"], "z_m": best_add["z_m"]}], "mode": "add"}}

    if site_limit is not None and site_count >= site_limit and best_move is not None:
        site_idx, cand = best_move
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "move", "id": site_idx, "row": cand["row"], "col": cand["col"]}}}

    if best_move is not None and best_move_score > max(best_add_score, base_score):
        site_idx, cand = best_move
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "move", "id": site_idx, "row": cand["row"], "col": cand["col"]}}}

    if best_add is not None:
        return {"name": "Propose", "args": {"sites": [{"row": best_add["row"], "col": best_add["col"], "z_m": best_add["z_m"]}], "mode": "add"}}

    if best_remove_idx is not None:
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "remove", "id": best_remove_idx}}}

    metrics = env._evaluate()
    return {
        "name": "Finish",
        "args": {
            "final_site_set": _sites_from_locs(env, current_locs),
            "metrics": {
                "coverage": float(metrics.coverage),
                "capacity": float(metrics.capacity),
                "redundancy_rate": float(metrics.redundancy_rate),
            },
        },
    }


def plan_action_random(env: RadioMapEnv, candidates: List[Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
    if not candidates:
        return {"name": "Propose", "args": {"sites": [{"row": 0, "col": 0, "z_m": 3.0}], "mode": "add"}}

    if env.last_metrics is None:
        env._evaluate()
    current_cov = _safe_float(env.last_metrics.coverage if env.last_metrics else 0.0)
    current_cap = _safe_float(env.last_metrics.capacity if env.last_metrics else 0.0)
    target_cov = env.goal.get("targets", {}).get("coverage_pct")
    target_cap = env.goal.get("targets", {}).get("capacity")
    site_limit = env.constraints.get("site_limit")
    site_count = len(env.tx_locs)
    cov_ok = target_cov is None or current_cov >= float(target_cov)
    cap_ok = target_cap is None or current_cap >= float(target_cap)
    if cov_ok and cap_ok:
        metrics = env._evaluate()
        return {
            "name": "Finish",
            "args": {
                "final_site_set": _sites_from_locs(env, list(env.tx_locs)),
                "metrics": {
                    "coverage": float(metrics.coverage),
                    "capacity": float(metrics.capacity),
                    "redundancy_rate": float(metrics.redundancy_rate),
                },
            },
        }

    actions = []
    if site_count > 0:
        actions.append("Move")
    if site_count > 1:
        actions.append("Remove")
    if site_limit is None or site_count < int(site_limit):
        actions.append("Add")
    if not actions:
        actions = ["Add"]

    choice = rng.choice(actions)
    if choice == "Move":
        site_idx = int(rng.integers(0, site_count))
        cand = candidates[int(rng.integers(0, len(candidates)))]
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "move", "id": site_idx, "row": cand["row"], "col": cand["col"]}}}
    if choice == "Remove":
        site_idx = int(rng.integers(0, site_count))
        return {"name": "Refine", "args": {"rule_or_delta": {"op": "remove", "id": site_idx}}}
    cand = candidates[int(rng.integers(0, len(candidates)))]
    return {"name": "Propose", "args": {"sites": [{"row": cand["row"], "col": cand["col"], "z_m": cand["z_m"]}], "mode": "add"}}


def _repair_json_payload(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    open_count = s.count("{")
    close_count = s.count("}")
    if close_count > open_count:
        excess = close_count - open_count
        while excess > 0 and s.endswith("}"):
            s = s[:-1].rstrip()
            excess -= 1
    elif open_count > close_count:
        s = s + ("}" * (open_count - close_count))
    s = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)
    s = re.sub(r',\s*,', ',', s)
    s = re.sub(r',\s*}', '}', s)
    return s


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


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = _repair_json_payload(raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def parse_decide_payload(text: str) -> Dict[str, Any]:
    upper = text.upper()
    idx = upper.find("DECIDE[")
    if idx != -1:
        brace_start = text.find("{", idx)
        if brace_start != -1:
            raw = _extract_balanced_json(text, brace_start)
            obj = _try_parse_json(raw)
            if isinstance(obj, dict):
                return obj
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        raw = _extract_balanced_json(text, i)
        obj = _try_parse_json(raw)
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM output missing DECIDE[...] payload")


def recover_direct_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text:
        return None
    name_match = re.search(r'"selected_action"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if not name_match:
            return None
    name = name_match.group(1).strip()
    if name == "Propose":
        candidate_index_match = re.search(r'"candidate_index"\s*:\s*(-?\d+)', text)
        if candidate_index_match:
            return {"name": "Propose", "args": {"candidate_index": int(candidate_index_match.group(1))}}
    if name == "Refine":
        compact_op = re.search(r'"refine_op"\s*:\s*"([^"]+)"', text)
        if compact_op:
            op = compact_op.group(1).strip()
            if op == "remove":
                id_match = re.search(r'"id"\s*:\s*(-?\d+)', text)
                if id_match:
                    return {"name": "Refine", "args": {"refine_op": "remove", "id": int(id_match.group(1))}}
            if op == "move":
                id_match = re.search(r'"id"\s*:\s*(-?\d+)', text)
                target_match = re.search(r'"target_action_id"\s*:\s*(-?\d+)', text)
                if id_match and target_match:
                    return {
                        "name": "Refine",
                        "args": {
                            "refine_op": "move",
                            "id": int(id_match.group(1)),
                            "target_action_id": int(target_match.group(1)),
                        },
                    }
    if name == "Finish":
        return {"name": "Finish", "args": {}}
    return None


def extract_weights(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    weights = payload.get("weights")
    return weights if isinstance(weights, dict) else None


def extract_rationale_weights_fallback(text: str) -> Tuple[str, Optional[Dict[str, float]]]:
    if not text:
        return "", None
    rationale = ""
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    def _find_num(key: str) -> Optional[float]:
        match = re.search(rf'"{key}"\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    weights = {
        "w_cov": _find_num("w_cov"),
        "w_cap": _find_num("w_cap"),
        "w_red": _find_num("w_red"),
        "w_sites": _find_num("w_sites"),
    }
    if any(value is None for value in weights.values()):
        return rationale, None
    return rationale, {key: float(value) for key, value in weights.items() if value is not None}


def _candidate_stats(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    scores = []
    for cand in candidates:
        if "score" in cand:
            try:
                scores.append(float(cand["score"]))
            except Exception:
                continue
    if not scores:
        return {}
    return {"score_max": max(scores), "score_min": min(scores), "score_mean": sum(scores) / len(scores), "count": float(len(scores))}


def score_candidates(env: RadioMapEnv, candidates: List[Dict[str, Any]], objective: Dict[str, float]) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates
    if env.last_metrics is None:
        env._evaluate()
    base_metrics = env.last_metrics
    base_locs = list(env.tx_locs)
    constraints = getattr(env, "constraints", {}) or {}
    goal = getattr(env, "goal", {}) or {}
    site_limit = constraints.get("site_limit")
    scored = []
    try:
        for cand in candidates:
            env.tx_locs = base_locs + [(cand["row"], cand["col"])]
            metrics = env._evaluate()
            cov = _safe_float(metrics.coverage)
            cap = _safe_float(metrics.capacity)
            red = _safe_float(metrics.redundancy_rate)
            entry = dict(cand)
            entry["coverage"] = cov
            entry["capacity"] = cap
            entry["redundancy_rate"] = red
            entry["redundancy_score"] = redundancy_balance_score(red, goal.get("targets", {}).get("redundancy_rate"))
            entry["score"] = _objective_score(
                objective,
                cov,
                cap,
                len(env.tx_locs),
                redundancy_rate=red,
                redundancy_target=goal.get("targets", {}).get("redundancy_rate"),
            )
            if site_limit is not None:
                entry["feasible"] = len(env.tx_locs) <= int(site_limit)
            scored.append(entry)
    finally:
        env.tx_locs = base_locs
        env.last_metrics = base_metrics
    return scored


def score_candidates_with_weights(
    env: RadioMapEnv,
    candidates: List[Dict[str, Any]],
    weights: Dict[str, float],
    redundancy_target: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if env.last_metrics is None:
        env._evaluate()
    base_metrics = env.last_metrics
    base_locs = list(env.tx_locs)
    constraints = getattr(env, "constraints", {}) or {}
    goal = getattr(env, "goal", {}) or {}
    site_limit = constraints.get("site_limit")
    target_cfg = normalize_redundancy_target(redundancy_target or goal.get("targets", {}).get("redundancy_rate"))
    normalized, ok = normalize_metric_weights(weights)
    if not ok or normalized is None:
        normalized = compute_dynamic_metric_weights({"goal": env.goal, "constraints": env.constraints, "state": {"site_count": len(base_locs), "last_metrics": {}}, "diagnosis": {"margins": {}}})

    scored: List[Dict[str, Any]] = []
    try:
        for cand in candidates:
            env.tx_locs = base_locs + [(cand["row"], cand["col"])]
            metrics = env._evaluate()
            cov = _safe_float(metrics.coverage)
            cap = _safe_float(metrics.capacity)
            red = _safe_float(metrics.redundancy_rate)
            red_score = redundancy_balance_score(red, target_cfg)
            weighted_score = (
                float(normalized.get("w_cov", 0.0)) * cov
                + float(normalized.get("w_cap", 0.0)) * cap
                + float(normalized.get("w_red", 0.0)) * red_score
                - float(normalized.get("w_sites", 0.0)) * len(env.tx_locs)
            )
            entry = dict(cand)
            entry["coverage"] = cov
            entry["capacity"] = cap
            entry["redundancy_rate"] = red
            entry["redundancy_score"] = red_score
            entry["weighted_score"] = weighted_score
            entry["score"] = weighted_score
            entry["score_components"] = {
                "coverage": float(normalized.get("w_cov", 0.0)) * cov,
                "capacity": float(normalized.get("w_cap", 0.0)) * cap,
                "redundancy": float(normalized.get("w_red", 0.0)) * red_score,
                "site_penalty": -float(normalized.get("w_sites", 0.0)) * len(env.tx_locs),
            }
            if site_limit is not None:
                entry["feasible"] = len(env.tx_locs) <= int(site_limit)
            scored.append(entry)
    finally:
        env.tx_locs = base_locs
        env.last_metrics = base_metrics
    return scored


def select_best_candidate_with_weights(
    env: RadioMapEnv,
    candidates: List[Dict[str, Any]],
    weights: Dict[str, float],
    redundancy_target: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scored = score_candidates_with_weights(
        env=env,
        candidates=candidates,
        weights=weights,
        redundancy_target=redundancy_target,
    )
    if not scored:
        return {"row": 0, "col": 0, "z_m": 3.0}, []
    feasible = [cand for cand in scored if cand.get("feasible", True)]
    pool = feasible or scored
    best = max(pool, key=lambda item: float(item.get("weighted_score", item.get("score", -1e9))))
    return best, scored


def compact_obs_for_llm_decide(observation: str, max_candidates: int = 16, exclude_action_ids: Optional[Set[int]] = None) -> str:
    try:
        data = json.loads(observation)
    except Exception:
        return observation
    if not isinstance(data, dict):
        return observation
    current_positions = set()
    state = data.get("state")
    if isinstance(state, dict):
        sites = state.get("sites")
        if isinstance(sites, list):
            for site in sites:
                if isinstance(site, dict):
                    try:
                        current_positions.add((int(site.get("row")), int(site.get("col"))))
                    except Exception:
                        pass
    blocked_ids = set(int(x) for x in (exclude_action_ids or set()))
    candidates = data.get("candidates")
    if isinstance(candidates, list):
        compact = []
        candidates_sorted = sorted([c for c in candidates if isinstance(c, dict)], key=lambda c: float(c.get("score", -1e9)), reverse=True)
        for cand in candidates_sorted:
            try:
                cand_action_id = int(cand.get("action_id"))
                rc = (int(cand.get("row")), int(cand.get("col")))
            except Exception:
                continue
            if cand_action_id in blocked_ids or rc in current_positions:
                continue
            compact.append(
                {
                    "candidate_index": len(compact),
                    "action_id": cand_action_id,
                    "row": cand.get("row"),
                    "col": cand.get("col"),
                    "z_m": cand.get("z_m"),
                    "score": cand.get("score"),
                    "coverage": cand.get("coverage"),
                    "capacity": cand.get("capacity"),
                    "redundancy_rate": cand.get("redundancy_rate"),
                    "redundancy_score": cand.get("redundancy_score"),
                    "feasible": cand.get("feasible", True),
                }
            )
            if len(compact) >= max(1, int(max_candidates)):
                break
        data["candidates"] = compact
        data["candidate_stats"] = _candidate_stats(compact)
    return json.dumps(data, ensure_ascii=True)


def build_openai_messages(prompt_text: str, observation: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": prompt_text}, {"role": "user", "content": observation}]


def call_openai_chat(api_key: str, model: str, messages: List[Dict[str, str]], base_url: str, response_format: str = "none") -> str:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for planner=openai")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def extract_selected_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload["selected_action"] if "selected_action" in payload else payload


def extract_rationale(payload: Dict[str, Any]) -> str:
    for key in ("rationale", "reason", "analysis", "thought"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def validate_action(env: RadioMapEnv, action: Dict[str, Any], obs_payload: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    constraints = env.constraints or {}
    site_limit = constraints.get("site_limit")
    site_exact = constraints.get("site_exact")
    current_locs = list(env.tx_locs)
    current_set = {(int(r), int(c)) for r, c in current_locs}
    candidates = []
    if isinstance(obs_payload, dict):
        raw_candidates = obs_payload.get("candidates") or []
        if isinstance(raw_candidates, list):
            candidates = [c for c in raw_candidates if isinstance(c, dict)]

    name = action.get("name")
    args = action.get("args", {}) or {}
    if name == "Propose":
        sites = args.get("sites") or []
        if not sites:
            reasons.append("propose_sites_missing")
        for site in sites:
            try:
                rc = (int(site.get("row", 0)), int(site.get("col", 0)))
            except Exception:
                reasons.append("propose_invalid_site")
                continue
            if rc in current_set:
                reasons.append("propose_duplicate_existing_site")
            if candidates:
                matched = next((cand for cand in candidates if int(cand.get("row")) == rc[0] and int(cand.get("col")) == rc[1]), None)
                if matched is None:
                    reasons.append("propose_site_not_in_candidates")
                elif matched.get("feasible") is False:
                    reasons.append("propose_infeasible_candidate")
        if site_limit is not None and len(current_locs) + len(sites) > int(site_limit):
            reasons.append("site_limit_exceeded")
    elif name == "Refine":
        delta = args.get("rule_or_delta") or {}
        op = delta.get("op")
        if op == "move":
            idx = int(delta.get("id", -1))
            if not (0 <= idx < len(current_locs)):
                reasons.append("invalid_site_id")
            else:
                rc = (int(delta.get("row", 0)), int(delta.get("col", 0)))
                if current_locs[idx] == rc:
                    reasons.append("refine_noop_move")
        elif op == "remove":
            idx = int(delta.get("id", -1))
            if not (0 <= idx < len(current_locs)):
                reasons.append("invalid_site_id")
    elif name == "Finish":
        if env.last_metrics is None:
            env._evaluate()
        target_cov = env.goal.get("targets", {}).get("coverage_pct")
        target_cap = env.goal.get("targets", {}).get("capacity")
        if target_cov is not None and env.last_metrics.coverage < float(target_cov):
            reasons.append("finish_before_target_cov")
        if target_cap is not None and env.last_metrics.capacity < float(target_cap):
            reasons.append("finish_before_target_cap")
        if site_exact is not None and len(current_locs) != int(site_exact):
            reasons.append("finish_before_exact_sites")
    else:
        reasons.append("unknown_action_name")
    return len(reasons) == 0, reasons


def repair_action_with_candidates(action: Dict[str, Any], obs_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(action, dict) or not isinstance(obs_payload, dict):
        return action
    if action.get("name") == "Propose":
        args = action.get("args", {}) or {}
        if args.get("sites"):
            return action
        candidate_index = args.get("candidate_index")
        if candidate_index is None:
            return action
        try:
            candidate_index = int(candidate_index)
            candidates = obs_payload.get("candidates") or []
            cand = candidates[candidate_index]
            return {
                "name": "Propose",
                "args": {"sites": [{"row": int(cand["row"]), "col": int(cand["col"]), "z_m": float(cand.get("z_m", 3.0))}], "mode": "add"},
            }
        except Exception:
            return action
    if action.get("name") == "Refine":
        args = action.get("args", {}) or {}
        if isinstance(args.get("rule_or_delta"), dict):
            return action
        op = args.get("refine_op")
        if op == "remove":
            return {"name": "Refine", "args": {"rule_or_delta": {"op": "remove", "id": int(args.get("id", -1))}}}
        if op == "move":
            target_action_id = args.get("target_action_id")
            try:
                target_action_id = int(target_action_id)
            except Exception:
                target_action_id = None
            candidates = obs_payload.get("candidates") or []
            if target_action_id is not None:
                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    try:
                        if int(cand.get("action_id")) == target_action_id:
                            return {
                                "name": "Refine",
                                "args": {"rule_or_delta": {"op": "move", "id": int(args.get("id", -1)), "row": int(cand["row"]), "col": int(cand["col"])}},
                            }
                    except Exception:
                        continue
    return action
