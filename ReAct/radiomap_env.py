"""
用途:
  无线接入点布局决策环境。维护当前站点集合，执行 Propose/Refine/Finish 三类动作，并返回覆盖率/容量/冗余率反馈。

示例命令:
  无。该文件是环境模块，供主入口导入。

参数说明:
  RadioMapEnv(city_map_path, goal, constraints, user_request='', init_locs=None, pmnet=None, candidate_sample=64, seed=42, eval_device='mps')
    city_map_path: 输入建筑高度图路径。
    goal: 目标字典，如 coverage/capacity 目标。
    constraints: 约束字典，如 site_limit/site_exact。
    user_request: 原始用户请求文本。
    init_locs: 初始站点集合 [(row, col), ...]。
    pmnet: 自定义评估函数；为空则使用默认 PMNet 代理。
    candidate_sample: observation 中展示的候选点数量上限。
    seed: 随机种子。
    eval_device: 默认 PMNet/RMNet 评估设备。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import gymnasium as gym
except ModuleNotFoundError:
    class _SimpleSpace:
        def contains(self, x) -> bool:
            return True

    class _SimpleEnv:
        pass

    class _SimpleSpaces:
        Space = _SimpleSpace

    class _SimpleGym:
        Env = _SimpleEnv
        spaces = _SimpleSpaces()

    gym = _SimpleGym()

from env_utils import (
    calc_action_mask,
    calc_upsampling_loc,
    default_redundancy_target,
    get_stats,
    load_map_normalized,
    map_size,
    normalize_redundancy_target,
    roi_threshold,
)
from surrogate_adapter import infer_pmnet, load_pmnet


class TextSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        return isinstance(x, str)


ACTION_SPACE = [
    {"name": "Propose", "args_schema": {"candidate_index": "int"}},
    {"name": "Refine", "args_schema": {"refine_op": "move|remove", "id": "int", "target_action_id": "int"}},
    {"name": "Finish", "args_schema": {"final_site_set": "list[{row:int,col:int,z_m:float}]", "metrics": "dict"}},
]


def height_from_gray(gray: float) -> float:
    if gray <= roi_threshold:
        return 3.0
    return 6.6 + gray * (19.8 - 6.6) + 3.0


def build_candidates(pixel_map: np.ndarray) -> List[Dict[str, Any]]:
    action_mask = calc_action_mask(pixel_map)
    # action_mask返回0或1的32*32的动作空间采样块中心位置的坐标
    # np.where(action_mask > 0.5)返回一个tuple，返回的是“索引位置”，而不是值本身，需要[0]解引用得到(768,)
    action_ids = np.where(action_mask > 0.5)[0]
    candidates: List[Dict[str, Any]] = []
    for aid in action_ids:
        row, col = calc_upsampling_loc(int(aid))
        gray = float(pixel_map[row, col])
        candidates.append(
            {
                "action_id": int(aid),
                "row": int(row),
                "col": int(col),
                "gray": gray,
                "z_m": float(height_from_gray(gray)),
                "feasible": True,
            }
        )
    return candidates


def sample_candidates(candidates: List[Dict[str, Any]], k: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if k <= 0 or k >= len(candidates):
        return candidates
    # 随机采样（无放回抽样），从candidates长度中抽取k个索引位置
    idx = rng.choice(len(candidates), size=k, replace=False)
    return [candidates[int(i)] for i in idx]


@dataclass
class Metrics:
    coverage: float
    capacity: float
    redundancy_rate: float


def evaluate_site_count_constraints(constraints: Dict[str, Any], site_count: int) -> Dict[str, Any]:
    site_limit = constraints.get("site_limit")
    site_exact = constraints.get("site_exact")
    site_over = None if site_limit is None else max(0, site_count - int(site_limit))
    site_gap = None if site_exact is None else max(0, int(site_exact) - site_count)
    return {
        "site_limit": site_limit,
        "site_exact": site_exact,
        "site_over": site_over,
        "site_gap": site_gap,
        "within_limit": site_over in (None, 0),
        "exact_satisfied": site_gap in (None, 0),
    }


class RadioMapEnv(gym.Env):
    def __init__(
        self,
        city_map_path: str,
        goal: Dict[str, Any],
        constraints: Dict[str, Any],
        user_request: str = "",
        init_locs: Optional[List[Tuple[int, int]]] = None,
        pmnet=None,
        candidate_sample: int = 16,
        seed: int = 42,
        eval_device: str = "mps",
    ) -> None:
        super().__init__()
        # 0-255 -> 0-1
        self.pixel_map = load_map_normalized(city_map_path)
        self.eval_device = eval_device
        # 默认使用PMNet
        if pmnet is None:
            load_pmnet(device=eval_device)
            self.pmnet = lambda inputs: infer_pmnet(inputs, device=eval_device)
        else:
            self.pmnet = pmnet
        self.goal = goal
        if isinstance(self.goal.get("targets"), dict):
            self.goal["targets"].setdefault("redundancy_rate", default_redundancy_target())
        self.constraints = constraints
        self.user_request = user_request
        self.parsed_request: Optional[Dict[str, Any]] = None
        self.tx_locs = list(init_locs or [])
        self.last_metrics: Optional[Metrics] = None
        self.steps = 0
        self.candidates = build_candidates(self.pixel_map)
        self.candidate_sample = candidate_sample
        self.rng = np.random.default_rng(seed)
        self.observation_space = self.action_space = TextSpace()

    def _clamp_loc(self, row: int, col: int) -> Tuple[int, int]:
        row = max(0, min(map_size - 1, row))
        col = max(0, min(map_size - 1, col))
        return row, col

    def _evaluate(self) -> Metrics:
        _, coverage, capacity, redundancy_rate = get_stats(self.pixel_map, self.tx_locs, self.pmnet)
        metrics = Metrics(
            coverage=float(coverage),
            capacity=float(capacity),
            redundancy_rate=float(redundancy_rate),
        )
        self.last_metrics = metrics
        return metrics

    def _payload(self) -> Dict[str, Any]:
        # 随机抽样
        sample = sample_candidates(self.candidates, self.candidate_sample, self.rng)
        metrics = None
        if self.last_metrics is not None:
            metrics = {
                "coverage": float(self.last_metrics.coverage),
                "capacity": float(self.last_metrics.capacity),
                "redundancy_rate": float(self.last_metrics.redundancy_rate),
            }

        target_cov = self.goal.get("targets", {}).get("coverage_pct")
        target_cap = self.goal.get("targets", {}).get("capacity")
        # redundancy_rate字段由两部分组成
        redundancy_target = normalize_redundancy_target(self.goal.get("targets", {}).get("redundancy_rate"))
        site_count_status = evaluate_site_count_constraints(self.constraints, len(self.tx_locs))
        cov_gap = None if metrics is None or target_cov is None else float(target_cov) - metrics["coverage"]
        cap_gap = None if metrics is None or target_cap is None else float(target_cap) - metrics["capacity"]
        red_gap = None
        red_state = None
        if metrics is not None:
            # metrics["redundancy_rate"]是当前状态计算得到的冗余率
            deviation = abs(float(metrics["redundancy_rate"]) - float(redundancy_target["ideal"]))
            red_gap = max(0.0, deviation - float(redundancy_target["tolerance"]))
            if deviation <= float(redundancy_target["tolerance"]):
                red_state = "near_ideal"
            elif float(metrics["redundancy_rate"]) < float(redundancy_target["ideal"]):
                red_state = "too_low"
            else:
                red_state = "too_high"
        site_over = site_count_status["site_over"]
        site_gap = site_count_status["site_gap"]
        miss_type = []
        if cov_gap is not None and cov_gap > 0:
            miss_type.append("coverage_shortfall")
        if cap_gap is not None and cap_gap > 0:
            miss_type.append("capacity_shortfall")
        if red_gap is not None and red_gap > 0:
            miss_type.append(f"redundancy_{red_state}")
        if site_over is not None and site_over > 0:
            miss_type.append("site_over_limit")
        if site_count_status["site_exact"] is not None and site_gap is not None and site_gap > 0:
            miss_type.append("site_count_shortfall")

        payload = {
            "state": {
                "site_count": len(self.tx_locs),
                "sites": [{"row": int(r), "col": int(c), "z_m": float(height_from_gray(self.pixel_map[r, c]))} for r, c in self.tx_locs],
                "last_metrics": metrics,
            },
            "user_request": self.user_request,
            "goal": self.goal,
            "constraints": self.constraints,
            "action_space": ACTION_SPACE,
            "candidates": sample,
            "step": self.steps,
            "diagnosis": {
                "ok": len(miss_type) == 0 and site_count_status["within_limit"] and site_count_status["exact_satisfied"],
                "miss_type": miss_type,
                "margins": {
                    "coverage_gap": cov_gap,
                    "capacity_gap": cap_gap,
                    "redundancy_gap": red_gap,
                    "site_over": site_over,
                    "site_gap": site_gap,
                },
                "redundancy_target": redundancy_target,
                "redundancy_state": red_state,
            },
        }
        if self.parsed_request is not None:
            payload["parsed_request"] = self.parsed_request
        return payload

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        self.steps = 0
        self._evaluate()
        # 把Python 字典（dict）对象转为json字符串
        obs = json.dumps(self._payload(), ensure_ascii=True)
        info = {"steps": self.steps, "answer": None}
        return obs, info

    def _parse_action(self, action: str) -> Dict[str, Any]:
        action = action.strip()
        if action.startswith("DECIDE[") and action.endswith("]"):
            action = action[len("DECIDE[") : -1]
        # json解析为Python字典对象
        data = json.loads(action)
        if "selected_action" in data:
            return data["selected_action"]
        return data

    def _is_done(self, metrics: Metrics) -> bool:
        target = self.goal.get("targets", {}).get("coverage_pct")
        if target is not None and metrics.coverage < float(target):
            return False
        cap_target = self.goal.get("targets", {}).get("capacity")
        if cap_target is not None and metrics.capacity < float(cap_target):
            return False
        site_count_status = evaluate_site_count_constraints(self.constraints, len(self.tx_locs))
        site_exact = site_count_status["site_exact"]
        if site_exact is not None and not site_count_status["exact_satisfied"]:
            return False
        site_limit = site_count_status["site_limit"]
        if site_limit is not None and not site_count_status["within_limit"]:
            return False
        return True

    def apply_parsed_request(self, parsed: Dict[str, Any]) -> None:
        self.parsed_request = parsed

    def step(self, action: str):
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"steps": self.steps}

        try:
            # 解析LLM输出的DECIDE[]中的selected_action字段
            selected = self._parse_action(action)
        except Exception as exc:
            obs = f"Invalid action: {exc}"
            self.steps += 1
            # obs, reward, terminated, truncated, info
            return obs, reward, True, False, info

        # DECIDE[{"parsed_request": {...}, "metrics_snapshot": {"coverage": 0.0, "capacity": 0.0, "redundancy_rate": 0.0},
        # "rationale": "...", "selected_action": {"name": "...", "args": {...}}}]
        # DECIDE[{"rationale": "...", "weights": {"w_cov": 0.3, "w_cap": 0.2, "w_red": 0.4, "w_sites": 0.1}}]
        name = selected.get("name")
        args = selected.get("args", {}) or {}

        if name == "Propose":
            sites = args.get("sites") or []
            if not sites:
                info["error"] = "empty_proposal"
            else:
                for site in sites:
                    # _clamp_loc把坐标裁剪到合法地图范围内
                    row, col = self._clamp_loc(int(site.get("row", 0)), int(site.get("col", 0)))
                    self.tx_locs.append((row, col))
        elif name == "Refine":
            delta = args.get("rule_or_delta") or {}
            op = delta.get("op")
            # 移动一个已有站点
            if op == "move":
                idx = int(delta.get("id", -1))
                if 0 <= idx < len(self.tx_locs):
                    row, col = self._clamp_loc(int(delta.get("row", 0)), int(delta.get("col", 0)))
                    self.tx_locs[idx] = (row, col)
            elif op == "remove":
                idx = int(delta.get("id", -1))
                if 0 <= idx < len(self.tx_locs):
                    self.tx_locs.pop(idx)
        elif name == "Finish":
            sites = args.get("final_site_set") or []
            if sites:
                final_locs = []
                for site in sites:
                    row, col = self._clamp_loc(int(site.get("row", 0)), int(site.get("col", 0)))
                    final_locs.append((row, col))
                self.tx_locs = final_locs
            metrics = self._evaluate()
            reward = metrics.coverage
            done = self._is_done(metrics)
            if not done:
                info["error"] = "finish_before_target"
            obs = json.dumps(self._payload(), ensure_ascii=True)
            self.steps += 1
            return obs, reward, done, False, info
        else:
            info["error"] = "unknown_action"

        metrics = self._evaluate()
        # 在当前这条 ReAct 决策链里是可行的，因为这里真正决定动作的不是 reward，而是 _objective_score 和加权候选打分
        reward = metrics.coverage
        done = self._is_done(metrics)
        obs = json.dumps(self._payload(), ensure_ascii=True)
        self.steps += 1
        return obs, reward, done, False, info

if __name__ == "__main__":
    city_map_path = "/Users/epiphanyer/Desktop/coding/paper_experiment/dataset/png/buildingsWHeight/0.png"
    pixel_map = load_map_normalized(city_map_path)
    candidates = build_candidates(pixel_map)
    print(len(candidates))
