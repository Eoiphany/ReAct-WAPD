"""
用途:
  汇总一组无线接入点决策轨迹，输出成功率、步数、覆盖率、容量、冗余率和站点数统计。

示例命令:
python ReAct/evaluate_decision_trajectories.py --traj-dirs /Users/epiphanyer/Desktop/coding/ReAct/trajs_llamafactory_v4
python ReAct/evaluate_decision_trajectories.py --traj-dirs /Users/epiphanyer/Desktop/coding/ReAct/trajs_llamafactory_v4 --by-request
参数说明:
  --traj-dirs: 一个或多个轨迹目录。
  --by-request: 是否输出按 request 分组的统计和需求文本。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _iter_traj_files(traj_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(traj_dir)):
        if name.endswith(".json"):
            yield os.path.join(traj_dir, name)


def _parse_action_name(action_str: str) -> str:
    if not action_str or "DECIDE[" not in action_str:
        return "unknown"
    try:
        payload = action_str[action_str.find("DECIDE[") + 7 :].rstrip("]")
        data = json.loads(payload)
        return data.get("selected_action", {}).get("name", "unknown")
    except Exception:
        return "unknown"

# 可以理解成文件名作为需求id
def _request_id_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    if "__" in base:
        return base.split("__", 1)[1]
    return "unknown"


def _collapse_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def _shorten_text(text: str, width: int) -> str:
    compact = _collapse_whitespace(text)
    if len(compact) <= width:
        return compact
    if width <= 3:
        return compact[:width]
    return compact[: width - 3] + "..."


def _fit_cell(text: str, width: int) -> str:
    return _shorten_text(text, width).ljust(width)


def _source_names_from_dirs(traj_dirs: List[str]) -> List[str]:
    normalized = [os.path.abspath(os.path.expanduser(path)) for path in traj_dirs]
    basenames = [os.path.basename(path.rstrip(os.sep)) or path for path in normalized]
    counts = Counter(basenames)
    return [base if counts[base] == 1 else path for base, path in zip(basenames, normalized)]


def _goal_constraints_ok(last_obs: Dict[str, Any]) -> bool:
    goal = last_obs.get("goal", {}) or {}
    constraints = last_obs.get("constraints", {}) or {}
    state = last_obs.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}

    cov = _safe_float(metrics.get("coverage"), default=0.0)
    cap = _safe_float(metrics.get("capacity"), default=0.0)
    site_count = int(state.get("site_count") or 0)

    targets = goal.get("targets", {}) or {}
    cov_t = targets.get("coverage_pct")
    cap_t = targets.get("capacity")
    site_limit = constraints.get("site_limit")
    site_exact = constraints.get("site_exact")

    if cov_t is not None and cov < float(cov_t):
        return False
    if cap_t is not None and cap < float(cap_t):
        return False
    if site_limit is not None and site_count > int(site_limit):
        return False
    if site_exact is not None and site_count != int(site_exact):
        return False
    return True

# [
#     {
#         "observations": [
#             "{\"state\": {\"site_count\": 1, \"sites\": [{\"row\": 75, \"col\": 115, \"z_m\": 3.0}], \"last_metrics\": {\"coverage\": 0.43926916895149853, \"capacity\": 3.2436287114585736}}, \"user_request\": \"Cost is the top priority. Keep site count <= 5 and avoid adding new sites unless coverage < 85%.\\nTarget coverage >= 88% if possible within the cost limits.\", \"goal\": {\"primary\": \"maximize_coverage\", \"targets\": {\"coverage_pct\": 0.88}}, \"constraints\": {\"site_limit\": 5}, \"action_space\": [{\"name\": \"Propose\", \"args_schema\": {\"sites\": \"list[{row:int,col:int,z_m:float}]\", \"mode\": \"add|replace\"}}, {\"name\": \"Refine\", \"args_schema\": {\"rule_or_delta\": \"dict\"}}, {\"name\": \"Finish\", \"args_schema\": {\"final_site_set\": \"list[{row:int,col:int,z_m:float}]\", \"metrics\": \"dict\"}}, {\"name\": \"add_site\", \"args_schema\": {\"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"move_site\", \"args_schema\": {\"id\": \"int\", \"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"remove_site\", \"args_schema\": {\"id\": \"int\"}}], \"candidates\": [{\"action_id\": 417, \"row\": 107, \"col\": 11, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5532641580788503, \"capacity\": 2.72175527656495, \"score\": 1.3804435726641051, \"score_components\": {\"coverage\": 0.6639169896946203, \"capacity\": 0.816526582969485, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 93, \"row\": 19, \"col\": 235, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5488489174624185, \"capacity\": 2.6888143423955726, \"score\": 1.3652630036735738, \"score_components\": {\"coverage\": 0.6586187009549022, \"capacity\": 0.8066443027186717, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 102, \"row\": 27, \"col\": 51, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5126477261983549, \"capacity\": 2.405405147519795, \"score\": 1.2367988156939642, \"score_components\": {\"coverage\": 0.6151772714380258, \"capacity\": 0.7216215442559385, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 730, \"row\": 179, \"col\": 211, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6352462891178973, \"capacity\": 3.3569624026878215, \"score\": 1.669384267747823, \"score_components\": {\"coverage\": 0.7622955469414767, \"capacity\": 1.0070887208063464, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 699, \"row\": 171, \"col\": 219, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.62205729412877, \"capacity\": 3.249366640211705, \"score\": 1.6212787450180355, \"score_components\": {\"coverage\": 0.746468752954524, \"capacity\": 0.9748099920635115, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 663, \"row\": 163, \"col\": 187, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.621284390658977, \"capacity\": 3.214331815245977, \"score\": 1.6098408133645654, \"score_components\": {\"coverage\": 0.7455412687907724, \"capacity\": 0.9642995445737931, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 509, \"row\": 123, \"col\": 235, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5976789259714475, \"capacity\": 3.0954343441559473, \"score\": 1.5458450144125209, \"score_components\": {\"coverage\": 0.7172147111657369, \"capacity\": 0.9286303032467842, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 766, \"row\": 187, \"col\": 243, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5974709274841639, \"capacity\": 3.039690290410363, \"score\": 1.5288722001041055, \"score_components\": {\"coverage\": 0.7169651129809966, \"capacity\": 0.9119070871231089, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 806, \"row\": 203, \"col\": 51, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6182731398317103, \"capacity\": 3.220734671822508, \"score\": 1.6081481693448048, \"score_components\": {\"coverage\": 0.7419277677980524, \"capacity\": 0.9662204015467524, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 720, \"row\": 179, \"col\": 131, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.616119882764489, \"capacity\": 3.1299616238633643, \"score\": 1.578332346476396, \"score_components\": {\"coverage\": 0.7393438593173868, \"capacity\": 0.9389884871590093, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 413, \"row\": 99, \"col\": 235, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5905360688285903, \"capacity\": 2.9356620261786084, \"score\": 1.4893418904478908, \"score_components\": {\"coverage\": 0.7086432825943084, \"capacity\": 0.8806986078535824, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 91, \"row\": 19, \"col\": 219, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5508201758532665, \"capacity\": 2.6817286322116214, \"score\": 1.3655028006874061, \"score_components\": {\"coverage\": 0.6609842110239198, \"capacity\": 0.8045185896634864, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 871, \"row\": 219, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6195920393306231, \"capacity\": 3.2289365456861123, \"score\": 1.6121914109025814, \"score_components\": {\"coverage\": 0.7435104471967477, \"capacity\": 0.9686809637058337, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 997, \"row\": 251, \"col\": 43, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5748723645646213, \"capacity\": 2.830636857851917, \"score\": 1.4390378948331206, \"score_components\": {\"coverage\": 0.6898468374775456, \"capacity\": 0.8491910573555751, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 765, \"row\": 187, \"col\": 235, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6087642998960008, \"capacity\": 3.127096226085394, \"score\": 1.568646027700819, \"score_components\": {\"coverage\": 0.7305171598752009, \"capacity\": 0.9381288678256181, \"site_penalty\": -0.1}, \"feasible\": true}, {\"action_id\": 194, \"row\": 51, \"col\": 19, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.5246572752198166, \"capacity\": 2.4683657782677897, \"score\": 1.2700984637441168, \"score_components\": {\"coverage\": 0.6295887302637799, \"capacity\": 0.7405097334803369, \"site_penalty\": -0.1}, \"feasible\": true}], \"step\": 0, \"diagnosis\": {\"ok\": false, \"miss_type\": [\"coverage_shortfall\"], \"margins\": {\"coverage_gap\": 0.44073083104850147, \"capacity_gap\": null, \"site_over\": 0}, \"summary\": \"coverage shortfall 0.4407\"}, \"last_eval\": {\"coverage\": 0.43926916895149853, \"capacity\": 3.2436287114585736}, \"candidate_stats\": {\"score_max\": 1.669384267747823, \"score_min\": 1.2367988156939642, \"score_mean\": 1.4930640898009893, \"count\": 16.0}}",
#             "{\"state\": {\"site_count\": 2, \"sites\": [{\"row\": 75, \"col\": 115, \"z_m\": 3.0}, {\"row\": 211, \"col\": 163, \"z_m\": 3.0}], \"last_metrics\": {\"coverage\": 0.629495603668337, \"capacity\": 3.324893003684976}}, \"user_request\": \"Cost is the top priority. Keep site count <= 5 and avoid adding new sites unless coverage < 85%.\\nTarget coverage >= 88% if possible within the cost limits.\", \"goal\": {\"primary\": \"maximize_coverage\", \"targets\": {\"coverage_pct\": 0.88}}, \"constraints\": {\"site_limit\": 5}, \"action_space\": [{\"name\": \"Propose\", \"args_schema\": {\"sites\": \"list[{row:int,col:int,z_m:float}]\", \"mode\": \"add|replace\"}}, {\"name\": \"Refine\", \"args_schema\": {\"rule_or_delta\": \"dict\"}}, {\"name\": \"Finish\", \"args_schema\": {\"final_site_set\": \"list[{row:int,col:int,z_m:float}]\", \"metrics\": \"dict\"}}, {\"name\": \"add_site\", \"args_schema\": {\"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"move_site\", \"args_schema\": {\"id\": \"int\", \"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"remove_site\", \"args_schema\": {\"id\": \"int\"}}], \"candidates\": [{\"action_id\": 119, \"row\": 27, \"col\": 187, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7017396237118276, \"capacity\": 3.7614258554757285, \"score\": 1.8205153050969116, \"score_components\": {\"coverage\": 0.8420875484541931, \"capacity\": 1.1284277566427185, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 982, \"row\": 243, \"col\": 179, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6518885317197693, \"capacity\": 3.5154441786043957, \"score\": 1.686899491645042, \"score_components\": {\"coverage\": 0.7822662380637232, \"capacity\": 1.0546332535813188, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 92, \"row\": 19, \"col\": 227, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7131369953673063, \"capacity\": 3.8876190456821282, \"score\": 1.8720501081454062, \"score_components\": {\"coverage\": 0.8557643944407676, \"capacity\": 1.1662857137046385, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 439, \"row\": 107, \"col\": 187, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7069183133213577, \"capacity\": 3.830831649625138, \"score\": 1.8475514708731704, \"score_components\": {\"coverage\": 0.8483019759856292, \"capacity\": 1.1492494948875414, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 283, \"row\": 67, \"col\": 219, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7526330717594781, \"capacity\": 4.218197305082783, \"score\": 2.0186188776362086, \"score_components\": {\"coverage\": 0.9031596861113738, \"capacity\": 1.2654591915248348, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 686, \"row\": 171, \"col\": 115, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6900278907062494, \"capacity\": 3.7163634408910866, \"score\": 1.7929425011148252, \"score_components\": {\"coverage\": 0.8280334688474992, \"capacity\": 1.1149090322673259, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 973, \"row\": 243, \"col\": 107, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6894511676278718, \"capacity\": 3.752948194908803, \"score\": 1.803225859626087, \"score_components\": {\"coverage\": 0.8273414011534462, \"capacity\": 1.1258844584726408, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 930, \"row\": 235, \"col\": 19, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7381535407015222, \"capacity\": 4.092091259111697, \"score\": 1.9634116265753359, \"score_components\": {\"coverage\": 0.8857842488418266, \"capacity\": 1.2276273777335092, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 708, \"row\": 179, \"col\": 35, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7543585137562636, \"capacity\": 4.241440719762766, \"score\": 2.027662432436346, \"score_components\": {\"coverage\": 0.9052302165075163, \"capacity\": 1.2724322159288297, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 329, \"row\": 83, \"col\": 75, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.6871182755034508, \"capacity\": 3.6984230267616063, \"score\": 1.7840688386326229, \"score_components\": {\"coverage\": 0.8245419306041409, \"capacity\": 1.1095269080284818, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 647, \"row\": 163, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7410182471400208, \"capacity\": 4.15737092257437, \"score\": 1.9864331733403362, \"score_components\": {\"coverage\": 0.889221896568025, \"capacity\": 1.247211276772311, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 323, \"row\": 83, \"col\": 27, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7121750023636191, \"capacity\": 3.8515376137300685, \"score\": 1.8600712869553635, \"score_components\": {\"coverage\": 0.8546100028363429, \"capacity\": 1.1554612841190206, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 610, \"row\": 155, \"col\": 19, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7376122719107497, \"capacity\": 4.109818728918745, \"score\": 1.9680803449685231, \"score_components\": {\"coverage\": 0.8851347262928997, \"capacity\": 1.2329456186756234, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 479, \"row\": 115, \"col\": 251, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7280939774983455, \"capacity\": 4.073781100849123, \"score\": 1.9458471032527513, \"score_components\": {\"coverage\": 0.8737127729980145, \"capacity\": 1.222134330254737, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 352, \"row\": 91, \"col\": 3, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7046185118653683, \"capacity\": 3.7867793935386085, \"score\": 1.8315760323000245, \"score_components\": {\"coverage\": 0.8455422142384419, \"capacity\": 1.1360338180615825, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}, {\"action_id\": 346, \"row\": 83, \"col\": 211, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7480405597050204, \"capacity\": 4.163195283127691, \"score\": 1.996607256584332, \"score_components\": {\"coverage\": 0.8976486716460245, \"capacity\": 1.2489585849383074, \"site_penalty\": -0.15000000000000002}, \"feasible\": true}], \"step\": 0, \"diagnosis\": {\"ok\": false, \"miss_type\": [\"coverage_shortfall\"], \"margins\": {\"coverage_gap\": 0.250504396331663, \"capacity_gap\": null, \"site_over\": 0}, \"summary\": \"coverage shortfall 0.2505\"}, \"last_eval\": {\"coverage\": 0.629495603668337, \"capacity\": 3.324893003684976}, \"candidate_stats\": {\"score_max\": 2.027662432436346, \"score_min\": 1.686899491645042, \"score_mean\": 1.8878476068239554, \"count\": 16.0}}",
#             "{\"state\": {\"site_count\": 3, \"sites\": [{\"row\": 75, \"col\": 115, \"z_m\": 3.0}, {\"row\": 211, \"col\": 163, \"z_m\": 3.0}, {\"row\": 203, \"col\": 51, \"z_m\": 3.0}], \"last_metrics\": {\"coverage\": 0.7612271910749739, \"capacity\": 4.382578706973096}}, \"user_request\": \"Cost is the top priority. Keep site count <= 5 and avoid adding new sites unless coverage < 85%.\\nTarget coverage >= 88% if possible within the cost limits.\", \"goal\": {\"primary\": \"maximize_coverage\", \"targets\": {\"coverage_pct\": 0.88}}, \"constraints\": {\"site_limit\": 5}, \"action_space\": [{\"name\": \"Propose\", \"args_schema\": {\"sites\": \"list[{row:int,col:int,z_m:float}]\", \"mode\": \"add|replace\"}}, {\"name\": \"Refine\", \"args_schema\": {\"rule_or_delta\": \"dict\"}}, {\"name\": \"Finish\", \"args_schema\": {\"final_site_set\": \"list[{row:int,col:int,z_m:float}]\", \"metrics\": \"dict\"}}, {\"name\": \"add_site\", \"args_schema\": {\"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"move_site\", \"args_schema\": {\"id\": \"int\", \"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"remove_site\", \"args_schema\": {\"id\": \"int\"}}], \"candidates\": [{\"action_id\": 963, \"row\": 243, \"col\": 27, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.767091330244871, \"capacity\": 4.418094329083625, \"score\": 2.0459378950189326, \"score_components\": {\"coverage\": 0.9205095962938451, \"capacity\": 1.3254282987250876, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 910, \"row\": 227, \"col\": 115, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7669424222369292, \"capacity\": 4.424986590895091, \"score\": 2.047826883952842, \"score_components\": {\"coverage\": 0.920330906684315, \"capacity\": 1.327495977268527, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 357, \"row\": 91, \"col\": 43, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8039897891651697, \"capacity\": 4.764568349841508, \"score\": 2.1941582519506557, \"score_components\": {\"coverage\": 0.9647877469982036, \"capacity\": 1.4293705049524525, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 246, \"row\": 59, \"col\": 179, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.820048217831143, \"capacity\": 4.813646209873831, \"score\": 2.2281517243595204, \"score_components\": {\"coverage\": 0.9840578613973716, \"capacity\": 1.4440938629621491, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 227, \"row\": 59, \"col\": 27, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8121915476978349, \"capacity\": 4.81449921961386, \"score\": 2.2189796231215597, \"score_components\": {\"coverage\": 0.9746298572374018, \"capacity\": 1.444349765884158, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 121, \"row\": 27, \"col\": 203, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8400987992814597, \"capacity\": 4.94016494105874, \"score\": 2.290168041455373, \"score_components\": {\"coverage\": 1.0081185591377515, \"capacity\": 1.4820494823176218, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 76, \"row\": 19, \"col\": 99, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.770320034036116, \"capacity\": 4.427223549373563, \"score\": 2.052551105655408, \"score_components\": {\"coverage\": 0.9243840408433392, \"capacity\": 1.328167064812069, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 737, \"row\": 187, \"col\": 11, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7723929280514323, \"capacity\": 4.468163521753274, \"score\": 2.0673205701877007, \"score_components\": {\"coverage\": 0.9268715136617187, \"capacity\": 1.3404490565259821, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 39, \"row\": 11, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.789663893353503, \"capacity\": 4.577080301130213, \"score\": 2.120720762363267, \"score_components\": {\"coverage\": 0.9475966720242035, \"capacity\": 1.3731240903390638, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 848, \"row\": 211, \"col\": 131, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7666918786045192, \"capacity\": 4.421438432967701, \"score\": 2.046461784215733, \"score_components\": {\"coverage\": 0.920030254325423, \"capacity\": 1.3264315298903102, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 611, \"row\": 155, \"col\": 27, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7847593835681195, \"capacity\": 4.570529476855439, \"score\": 2.1128701033383748, \"score_components\": {\"coverage\": 0.9417112602817433, \"capacity\": 1.3711588430566317, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 951, \"row\": 235, \"col\": 187, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7771934385931739, \"capacity\": 4.531239188546762, \"score\": 2.0920038828758374, \"score_components\": {\"coverage\": 0.9326321263118087, \"capacity\": 1.3593717565640286, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 7, \"row\": 3, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7849933818663137, \"capacity\": 4.5351119395195525, \"score\": 2.102525640095442, \"score_components\": {\"coverage\": 0.9419920582395764, \"capacity\": 1.3605335818558657, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 360, \"row\": 91, \"col\": 67, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7996525479814692, \"capacity\": 4.728556649620078, \"score\": 2.1781500524637862, \"score_components\": {\"coverage\": 0.9595830575777631, \"capacity\": 1.4185669948860233, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 174, \"row\": 43, \"col\": 115, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.767944596766569, \"capacity\": 4.424211398294123, \"score\": 2.0487969356081193, \"score_components\": {\"coverage\": 0.9215335161198828, \"capacity\": 1.3272634194882367, \"site_penalty\": -0.2}, \"feasible\": true}, {\"action_id\": 147, \"row\": 35, \"col\": 155, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.7921551479625604, \"capacity\": 4.5769324654524555, \"score\": 2.1236659171908085, \"score_components\": {\"coverage\": 0.9505861775550724, \"capacity\": 1.3730797396357366, \"site_penalty\": -0.2}, \"feasible\": true}], \"step\": 1, \"diagnosis\": {\"ok\": false, \"miss_type\": [\"coverage_shortfall\"], \"margins\": {\"coverage_gap\": 0.11877280892502606, \"capacity_gap\": null, \"site_over\": 0}, \"summary\": \"coverage shortfall 0.1188\"}, \"last_eval\": {\"coverage\": 0.7612271910749739, \"capacity\": 4.382578706973096}, \"candidate_stats\": {\"score_max\": 2.290168041455373, \"score_min\": 2.0459378950189326, \"score_mean\": 2.123143073365835, \"count\": 16.0}}",
#             "{\"state\": {\"site_count\": 4, \"sites\": [{\"row\": 75, \"col\": 115, \"z_m\": 3.0}, {\"row\": 211, \"col\": 163, \"z_m\": 3.0}, {\"row\": 203, \"col\": 51, \"z_m\": 3.0}, {\"row\": 35, \"col\": 243, \"z_m\": 3.0}], \"last_metrics\": {\"coverage\": 0.8476883804481422, \"capacity\": 4.999816034226358}}, \"user_request\": \"Cost is the top priority. Keep site count <= 5 and avoid adding new sites unless coverage < 85%.\\nTarget coverage >= 88% if possible within the cost limits.\", \"goal\": {\"primary\": \"maximize_coverage\", \"targets\": {\"coverage_pct\": 0.88}}, \"constraints\": {\"site_limit\": 5}, \"action_space\": [{\"name\": \"Propose\", \"args_schema\": {\"sites\": \"list[{row:int,col:int,z_m:float}]\", \"mode\": \"add|replace\"}}, {\"name\": \"Refine\", \"args_schema\": {\"rule_or_delta\": \"dict\"}}, {\"name\": \"Finish\", \"args_schema\": {\"final_site_set\": \"list[{row:int,col:int,z_m:float}]\", \"metrics\": \"dict\"}}, {\"name\": \"add_site\", \"args_schema\": {\"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"move_site\", \"args_schema\": {\"id\": \"int\", \"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"remove_site\", \"args_schema\": {\"id\": \"int\"}}], \"candidates\": [{\"action_id\": 112, \"row\": 27, \"col\": 131, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.856400680722322, \"capacity\": 5.0540343228105895, \"score\": 2.293891113709963, \"score_components\": {\"coverage\": 1.0276808168667864, \"capacity\": 1.5162102968431768, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 547, \"row\": 139, \"col\": 27, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8790819703129432, \"capacity\": 5.277407665062073, \"score\": 2.3881206638941537, \"score_components\": {\"coverage\": 1.0548983643755319, \"capacity\": 1.5832222995186218, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 397, \"row\": 99, \"col\": 107, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.854819419495131, \"capacity\": 5.050804252147126, \"score\": 2.291024579038295, \"score_components\": {\"coverage\": 1.025783303394157, \"capacity\": 1.5152412756441376, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 962, \"row\": 243, \"col\": 19, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8528528883426303, \"capacity\": 5.032421549572485, \"score\": 2.2831499308829017, \"score_components\": {\"coverage\": 1.0234234660111563, \"capacity\": 1.5097264648717454, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 855, \"row\": 211, \"col\": 187, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8617471872931832, \"capacity\": 5.134289399764393, \"score\": 2.3243834446811373, \"score_components\": {\"coverage\": 1.0340966247518197, \"capacity\": 1.5402868199293178, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 268, \"row\": 67, \"col\": 99, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8542852415618796, \"capacity\": 5.0371728786953955, \"score\": 2.2862941534828742, \"score_components\": {\"coverage\": 1.0251422898742555, \"capacity\": 1.5111518636086185, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 913, \"row\": 227, \"col\": 139, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8527654344332041, \"capacity\": 5.033845638231489, \"score\": 2.2834722127892917, \"score_components\": {\"coverage\": 1.0233185213198448, \"capacity\": 1.5101536914694467, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 506, \"row\": 123, \"col\": 211, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8956934858655574, \"capacity\": 5.430996351729656, \"score\": 2.4541310885575656, \"score_components\": {\"coverage\": 1.074832183038669, \"capacity\": 1.6292989055188967, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 11, \"row\": 3, \"col\": 91, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8573437647726199, \"capacity\": 5.055189269903203, \"score\": 2.295369298698105, \"score_components\": {\"coverage\": 1.0288125177271439, \"capacity\": 1.516556780970961, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 100, \"row\": 27, \"col\": 35, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8875862720998393, \"capacity\": 5.31716275375589, \"score\": 2.410252352646574, \"score_components\": {\"coverage\": 1.0651035265198072, \"capacity\": 1.595148826126767, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 946, \"row\": 235, \"col\": 147, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.853517065330434, \"capacity\": 5.036949789471774, \"score\": 2.285305415238053, \"score_components\": {\"coverage\": 1.0242204783965208, \"capacity\": 1.5110849368415322, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 653, \"row\": 163, \"col\": 107, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8549305095962938, \"capacity\": 5.053546421123919, \"score\": 2.2919805378527283, \"score_components\": {\"coverage\": 1.0259166115155525, \"capacity\": 1.5160639263371756, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 359, \"row\": 91, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8880991774605276, \"capacity\": 5.38108042865473, \"score\": 2.430043141549052, \"score_components\": {\"coverage\": 1.065719012952633, \"capacity\": 1.614324128596419, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 106, \"row\": 27, \"col\": 83, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8634419022407109, \"capacity\": 5.091402419002259, \"score\": 2.3135510083895308, \"score_components\": {\"coverage\": 1.036130282688853, \"capacity\": 1.5274207257006778, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 59, \"row\": 11, \"col\": 219, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8568190413160632, \"capacity\": 5.080229448075245, \"score\": 2.302251684001849, \"score_components\": {\"coverage\": 1.0281828495792757, \"capacity\": 1.5240688344225732, \"site_penalty\": -0.25}, \"feasible\": true}, {\"action_id\": 232, \"row\": 59, \"col\": 67, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.8860144653493429, \"capacity\": 5.3208221190641805, \"score\": 2.4094639941384655, \"score_components\": {\"coverage\": 1.0632173584192115, \"capacity\": 1.5962466357192542, \"site_penalty\": -0.25}, \"feasible\": true}], \"step\": 2, \"diagnosis\": {\"ok\": false, \"miss_type\": [\"coverage_shortfall\"], \"margins\": {\"coverage_gap\": 0.032311619551857795, \"capacity_gap\": null, \"site_over\": 0}, \"summary\": \"coverage shortfall 0.0323\"}, \"last_eval\": {\"coverage\": 0.8476883804481422, \"capacity\": 4.999816034226358}, \"candidate_stats\": {\"score_max\": 2.4541310885575656, \"score_min\": 2.2831499308829017, \"score_mean\": 2.3339177887219087, \"count\": 16.0}}",
#             "{\"state\": {\"site_count\": 5, \"sites\": [{\"row\": 75, \"col\": 115, \"z_m\": 3.0}, {\"row\": 211, \"col\": 163, \"z_m\": 3.0}, {\"row\": 203, \"col\": 51, \"z_m\": 3.0}, {\"row\": 35, \"col\": 243, \"z_m\": 3.0}, {\"row\": 179, \"col\": 227, \"z_m\": 3.0}], \"last_metrics\": {\"coverage\": 0.9060579559421387, \"capacity\": 5.528334649157607}}, \"user_request\": \"Cost is the top priority. Keep site count <= 5 and avoid adding new sites unless coverage < 85%.\\nTarget coverage >= 88% if possible within the cost limits.\", \"goal\": {\"primary\": \"maximize_coverage\", \"targets\": {\"coverage_pct\": 0.88}}, \"constraints\": {\"site_limit\": 5}, \"action_space\": [{\"name\": \"Propose\", \"args_schema\": {\"sites\": \"list[{row:int,col:int,z_m:float}]\", \"mode\": \"add|replace\"}}, {\"name\": \"Refine\", \"args_schema\": {\"rule_or_delta\": \"dict\"}}, {\"name\": \"Finish\", \"args_schema\": {\"final_site_set\": \"list[{row:int,col:int,z_m:float}]\", \"metrics\": \"dict\"}}, {\"name\": \"add_site\", \"args_schema\": {\"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"move_site\", \"args_schema\": {\"id\": \"int\", \"row\": \"int\", \"col\": \"int\", \"z_m\": \"float\"}}, {\"name\": \"remove_site\", \"args_schema\": {\"id\": \"int\"}}], \"candidates\": [{\"action_id\": 779, \"row\": 195, \"col\": 91, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9097759289023353, \"capacity\": 5.559229091549756, \"score\": 1.7216217472122863, \"score_components\": {\"coverage\": 0.9097759289023353, \"capacity\": 1.1118458183099513, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 724, \"row\": 179, \"col\": 163, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9130259052661436, \"capacity\": 5.602621959081019, \"score\": 1.733550297082347, \"score_components\": {\"coverage\": 0.9130259052661436, \"capacity\": 1.1205243918162038, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 936, \"row\": 235, \"col\": 67, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9098515647158929, \"capacity\": 5.561771895615573, \"score\": 1.7222059438390074, \"score_components\": {\"coverage\": 0.9098515647158929, \"capacity\": 1.1123543791231147, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 924, \"row\": 227, \"col\": 227, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9160158835208472, \"capacity\": 5.631292005103673, \"score\": 1.7422742845415817, \"score_components\": {\"coverage\": 0.9160158835208472, \"capacity\": 1.1262584010207346, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 844, \"row\": 211, \"col\": 99, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9100051999621821, \"capacity\": 5.561447854506941, \"score\": 1.7222947708635703, \"score_components\": {\"coverage\": 0.9100051999621821, \"capacity\": 1.1122895709013882, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 117, \"row\": 27, \"col\": 171, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9147938924080552, \"capacity\": 5.627503470665229, \"score\": 1.7402945865411013, \"score_components\": {\"coverage\": 0.9147938924080552, \"capacity\": 1.1255006941330459, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 401, \"row\": 99, \"col\": 139, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9130306325044909, \"capacity\": 5.6102508720500985, \"score\": 1.7350808069145105, \"score_components\": {\"coverage\": 0.9130306325044909, \"capacity\": 1.1220501744100198, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 711, \"row\": 179, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9130353597428382, \"capacity\": 5.587963996072389, \"score\": 1.7306281589573158, \"score_components\": {\"coverage\": 0.9130353597428382, \"capacity\": 1.1175927992144779, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 91, \"row\": 19, \"col\": 219, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9127753616337336, \"capacity\": 5.602080351441863, \"score\": 1.7331914319221065, \"score_components\": {\"coverage\": 0.9127753616337336, \"capacity\": 1.1204160702883728, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 730, \"row\": 179, \"col\": 211, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9159757019948946, \"capacity\": 5.651838059834096, \"score\": 1.7463433139617137, \"score_components\": {\"coverage\": 0.9159757019948946, \"capacity\": 1.1303676119668191, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 719, \"row\": 179, \"col\": 123, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.911470643849863, \"capacity\": 5.574610868139256, \"score\": 1.7263928174777143, \"score_components\": {\"coverage\": 0.911470643849863, \"capacity\": 1.1149221736278514, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 664, \"row\": 163, \"col\": 195, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.916840786612461, \"capacity\": 5.6561753620627595, \"score\": 1.7480758590250127, \"score_components\": {\"coverage\": 0.916840786612461, \"capacity\": 1.131235072412552, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 683, \"row\": 171, \"col\": 91, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9115793703318522, \"capacity\": 5.573764464886134, \"score\": 1.726332263309079, \"score_components\": {\"coverage\": 0.9115793703318522, \"capacity\": 1.1147528929772268, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 463, \"row\": 115, \"col\": 123, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9121915476978348, \"capacity\": 5.593012592114131, \"score\": 1.7307940661206611, \"score_components\": {\"coverage\": 0.9121915476978348, \"capacity\": 1.1186025184228263, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 199, \"row\": 51, \"col\": 59, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9439349532003404, \"capacity\": 5.843736529297576, \"score\": 1.8126822590598557, \"score_components\": {\"coverage\": 0.9439349532003404, \"capacity\": 1.1687473058595153, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}, {\"action_id\": 320, \"row\": 83, \"col\": 3, \"gray\": 0.0, \"z_m\": 3.0, \"coverage\": 0.9431880495414579, \"capacity\": 5.881398760461157, \"score\": 1.8194678016336894, \"score_components\": {\"coverage\": 0.9431880495414579, \"capacity\": 1.1762797520922315, \"site_penalty\": -0.30000000000000004}, \"feasible\": false}], \"step\": 3, \"diagnosis\": {\"ok\": true, \"miss_type\": [], \"margins\": {\"coverage_gap\": -0.026057955942138666, \"capacity_gap\": null, \"site_over\": 0}, \"summary\": \"all goals and constraints satisfied\"}, \"last_eval\": {\"coverage\": 0.9060579559421387, \"capacity\": 5.528334649157607}, \"candidate_stats\": {\"score_max\": 1.8194678016336894, \"score_min\": 1.7216217472122863, \"score_mean\": 1.743201900528847, \"count\": 16.0}}"
#         ],
#         "actions": [
#             "DECIDE[{\"selected_action\": {\"name\": \"Propose\", \"args\": {\"sites\": [{\"row\": 211, \"col\": 163, \"z_m\": 3.0}], \"mode\": \"add\"}}}]",
#             "DECIDE[{\"selected_action\": {\"name\": \"Propose\", \"args\": {\"sites\": [{\"row\": 203, \"col\": 51, \"z_m\": 3.0}], \"mode\": \"add\"}}}]",
#             "DECIDE[{\"selected_action\": {\"name\": \"Propose\", \"args\": {\"sites\": [{\"row\": 35, \"col\": 243, \"z_m\": 3.0}], \"mode\": \"add\"}}}]",
#             "DECIDE[{\"selected_action\": {\"name\": \"Propose\", \"args\": {\"sites\": [{\"row\": 179, \"col\": 227, \"z_m\": 3.0}], \"mode\": \"add\"}}}]"
#         ],
#         "rationales": [
#             "auto: coverage shortfall 0.4407; action=Propose",
#             "auto: coverage shortfall 0.2505; action=Propose",
#             "auto: coverage shortfall 0.1188; action=Propose",
#             "auto: coverage shortfall 0.0323; action=Propose"
#         ],
#         "steps": 3
#     }
# ]
def _summarize_traj(path: str) -> Tuple[Dict[str, Any], Counter]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    if not isinstance(data, list) or not data:
        return {}, Counter()
    item = data[0]
    actions = item.get("actions", []) or []
    last_obs = {}
    if item.get("observations"):
        try:
            last_obs = json.loads(item["observations"][-1])
        except Exception:
            last_obs = {}
    state = last_obs.get("state", {}) or {}
    metrics = state.get("last_metrics", {}) or {}
    request_text = str(last_obs.get("user_request") or "").strip()
    summary = {
        "path": path,
        # 可以理解成文件名作为需求id
        "request_id": _request_id_from_path(path),
        "request_text": request_text,
        "steps": int(item.get("steps") or (len(actions) - 1 if actions else 0)),
        "coverage": _safe_float(metrics.get("coverage"), default=0.0),
        "capacity": _safe_float(metrics.get("capacity"), default=0.0),
        "redundancy_rate": _safe_float(metrics.get("redundancy_rate"), default=0.0),
        "sites": int(state.get("site_count") or 0),
        "ok": _goal_constraints_ok(last_obs),
    }
    action_counts = Counter(_parse_action_name(a) for a in actions)
    return summary, action_counts


def _avg(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dirs", nargs="+", required=True)
    parser.add_argument("--by-request", action="store_true")
    args = parser.parse_args()

    source_names = _source_names_from_dirs(args.traj_dirs)

    rows = []
    per_request = defaultdict(list)
    per_request_actions = defaultdict(Counter)
    per_actions = {}
    for source_name, traj_dir in zip(source_names, args.traj_dirs):
        summaries = []
        action_counts = Counter()
        for path in _iter_traj_files(traj_dir):
            # 返回评估结果和动作数量
            summary, counts = _summarize_traj(path)
            if not summary:
                continue
            summaries.append(summary)
            action_counts.update(counts)
            per_request[(source_name, summary["request_id"])].append(summary)
            per_request_actions[(source_name, summary["request_id"])].update(counts)
        per_actions[source_name] = action_counts
        if summaries:
            rows.append(
                {
                    "source": source_name,
                    "n": len(summaries),
                    "ok_rate": _avg([1.0 if s["ok"] else 0.0 for s in summaries]),
                    "steps": _avg([s["steps"] for s in summaries]),
                    "coverage": _avg([s["coverage"] for s in summaries]),
                    "capacity": _avg([s["capacity"] for s in summaries]),
                    "redundancy_rate": _avg([s["redundancy_rate"] for s in summaries]),
                    "sites": _avg([s["sites"] for s in summaries]),
                }
            )

    header = "Source".ljust(20) + "N".rjust(6) + "OK%".rjust(8) + "Steps".rjust(8) + "Coverage".rjust(12) + "Capacity".rjust(12) + "Redundancy".rjust(12) + "Sites".rjust(8)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            _fit_cell(row["source"], 20)
            + f"{row['n']:6d}"
            + f"{row['ok_rate']*100:8.1f}"
            + f"{row['steps']:8.2f}"
            + f"{row['coverage']:12.4f}"
            + f"{row['capacity']:12.4f}"
            + f"{row['redundancy_rate']:12.4f}"
            + f"{row['sites']:8.2f}"
        )

    print("\nAction distribution:")
    for source_name in source_names:
        counts = per_actions.get(source_name, Counter())
        total = sum(counts.values()) or 1
        parts = [f"{k}:{counts[k]}({counts[k]/total:.2f})" for k in sorted(counts)]
        print(f"{source_name}: " + ", ".join(parts))

    if args.by_request:
        print("\nPer-request summary:")
        req_header = (
            "Source".ljust(20)
            + "Request".ljust(20)
            + "Need".ljust(48)
            + "N".rjust(6)
            + "OK%".rjust(8)
            + "Coverage".rjust(12)
            + "Capacity".rjust(12)
            + "Redundancy".rjust(12)
            + "Sites".rjust(8)
        )
        print(req_header)
        print("-" * len(req_header))
        for (source_name, req_id), summaries in sorted(per_request.items()):
            ok_rate = _avg([1.0 if s["ok"] else 0.0 for s in summaries])
            cov = _avg([s["coverage"] for s in summaries])
            cap = _avg([s["capacity"] for s in summaries])
            red = _avg([s["redundancy_rate"] for s in summaries])
            sites = _avg([s["sites"] for s in summaries])
            request_text = next((s["request_text"] for s in summaries if s.get("request_text")), "")
            print(
                _fit_cell(source_name, 20)
                + _fit_cell(req_id, 20)
                + _fit_cell(request_text, 48)
                + f"{len(summaries):6d}"
                + f"{ok_rate*100:8.1f}"
                + f"{cov:12.4f}"
                + f"{cap:12.4f}"
                + f"{red:12.4f}"
                + f"{sites:8.2f}"
            )

        print("\nPer-request action distribution:")
        for key in sorted(per_request_actions):
            source_name, req_id = key
            counts = per_request_actions[key]
            total = sum(counts.values()) or 1
            request_text = next((s["request_text"] for s in per_request[key] if s.get("request_text")), "")
            parts = [f"{name}:{counts[name]}({counts[name]/total:.2f})" for name in sorted(counts)]
            print(f"{source_name} | {req_id} | {_shorten_text(request_text, 64)}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
