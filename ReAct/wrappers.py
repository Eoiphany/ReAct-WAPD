"""
用途:
  轨迹包装器。HistoryWrapper 负责把历史动作拼回 observation，LoggingWrapper 负责把整条轨迹写入 JSON。

示例命令:
  无。该文件是公共模块，供主入口导入。

参数说明:
  HistoryWrapper(env, obs_format, prompt=None)
    obs_format: obs 或 history。
    prompt: 追加到 history observation 前面的提示词。
  LoggingWrapper(env, folder='outputs/trajs', file_id=None)
    folder: 轨迹输出目录。
    file_id: 输出文件名，不含 .json。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

try:
    import gymnasium as gym
except ModuleNotFoundError:
    class _SimpleWrapper:
        def __init__(self, env):
            self.env = env

        def reset(self, *args, **kwargs):
            return self.env.reset(*args, **kwargs)

        def step(self, *args, **kwargs):
            return self.env.step(*args, **kwargs)

        def close(self):
            if hasattr(self.env, "close"):
                return self.env.close()
            return None

    class _SimpleObservationWrapper(_SimpleWrapper):
        def observation(self, obs):
            return obs

    class _SimpleGym:
        Wrapper = _SimpleWrapper
        ObservationWrapper = _SimpleObservationWrapper

    gym = _SimpleGym()


class HistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_format, prompt=None):
        super().__init__(env)
        assert obs_format in ["obs", "history"]
        self.obs_format = obs_format
        self.prompt = prompt if prompt is not None else ""

    def observation(self, obs):
        if self.obs_format == "obs":
            return obs
        observation = self.env.traj["observations"][0] + "\n"
        rationales = self.env.traj.get("rationales", [])
        for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
            rationale = rationales[i - 1] if (i - 1) < len(rationales) else ""
            if rationale:
                observation += f"Reason {i}: {rationale}\n"
            observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
        return self.prompt + observation


class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, folder="outputs/trajs", file_id=None):
        super().__init__(env)
        self.trajs = []
        self.traj = {"observations": [], "actions": [], "rationales": []}
        self.folder = folder
        self.file_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if file_id is None else file_id
        self.file_path = os.path.join(self.folder, f"{self.file_id}.json")
        os.makedirs(self.folder, exist_ok=True)
        self.last_rationale = ""

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
        if isinstance(output, tuple) and len(output) == 2:
            observation, info = output
        else:
            observation, info = output, {}
        self.traj = {"observations": [observation], "actions": [], "rationales": []}
        return observation, info

    def step(self, action):
        output = self.env.step(action)
        if isinstance(output, tuple) and len(output) == 5:
            obs, reward, terminated, truncated, info = output
            done = terminated or truncated
        else:
            obs, reward, done, info = output
            terminated, truncated = done, False
        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)
        self.traj["rationales"].append(self.last_rationale or "")
        self.last_rationale = ""
        if done:
            self.traj.update(info)
        return obs, reward, terminated, truncated, info

    def update_record(self):
        if len(self.traj) > 0:
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": [], "rationales": []}

    def write(self):
        self.update_record()
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.trajs, f, ensure_ascii=True, indent=2)

    def close(self):
        self.write()
