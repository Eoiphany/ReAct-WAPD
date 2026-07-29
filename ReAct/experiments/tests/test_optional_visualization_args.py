"""注释
命令示例:
python -m unittest ReAct.experiments.tests.test_optional_visualization_args

参数含义:
- 无命令行参数；测试使用缺少可视化同步字段的旧版Namespace。

逻辑说明:
验证可选的可视化同步参数缺失时自动采用关闭同步的默认值，避免旧实验调用方崩溃。
"""

import argparse
import unittest

from ReAct import run_access_point_decision


class OptionalVisualizationArgsTests(unittest.TestCase):
    def test_missing_visualization_args_disable_synchronization(self) -> None:
        sync_dir, timeout_sec = run_access_point_decision.resolve_visualization_sync_settings(
            argparse.Namespace()
        )

        self.assertEqual(sync_dir, "")
        self.assertEqual(timeout_sec, 0.0)

    def test_explicit_visualization_args_are_preserved(self) -> None:
        sync_dir, timeout_sec = run_access_point_decision.resolve_visualization_sync_settings(
            argparse.Namespace(
                visualization_sync_dir="/tmp/visualization-sync",
                visualization_sync_timeout_sec=2.5,
            )
        )

        self.assertEqual(sync_dir, "/tmp/visualization-sync")
        self.assertEqual(timeout_sec, 2.5)


if __name__ == "__main__":
    unittest.main()
