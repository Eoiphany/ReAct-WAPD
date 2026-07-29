"""注释
命令示例:
python -m unittest ReAct.experiments.tests.test_exp2_output_naming

参数含义:
- 无命令行参数；测试直接验证Exp2输出名称生成函数。

逻辑说明:
确认候选配置进入单实验目录名和最终汇总文件名，避免不同配置互相覆盖。
"""

import unittest

from ReAct.experiments.run_exp2_init_decision_matrix import (
    candidate_config_suffix,
    experiment_suite_name,
    summary_output_stem,
)


class Exp2OutputNamingTests(unittest.TestCase):
    def test_candidate_configuration_is_part_of_all_output_names(self) -> None:
        self.assertEqual(candidate_config_suffix(32, 16), "cs32_topk16")
        self.assertEqual(
            experiment_suite_name("two_stage_qwen_decide", 32, 16),
            "two_stage_qwen_decide_cs32_topk16",
        )
        self.assertEqual(
            summary_output_stem("llm", 32, 16),
            "exp2_init_decision_matrix_llm_cs32_topk16",
        )

    def test_different_candidate_configurations_have_different_names(self) -> None:
        self.assertNotEqual(
            experiment_suite_name("two_stage_qwen_decide", 32, 16),
            experiment_suite_name("two_stage_qwen_decide", 16, 8),
        )


if __name__ == "__main__":
    unittest.main()
