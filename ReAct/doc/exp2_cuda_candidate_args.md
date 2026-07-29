# Exp2候选参数与CUDA默认值验证

## 修改内容

- `run_exp2_init_decision_matrix.py`新增`--candidate-sample`，默认值为`32`。
- 新增`--llm-top-k-candidates`，默认值为`16`。
- 两个参数同时透传到启发式组和LLM交叉矩阵组的`make_suite_args()`。
- `--eval-device`默认值由`mps`改为`cuda`。`make_suite_args()`同时令`qwen_device=eval_device`，因此PMNet/RMNet、模型驱动初始化、Qwen和LLaMA-Factory均请求CUDA。
- 单实验目录和最终汇总文件均加入`cs<候选数>_topk<保留数>`后缀，避免不同候选配置覆盖旧结果。

当参数为`32/16`时，`two_stage + qwen + decide`对应的目录名为`two_stage_qwen_decide_cs32_topk16`，LLM组汇总文件前缀为`exp2_init_decision_matrix_llm_cs32_topk16`。

## 命令示例

```bash
python -m ReAct.experiments.run_exp2_init_decision_matrix \
  --group llm \
  --candidate-sample 32 \
  --llm-top-k-candidates 16
```

## 验证结果

```text
python -m unittest ReAct.test.test_experiment_scripts.ExperimentScriptTests.test_exp2_parser_supports_group_switch ReAct.test.test_experiment_scripts.ExperimentScriptTests.test_exp2_parser_defaults_models_to_cuda ReAct.test.test_experiment_scripts.ExperimentScriptTests.test_exp2_parser_accepts_candidate_overrides ReAct.test.test_experiment_scripts.ExperimentScriptTests.test_exp2_candidate_config_suffix_distinguishes_output_names
Ran 4 tests in 0.001s
OK

python -m py_compile ReAct/experiments/run_exp2_init_decision_matrix.py ReAct/test/test_experiment_scripts.py
exit code: 0

参数检查输出:
{'candidate_sample': 32, 'llm_top_k_candidates': 16, 'eval_device': 'cuda'}
```

完整`ReAct.test.test_experiment_scripts`仍有一个与本次修改无关的既存失败：测试期望`1.70 bps/Hz`，当前模板输出`1.85 bps/Hz`。本次未修改该需求阈值或旧测试。

当前本机没有NVIDIA CUDA设备，因此未验证真实显存占用；服务器运行时仍需通过`nvidia-smi`确认。
