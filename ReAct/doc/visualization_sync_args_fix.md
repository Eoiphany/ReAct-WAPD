# 批量实验可视化同步参数修复验证

## 问题

`run_experiment_suite.py` 创建的 `argparse.Namespace` 缺少 `visualization_sync_dir` 和 `visualization_sync_timeout_sec`，但其调用的 `run_access_point_decision.run_task()` 会直接读取这两个字段，导致首个批量任务抛出 `AttributeError`。

## 修改

在批量实验 parser 中补充与单任务 parser 完全一致的两个参数：

- `--visualization-sync-dir`，默认值为 `""`。
- `--visualization-sync-timeout-sec`，类型为 `float`，默认值为 `0.0`。

`_build_task_args()` 继续复用原有的命名空间复制逻辑，不修改单任务执行主流程。

## 验证结果

```text
python -m unittest ReAct.test.test_experiment_suite_visualization_args
Ran 1 test in 0.001s
OK

python -m py_compile ReAct/run_experiment_suite.py ReAct/test/test_experiment_suite_visualization_args.py
exit code: 0

python -c "from ReAct.run_experiment_suite import build_parser; args = build_parser().parse_args([]); assert args.visualization_sync_dir == '' and args.visualization_sync_timeout_sec == 0.0"
exit code: 0
```

未执行完整的100任务实验，因为本地环境不具备报错日志中服务器侧的地图、模型权重和运行配置；当前验证覆盖了触发异常的参数构造与透传边界。
