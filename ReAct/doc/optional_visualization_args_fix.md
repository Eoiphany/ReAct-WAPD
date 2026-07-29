# 可视化同步参数缺失修复验证

## 问题

旧版实验调用方构造的`argparse.Namespace`可能缺少`visualization_sync_dir`和`visualization_sync_timeout_sec`。`run_access_point_decision.run_task()`原先直接访问这两个字段，即使没有启用可视化同步，也会抛出`AttributeError`。

本地当前`summary_utils.make_suite_args()`已经包含这两个字段，因此服务器再次报错说明其相关文件很可能尚未同步到当前版本；仅凭本地文件不足以确认服务器具体版本。

## 修改

在`run_access_point_decision.py`中新增`resolve_visualization_sync_settings()`：

- Namespace缺少字段时返回`("", 0.0)`，即关闭同步且不等待。
- Namespace显式提供字段时保留原配置。
- `run_task()`在入口只解析一次，初始快照和逐步快照共用该结果。

## 验证结果

```text
python -m unittest ReAct.experiments.tests.test_optional_visualization_args
Ran 2 tests in 0.000s
OK

python -m py_compile ReAct/run_access_point_decision.py ReAct/experiments/tests/test_optional_visualization_args.py
exit code: 0

缺字段Namespace解析结果:
('', 0.0)
```

未在服务器执行完整100图实验，因此真实服务器版本及完整运行结果仍需服务器侧重新同步代码后验证。
