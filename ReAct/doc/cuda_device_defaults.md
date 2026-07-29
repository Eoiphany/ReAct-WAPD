# 批量实验CUDA设备检查与验证

## 检查结论

`run_experiment_suite.py`原先将`eval_device`和`qwen_device`均默认设置为`mps`，因此通过该脚本启动批量任务时，PMNet/RMNet、模型驱动初始化和本地大模型推理不会默认使用CUDA。

设备参数的实际作用范围如下：

- `eval_device`：控制PMNet/RMNet代理模型推理、Greedy初始化、启发式初始化、二阶段策略初始化，以及LLaMA-Factory模型推理。
- `qwen_device`：控制`planner=qwen`时的本地Qwen模型推理。
- `planner=openai`：调用远程API，不涉及本地CUDA设备。
- `init_mode=random`：仅执行NumPy随机采样，不加载神经网络模型。
- `eval_model=proxy`：采用NumPy计算，不使用GPU。

## 修改

- 将`--eval-device`默认值从`mps`改为`cuda`。
- 将`--qwen-device`默认值从`mps`改为`cuda`。
- 将文件头命令示例和参数说明同步更新为CUDA口径。

## 验证结果

```text
python -m unittest ReAct.test.test_experiment_suite_visualization_args
Ran 2 tests in 0.001s
OK

python -m py_compile ReAct/run_experiment_suite.py ReAct/test/test_experiment_suite_visualization_args.py
exit code: 0

python -c "from pathlib import Path; from ReAct.run_experiment_suite import build_parser, _build_task_args; a=build_parser().parse_args([]); t=_build_task_args(a, {'city_map_path':'/tmp/8.png','user_request_path':'/tmp/request.txt'}, Path('/tmp/trajs')); print({'eval_device':t.eval_device,'qwen_device':t.qwen_device}); assert t.eval_device == t.qwen_device == 'cuda'"
{'eval_device': 'cuda', 'qwen_device': 'cuda'}
```

当前本机没有服务器侧NVIDIA CUDA运行环境和模型权重，因此这里只验证了参数默认值、任务透传链路和Python语法；真实GPU占用仍需在服务器运行时通过日志或`nvidia-smi`确认。
