# Exp2候选配置对比图验证

## 图表契约

- 核心结论：在100个相同任务上，增大候选采样数和Top-k保留数提高了OK率、覆盖率和平均频谱效率，但增加了平均轨迹处理时间。
- 图表类型：四联定量比较图。
- 数据来源：两个实验目录各自的`summary.json`。
- 正式输出：可编辑文本的SVG；中文使用宋体候选字体，西文使用Times New Roman。
- 面板映射：OK率、覆盖率、平均频谱效率、平均轨迹处理时间。
- 统计边界：图中为100个任务的汇总均值，当前结果没有提供多随机种子置信区间，因此不绘制误差条，也不作显著性声明。

## 数据口径

`cs16/topk8`对应`TSPL+SFT-ReAct`，`cs32/topk16`对应`TSPL+SFT-ReAct-Plus`。前三项指标越高越优，平均轨迹处理时间越低越优。时间严格按`perf.runtime_total_sec / perf.tasks_count`计算，并与`perf.runtime_mean_sec`交叉校验；不计入`perf.suite_runtime_sec`包含的套件调度与墙钟额外开销。

## 运行命令

```bash
python -m ReAct.test.plot_exp2_candidate_comparison
```

## 验证结果

```text
python -m py_compile ReAct/test/plot_exp2_candidate_comparison.py
exit code: 0

python -m ReAct.test.plot_exp2_candidate_comparison
exit code: 0
输出SVG: ReAct/plot/react_wapd_doc_tables/specific_requirement.svg

cs16/topk8: OK率=89.00%, 覆盖率=94.55%, 平均频谱效率=2.18 bps/Hz, 平均轨迹处理时间=39.25 s
cs32/topk16: OK率=93.00%, 覆盖率=94.82%, 平均频谱效率=2.22 bps/Hz, 平均轨迹处理时间=40.46 s
```

SVG结构检查确认四个面板和全部关键数值均保留为可编辑文本；同一Python后端生成的临时PNG预览未见标签重叠或面板布局异常。
