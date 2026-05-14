# 修复代理模型训练可视化图片闪烁问题

## 问题描述

在代理模型训练过程中，训练指标图表（Best Val RMSE、Training Loss等）出现持续闪烁，在"图片加载中"和"显示图片"之间不断切换，即使训练数据没有更新也会发生。

### 用户观察到的现象
- 训练到epoch 3时，前端仍然只显示epoch 1的数据
- 图片在"加载中"和"显示"状态之间快速切换
- 必须手动刷新页面才能看到最新的训练曲线
- 控制台显示"Card not found for index 0"错误

## 根本原因分析

### 1. 后端问题：图片过度生成
**位置**: `visualization/web_runtime.py` 的 `_refresh_artifacts_locked()` 方法

**原因**: 
- 该方法每0.5秒被SSE轮询调用一次
- 之前的代码每次调用都会重新生成图片，即使`history.json`文件内容没有变化
- 虽然已经有`data_changed`检查逻辑，但缺少调试日志，难以确认是否正常工作

**影响**:
- 即使训练数据没有更新，`metric_version`也会因为文件修改时间变化而改变
- 导致前端认为有新数据，触发图片重新加载

### 2. 前端问题：DOM卡片丢失
**位置**: `visualization/web_static/index.html` 的 `renderSurrogateMetricGridFromUrls()` 函数

**原因**:
- 函数假设卡片已经通过`renderSurrogateMetricGrid()`初始化
- 但在某些情况下（如页面状态切换），卡片可能被清空
- 只检查`host.children.length === 0`不够，因为可能存在没有`data-metric-index`属性的卡片

**影响**:
- `querySelector('[data-metric-index="0"]')`返回null
- 导致"Card not found"错误，图片无法更新

## 解决方案

### 1. 后端优化：添加调试日志
**文件**: `visualization/web_runtime.py`

**修改**:
```python
if data_changed:
    # 数据变化了，重新生成图片和 metric_version
    stat = history_path.stat()
    self.metric_version = f"{int(stat.st_mtime_ns)}:{self.history_length}:{self.latest_epoch}"
    title_prefix = f"{SURROGATE_DATASET_LABELS.get(str(self.payload.get('datasetKey')), '')} / {SURROGATE_MODEL_LABELS.get(str(self.payload.get('modelType')), '')}"
    self.metric_plots = _render_surrogate_metric_panels(history_path, self.runtime_dir / "metric_panels", title_prefix)
    print(f"[surrogate] Metric data changed: epoch={self.latest_epoch}, history_length={self.history_length}, regenerated images, new metric_version={self.metric_version}")
else:
    print(f"[surrogate] Metric data unchanged: epoch={self.latest_epoch}, history_length={self.history_length}, keeping existing metric_version={self.metric_version}")
```

**效果**:
- 清楚显示何时重新生成图片
- 确认缓存机制是否正常工作
- 帮助诊断是否有不必要的图片重新生成

### 2. 前端增强：确保卡片始终存在
**文件**: `visualization/web_static/index.html`

**修改**:
```javascript
function renderSurrogateMetricGridFromUrls(plots) {
  const host = byId("surrogate-train-view");
  
  // 确保卡片已初始化
  if (host.children.length === 0) {
    console.log("[Surrogate] Initializing metric cards");
    SURROGATE_METRIC_TITLES.forEach((title, index) => {
      const card = buildMetricCard(title, "");
      card.dataset.metricIndex = index;
      host.appendChild(card);
    });
  }
  
  // 验证所有卡片都有 data-metric-index 属性
  const cardsWithIndex = host.querySelectorAll("[data-metric-index]");
  if (cardsWithIndex.length !== SURROGATE_METRIC_TITLES.length) {
    console.warn(`[Surrogate] Card count mismatch! Expected ${SURROGATE_METRIC_TITLES.length}, found ${cardsWithIndex.length}. Reinitializing...`);
    host.innerHTML = "";
    SURROGATE_METRIC_TITLES.forEach((title, index) => {
      const card = buildMetricCard(title, "");
      card.dataset.metricIndex = index;
      host.appendChild(card);
    });
  }
  
  // ... 后续更新逻辑
}
```

**效果**:
- 双重检查确保卡片存在且有正确的属性
- 如果卡片丢失或不完整，自动重新初始化
- 防止"Card not found"错误

## 预期效果

修复后的行为：

1. **后端**:
   - 只在epoch完成、`history.json`更新时重新生成图片
   - `metric_version`只在数据真正变化时更新
   - 控制台清楚显示何时重新生成图片

2. **前端**:
   - 卡片始终存在且有正确的`data-metric-index`属性
   - 只在`metric_version`变化时更新图片URL
   - 图片平滑更新，不再闪烁

3. **用户体验**:
   - 训练曲线实时更新，无需手动刷新
   - 每个epoch完成后自动显示最新数据
   - 图片加载流畅，无闪烁

## 测试建议

1. 启动代理模型训练
2. 观察控制台日志：
   - 后端应该只在epoch完成时打印"Metric data changed"
   - 前端应该只在`metric_version`变化时更新图片
3. 确认图片在epoch完成后自动更新
4. 确认没有"Card not found"错误
5. 确认图片不再闪烁

## 相关文件

- `visualization/web_runtime.py` - 后端状态管理和图片生成
- `visualization/web_static/index.html` - 前端渲染逻辑
- `visualization/SYSTEM_ARCHITECTURE.md` - 系统架构文档
- `visualization/PAPER_VISUALIZATION_SECTION.md` - 论文可视化章节
