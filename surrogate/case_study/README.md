# Case Study

这个目录是一个面向 `RadioMap3DSeer` 风格数据的最小合成链路，用程序化建筑场景 + Blender/Sionna 兼容导出 + RMNet 评估，快速迭代一张 `256 x 256`、`1 m/pixel` 的 3D radio map 样本。

## 目录用途

- `synthetic_scene_generator.py`
  生成程序化场景。负责输出：
  - `buildingsWHeight`
  - `antennasWHeight`
  - 建筑 mesh
  - metadata
  - Blender manifest

- `build_sionna_scene_xml.py`
  根据生成器导出的 manifest，把 mesh 列表写成 Mitsuba/Sionna 可加载的 `scene.xml`。

- `run_case_study_sionna.py`
  用 Sionna RT 对当前场景做覆盖仿真，并导出：
  - `gain`
  - `path_gain_db.npy`
  - `rx_power_dbm.npy`
  - `snr_db.npy`
  - `stats_case_study.txt`

- `blender_import_synthetic_scene.py`
  Blender 侧的导入逻辑，把 manifest 中记录的建筑 mesh 和地面恢复成 Blender 场景。

- `run_blender_import.py`
  Blender 导入的启动包装脚本，方便从命令行直接触发。

- `case_study_paths.py`
  统一维护当前子项目里常用路径常量，避免在各脚本里重复硬编码目录。

- `RUN_STEPS_FROM_SCRATCH.md`
  从零开始的执行步骤说明，包含推荐命令和当前默认参数。

- `search_runs/`
  历史实验结果目录。这里保留的是前面做过的场景布局、阈值和标签导出对比，不参与主链路运行，但便于回溯。

- `output_dataset/`
  当前主结果目录，也是各脚本默认读写的位置。

## `output_dataset/` 里的模块

- `png/`
  当前主样本的输入图：
  - `buildingsWHeight`
  - `antennasWHeight`

- `gain/`
  当前导出的 8-bit radio map 标签图。

- `metadata/`
  当前样本的元信息，包括：
  - 场景配置
  - `tx_position_m`
  - `building_count`
  - `building_ratio`

- `meshes/`
  生成的建筑 PLY mesh，供 Blender/Sionna 场景恢复使用。

- `blender/`
  Blender manifest，以及可选导入后保存的 `.blend` 文件。

- `sionna/`
  当前用于仿真的 `synthetic_scene.xml`。

- `analysis/`
  RMNet 预测分析结果，包括：
  - 预测图
  - 标签图
  - 误差图
  - 三联总览图
  - JSON 指标摘要

- `path_gain_db.npy`
  Sionna 计算得到的 dB 域 path gain 原始栅格。

- `rx_power_dbm.npy`
  在 path gain 基础上加上 Tx power 之后的接收功率。

- `snr_db.npy`
  按带宽和噪声系数计算得到的 SNR 栅格。

- `stats_case_study.txt`
  当前主运行的统计摘要，包含：
  - Tx / Rx / 频率 / 带宽
  - path gain 分位数
  - `PL threshold`
  - `analytic truncation threshold`
  - 最终导出文件路径

## 当前主链路假设

- 地图尺寸：`256 x 256`
- 分辨率：`1 m/pixel`
- Rx height：`1.5 m`
- Tx power：`23 dBm`
- Tx height：`rooftop + 3 m`
- Carrier：`3.5 GHz`
- Bandwidth：`20 MHz`
- Noise figure：`20 dB`
- Max interactions：`2`
- Building material：统一 `generic` 材质；当前主链路默认导出为 `uniform_concrete`，不再按楼交替材质
- 当前主导出参数：
  - `db_min = -111.25`
  - `db_max = -52`
  - `label_gain = 0.92`
  - `label_gamma = 0.92`

## 建议使用顺序

1. 跑 `synthetic_scene_generator.py`
2. 跑 `build_sionna_scene_xml.py`
3. 跑 `run_case_study_sionna.py`
4. 跑 `paper_experiment.surrogate.test.export_case_study_rmnet_analysis`

如果要看完整命令，直接参考 `RUN_STEPS_FROM_SCRATCH.md`。
