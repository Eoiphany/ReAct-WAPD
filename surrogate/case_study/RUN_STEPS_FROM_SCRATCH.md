# Case Study From Scratch

在项目根目录 `/Users/epiphanyer/Desktop/coding` 下执行。

## 环境

这条链路里的 Sionna 安装在 `autobs` 环境中。  
如果你手动激活环境，可以直接用 `python -m ...`；如果不想切环境，就直接用下面给出的解释器绝对路径。

```bash
cd /Users/epiphanyer/Desktop/coding
```

## 1. 生成程序化场景

```bash
/Users/epiphanyer/Miniconda/envs/autobs/bin/python -m paper_experiment.surrogate.case_study.synthetic_scene_generator
```

当前默认生成器已经切到一组“更大楼体、更少数量”的参数，并修正到更接近论文 3D 数据集的规则：
- `target_occupancy=0.24`
- `road_width_m=12`
- `road_width_jitter_m=2`
- `min_buildings_per_block=2`
- `max_buildings_per_block=4`
- `building_spacing_m=4.5`
- `footprint_min_edge_m=12`
- `footprint_max_edge_m=32`
- `annex_probability=0.08`
- `annex_max_count=1`
- building heights are sampled from the dataset-style `255` discrete levels in `[6.6, 19.8] m`
- Tx is placed on a rooftop of height at least `16.5 m`, within the center `150 x 150` if possible, and at `rooftop + 3 m`

输出：
- `paper_experiment/surrogate/case_study/output_dataset/png/buildingsWHeight/synthetic_scene.png`
- `paper_experiment/surrogate/case_study/output_dataset/png/antennasWHeight/synthetic_scene_0.png`
- `paper_experiment/surrogate/case_study/output_dataset/metadata/synthetic_scene_0.json`
- `paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0_manifest.json`

## 2. 生成 Sionna 场景 XML

```bash
/Users/epiphanyer/Miniconda/envs/autobs/bin/python -m paper_experiment.surrogate.case_study.build_sionna_scene_xml
```

输出：
- `paper_experiment/surrogate/case_study/output_dataset/sionna/synthetic_scene.xml`
- 当前默认会把全部建筑统一为 `uniform_concrete` 材质，而不是交替材质

## 3. 运行 Sionna 仿真并导出 gain

当前默认配置已经对齐到论文 3D 数据集的关键参数：
- `256 x 256`
- `1 m / pixel`
- `3.5 GHz`
- `23 dBm`
- `20 MHz`
- `noise figure = 20 dB`
- `max_depth = 2`
- `dataset truncation range ≈ [-111.25, -52] dB`
- `label_gain = 0.92`
- `label_gamma = 0.92`

```bash
/Users/epiphanyer/Miniconda/envs/autobs/bin/python -m paper_experiment.surrogate.case_study.run_case_study_sionna \
  --max-depth 2
```

输出：
- `paper_experiment/surrogate/case_study/output_dataset/gain/synthetic_scene_0.png`
- `paper_experiment/surrogate/case_study/output_dataset/path_gain_db.npy`
- `paper_experiment/surrogate/case_study/output_dataset/rx_power_dbm.npy`
- `paper_experiment/surrogate/case_study/output_dataset/snr_db.npy`
- `paper_experiment/surrogate/case_study/output_dataset/stats_case_study.txt`

## 4. 导出 RMNet 分析图

```bash
/Users/epiphanyer/Miniconda/envs/autobs/bin/python -m paper_experiment.surrogate.test.export_case_study_rmnet_analysis
```

输出：
- `paper_experiment/surrogate/case_study/output_dataset/analysis/synthetic_scene_0_rmnet_analysis.png`
- `paper_experiment/surrogate/case_study/output_dataset/analysis/synthetic_scene_0_building_tx_overview.png`
- `paper_experiment/surrogate/case_study/output_dataset/analysis/synthetic_scene_0_rmnet_analysis.json`

## 5. 可选：导入 Blender 检查场景

如果你想把同一份 manifest 导入 Blender：

```bash
/Users/epiphanyer/Miniconda/envs/autobs/bin/python -m paper_experiment.surrogate.case_study.run_blender_import
```

输出：
- `paper_experiment/surrogate/case_study/output_dataset/blender/synthetic_scene_0.blend`

## 6. 最小验证命令

```bash
/Users/epiphanyer/Miniconda/envs/autobs/bin/python -m py_compile \
  /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/synthetic_scene_generator.py \
  /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/build_sionna_scene_xml.py \
  /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/case_study/run_case_study_sionna.py \
  /Users/epiphanyer/Desktop/coding/paper_experiment/surrogate/test/export_case_study_rmnet_analysis.py
```
