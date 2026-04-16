"""
用途:
  共享 USC 与 RadioMap3DSeer 数据读取逻辑，供训练/微调/评估脚本导入。

直接运行命令:
  无。该文件是公共模块，不单独运行。

导出对象与参数:
  USCDataset(data_root, sample_ids)
    data_root: USC 数据集根目录，内部应包含 map/、Tx/、pmap/。
    sample_ids: 要读取的样本 ID 列表。
  RadioMap3DSeerDataset(data_root, sample_pairs, use_height=True)
    data_root: RadioMap3DSeer 数据集根目录，内部应包含 png/ 与 gain/。
    sample_pairs: [(scene_id, tx_id), ...] 样本列表。
    use_height: True 读取 buildingsWHeight/antennasWHeight；False 读取 buildings_complete/antennas。
  resolve_usc_sample_ids(data_root, csv_file=None)
    data_root: USC 数据集根目录。
    csv_file: 可选样本列表 CSV；为空时优先读 Data_coarse_train.csv，否则扫描 pmap/。
  resolve_radiomap_sample_pairs(data_root, csv_file=None)
    data_root: RadioMap3DSeer 数据集根目录。
    csv_file: 可选 (scene_id, tx_id) CSV；为空时扫描 gain/。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# numeric_sort_key 与 pair_sort_key 分别定义了针对字符串及二元组的排序策略

# 传入sample id，其定义了排序规则，用于后续输入map排序
# sample_ids.sort(key=numeric_sort_key)
def numeric_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)

# 0，sample id，其定义了排序规则，用于后续的输入对map+Tx排序
# sample_pairs.sort(key=pair_sort_key)
def pair_sort_key(pair: tuple[str, str]):
    return numeric_sort_key(pair[0]), numeric_sort_key(pair[1])


def to_tensor_uint8(array: np.ndarray) -> torch.Tensor:
    array = np.array(array, copy=True)
    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0)
    # HWC -> CHW
    elif array.ndim == 3:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")
    # 归一化至 [0,1]
    return tensor.float().div(255.0)


def read_usc_sample_ids(csv_path: Path) -> list[str]:
    sample_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            candidate = row[0].strip()
            if not candidate:
                continue
            if candidate.lower() in {"id", "sample_id"}:
                continue
            if row_idx == 0 and len(row) == 1 and candidate == "0":
                continue
            sample_ids.append(candidate)
    return sample_ids


def discover_usc_sample_ids(data_root: Path) -> list[str]:
    power_dir = data_root / "pmap"
    if not power_dir.exists():
        raise FileNotFoundError(f"Missing directory: {power_dir}")

    sample_ids = [path.stem for path in power_dir.glob("*.png")]
    if not sample_ids:
        raise FileNotFoundError(f"No PNG files found in: {power_dir}")

    sample_ids.sort(key=numeric_sort_key)
    return sample_ids


def resolve_usc_sample_ids(data_root: str, csv_file: str | None = None) -> list[str]:
    root = Path(data_root)
    if csv_file:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        sample_ids = read_usc_sample_ids(csv_path)
    else:
        default_csv = root / "Data_coarse_train.csv"
        if default_csv.exists():
            sample_ids = read_usc_sample_ids(default_csv)
        else:
            # usc: 如果不传 --csv-file，会优先读取 data-root 下的 Data_coarse_train.csv；如果没有，就扫描 pmap/ 里的全部样本
            sample_ids = discover_usc_sample_ids(root)

    valid_sample_ids: list[str] = []
    for sample_id in sample_ids:
        required_paths = [
            root / "map" / f"{sample_id}.png",
            root / "Tx" / f"{sample_id}.png",
            root / "pmap" / f"{sample_id}.png",
        ]
        if all(path.exists() for path in required_paths):
            valid_sample_ids.append(sample_id)

    if not valid_sample_ids:
        raise ValueError("No valid USC sample ids were found.")

    return valid_sample_ids


def read_radiomap_sample_pairs(csv_path: Path) -> list[tuple[str, str]]:
    sample_pairs: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) >= 2:
                scene_id = row[0].strip()
                tx_id = row[1].strip()
            else:
                value = row[0].strip()
                if "_" not in value:
                    continue
                # 最大分割次数（maxsplit）为 1
                scene_id, tx_id = value.split("_", 1)
            if not scene_id or not tx_id:
                continue
            if scene_id.lower() == "scene_id" or tx_id.lower() == "tx_id":
                continue
            sample_pairs.append((scene_id, tx_id))

    sample_pairs.sort(key=pair_sort_key)
    return sample_pairs


def discover_radiomap_sample_pairs(data_root: Path) -> list[tuple[str, str]]:
    gain_dir = data_root / "gain"
    if not gain_dir.exists():
        raise FileNotFoundError(f"Missing directory: {gain_dir}")

    sample_pairs: list[tuple[str, str]] = []
    for path in gain_dir.glob("*.png"):
        if "_" not in path.stem:
            continue
        scene_id, tx_id = path.stem.split("_", 1)
        sample_pairs.append((scene_id, tx_id))

    if not sample_pairs:
        raise FileNotFoundError(f"No valid gain PNG files found in: {gain_dir}")

    sample_pairs.sort(key=pair_sort_key)
    return sample_pairs


def resolve_radiomap_sample_pairs(data_root: str, csv_file: str | None = None) -> list[tuple[str, str]]:
    root = Path(data_root)
    if csv_file:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        sample_pairs = read_radiomap_sample_pairs(csv_path)
    else:
        # radiomap3dseer: 如果不传 --csv-file，会直接扫描 data-root/gain/ 下所有 *.png，自动推断 (scene_id, tx_id)
        sample_pairs = discover_radiomap_sample_pairs(root)

    if not sample_pairs:
        raise ValueError("No RadioMap3DSeer samples were found.")

    return sample_pairs


class USCDataset(Dataset):
    def __init__(self, data_root: str, sample_ids: Iterable[str]):
        self.data_root = Path(data_root)
        self.sample_ids = list(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        image_buildings = np.asarray(Image.open(self.data_root / "map" / f"{sample_id}.png"))
        image_tx = np.asarray(Image.open(self.data_root / "Tx" / f"{sample_id}.png"))
        image_power = np.asarray(Image.open(self.data_root / "pmap" / f"{sample_id}.png"))

        # numpy: HWC
        inputs = np.stack([image_buildings, image_tx], axis=2)
        # to tensor: HWC -> CHW
        return to_tensor_uint8(inputs), to_tensor_uint8(image_power)


class RadioMap3DSeerDataset(Dataset):
    def __init__(self, data_root: str, sample_pairs: list[tuple[str, str]], use_height: bool = True):
        self.data_root = Path(data_root)
        self.sample_pairs = list(sample_pairs)
        self.use_height = use_height

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, idx: int):
        scene_id, tx_id = self.sample_pairs[idx]
        png_root = self.data_root / "png"
        map_subdir = "buildingsWHeight" if self.use_height else "buildings_complete"
        tx_subdir = "antennasWHeight" if self.use_height else "antennas"

        map_path = png_root / map_subdir / f"{scene_id}.png"
        tx_path = png_root / tx_subdir / f"{scene_id}_{tx_id}.png"
        label_path = self.data_root / "gain" / f"{scene_id}_{tx_id}.png"

        image_map = np.asarray(Image.open(map_path))
        image_tx = np.asarray(Image.open(tx_path))
        image_label = np.asarray(Image.open(label_path))

        inputs = np.stack([image_map, image_tx], axis=2)
        return to_tensor_uint8(inputs), to_tensor_uint8(image_label), scene_id, tx_id
