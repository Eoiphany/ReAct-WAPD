from typing import Optional

import numpy as np
import torch

from PMNet import PMNet
from RMNet import RMNet


DB_MIN = -162.0
DB_MAX = -75.0
GROUND_TX_NORM = 0.5
GROUND_TX_GRAY = int(round(GROUND_TX_NORM * 255.0))


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _numpy_image_to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        tensor = torch.tensor(array.tolist(), dtype=torch.float32).unsqueeze(0)
    elif array.ndim == 3:
        tensor = torch.tensor(array.tolist(), dtype=torch.float32).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return tensor / 255.0


def init_model(network_type: str) -> torch.nn.Module:
    if network_type == "pmnet":
        return PMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
        )
    if network_type == "pmnet_v3":
        return PMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    if network_type == "rmnet":
        return RMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
        )
    if network_type == "rmnet_v3":
        return RMNet(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    raise ValueError(f"Unsupported network type: {network_type}")


def infer_checkpoint_family(state: dict) -> Optional[str]:
    if not isinstance(state, dict):
        return None

    keys = list(state.keys())
    if any(key.startswith("module.") for key in keys):
        keys = [key.replace("module.", "", 1) for key in keys]

    if any(key.startswith("fuse") or key.startswith("context.") or key.startswith("input_fuse.") for key in keys):
        return "rmnet"

    if any(key.startswith("fc1.") or key.startswith("conv_up") for key in keys):
        return "pmnet"

    return None


def load_model(model_path: str, network_type: str, device: torch.device) -> torch.nn.Module:
    model = init_model(network_type)
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        inferred_family = infer_checkpoint_family(state)
        hint = ""
        if inferred_family is not None and inferred_family != network_type and not network_type.startswith(inferred_family):
            hint = f" Checkpoint looks like `{inferred_family}` weights, but `--network_type` is `{network_type}`."
        raise RuntimeError(f"{exc}{hint}") from exc

    model.to(device)
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


def _to_db_scale(array: np.ndarray, db_min: float = DB_MIN, db_max: float = DB_MAX) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return array * (db_max - db_min) + db_min


def encode_tx_gray_value(height_gray: int) -> np.uint8:
    """
    Encode the transmitter pixel for the second channel.

    Rule used in this project:
    - If the selected location already has a non-zero height-map gray value,
      reuse it directly.
    - If the selected location is ground / ROI with gray value 0, use the
      midpoint gray value instead of 0.

    The midpoint corresponds to the mean building-height fraction 0.5. After
    adding the 3 m transmitter height, the transmitter-height range remains at
    the same normalized midpoint, so the fallback gray code is still 0.5.
    """
    value = int(height_gray)
    if value > 0:
        return np.uint8(value)
    return np.uint8(GROUND_TX_GRAY)
