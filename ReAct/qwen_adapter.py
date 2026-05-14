"""
用途:
  直接从本地 Qwen 基座模型目录加载 tokenizer / model，可选叠加 LoRA adapter，并以 chat 方式生成单轮决策结果。

示例命令:
  python ReAct/run_access_point_decision.py \
    --planner qwen \
    --qwen-model-path Qwen2.5-7B \
    --city-map-path dataset/example.png \
    --user-request-path ReAct/requests/task1.txt

  python ReAct/run_access_point_decision.py \
    --planner llamafactory \
    --llamafactory-model Qwen2.5-7B \
    --llamafactory-adapter Qwen/LLaMA-Factory/saves/Qwen2.5-7B/lora/train_xxx \
    --city-map-path dataset/example.png \
    --user-request-path ReAct/requests/task1.txt

参数说明:
  load_qwen_bundle(model_path, device, dtype, adapter_path): 加载并缓存本地 Qwen 模型，可选合并 LoRA adapter。
  call_qwen_chat(model_path, messages, device, dtype, max_new_tokens, do_sample, temperature, top_p, top_k, adapter_path): 执行一次本地 chat 推理。

逻辑说明:
  这个模块只负责本地模型推理，不参与动作解析、候选修复或环境交互。上层把标准 chat messages 传进来，
  这里用 tokenizer 的 chat template 拼 prompt，再解码新增 token，返回模型原始文本。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


_QWEN_CACHE: Dict[Tuple[str, str, str, str], Tuple[object, object, str]] = {}


def clear_qwen_cache() -> None:
    cached_values = list(_QWEN_CACHE.values())
    _QWEN_CACHE.clear()
    for entry in cached_values:
        try:
            _, model, _ = entry
        except Exception:
            continue
        try:
            del model
        except Exception:
            pass

    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported qwen dtype: {dtype}")
    return mapping[dtype]


def load_qwen_bundle(model_path: str, device: str = "mps", dtype: str = "auto", adapter_path: str = ""):
    resolved_dir = Path(model_path).expanduser().resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Qwen model path does not exist: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Qwen model path is not a directory: {resolved_dir}")
    required_files = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_files if not (resolved_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Qwen model directory is incomplete: {resolved_dir}. Missing files: {', '.join(missing)}"
        )

    resolved_adapter = ""
    if adapter_path:
        adapter_dir = Path(adapter_path).expanduser().resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Qwen adapter path does not exist: {adapter_dir}")
        if not adapter_dir.is_dir():
            raise NotADirectoryError(f"Qwen adapter path is not a directory: {adapter_dir}")
        adapter_required = ("adapter_config.json",)
        adapter_missing = [name for name in adapter_required if not (adapter_dir / name).exists()]
        if adapter_missing:
            raise FileNotFoundError(
                f"Qwen adapter directory is incomplete: {adapter_dir}. Missing files: {', '.join(adapter_missing)}"
            )
        resolved_adapter = str(adapter_dir)

    resolved_path = str(resolved_dir)
    resolved_device = _resolve_device(device)
    cache_key = (resolved_path, resolved_device, dtype, resolved_adapter)
    cached = _QWEN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Qwen planner requires transformers, peft and torch in the current environment.") from exc

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": _resolve_dtype(dtype),
        "low_cpu_mem_usage": True,
    }
    if resolved_device == "cuda":
        model_kwargs["device_map"] = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(resolved_path, **model_kwargs)
    if resolved_adapter:
        peft_kwargs = {}
        if resolved_device == "cuda":
            peft_kwargs["device_map"] = {"": 0}
        model = PeftModel.from_pretrained(model, resolved_adapter, **peft_kwargs)
    if resolved_device != "cuda":
        model.to(torch.device(resolved_device))
    model.eval()

    bundle = (tokenizer, model, resolved_device)
    _QWEN_CACHE[cache_key] = bundle
    return bundle


def call_qwen_chat(
    model_path: str,
    messages: List[Dict[str, str]],
    device: str = "mps",
    dtype: str = "auto",
    max_new_tokens: int = 320,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    adapter_path: str = "",
) -> str:
    tokenizer, model, resolved_device = load_qwen_bundle(
        model_path=model_path,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer([prompt], return_tensors="pt")
    encoded = {key: value.to(resolved_device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(float(temperature), 1e-6)
        generation_kwargs["top_p"] = float(top_p)
        generation_kwargs["top_k"] = int(top_k)

    outputs = model.generate(**encoded, **generation_kwargs)
    input_length = int(encoded["input_ids"].shape[-1])
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
