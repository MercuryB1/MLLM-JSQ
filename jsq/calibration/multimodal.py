"""Multimodal (image + text) calibration dataset loaders for MLLM compression.

For MLLM calibration the goal is NOT to process images in the compression code.
Instead, we feed the image-text batches through the full model forward pass
(ViT FP16 → merger → embed fusion), and let the Catcher hook capture the
fused hidden states arriving at the first LLM decoder block.

Returned batches are dicts with keys expected by the model's forward():
    - input_ids
    - pixel_values (or pixel_values_videos for Qwen2-VL)
    - attention_mask
    - image_grid_thw (Qwen2-VL specific)
"""
import random
from typing import Dict, List

import torch
from loguru import logger


def _get_coco_captions(processor, n_samples: int, seed: int) -> List[Dict]:
    """Load COCO Captions val2017 image-text pairs."""
    from datasets import load_dataset

    dataset = load_dataset("shunk031/MSCOCO", "2017_captions", split="validation",
                           trust_remote_code=True)
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = []
    for idx in indices:
        item = dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["captions"]["raw_annotation"][0]["caption"]
        # Build a simple VQA-style prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": caption},
            ]}
        ]
        samples.append(messages)
    return samples


def _get_sharegpt4v(processor, n_samples: int, seed: int) -> List[Dict]:
    """Load ShareGPT4V image-text pairs (requires HF access)."""
    from datasets import load_dataset

    dataset = load_dataset("Lin-Chen/ShareGPT4V", split="train",
                           trust_remote_code=True)
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = []
    for idx in indices:
        item = dataset[idx]
        image = item["image"].convert("RGB") if item.get("image") else None
        text = item.get("caption", item.get("conversations", [{}])[0].get("value", ""))
        if image is None:
            continue
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]}
        ]
        samples.append(messages)
    return samples


def _process_qwen2vl_batch(messages_list: List, processor) -> List[Dict]:
    """Process a list of Qwen2-VL message dicts into model input dicts."""
    from qwen_vl_utils import process_vision_info

    batches = []
    for messages in messages_list:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
        )
        batches.append({k: v for k, v in inputs.items()})
    return batches


def get_multimodal_calib_data(
    dataset: str,
    processor,
    model_type: str,
    n_samples: int = 128,
    seed: int = 42,
) -> List[Dict]:
    """Return a list of model-input dicts for multimodal calibration.

    Each element is a dict that can be unpacked as **kwargs into model.forward().
    The list length equals n_samples (or fewer if the dataset is smaller).

    Supported datasets: coco_captions, sharegpt4v.
    Supported model_type: qwen2_vl.
    """
    logger.info(
        f"Loading multimodal calibration data: {dataset} "
        f"({n_samples} samples, model_type={model_type})"
    )

    if dataset == "coco_captions":
        raw = _get_coco_captions(processor, n_samples, seed)
    elif dataset == "sharegpt4v":
        raw = _get_sharegpt4v(processor, n_samples, seed)
    else:
        raise ValueError(
            f"Unknown multimodal calibration dataset: '{dataset}'. "
            f"Choose from: coco_captions, sharegpt4v"
        )

    if model_type == "qwen2_vl":
        return _process_qwen2vl_batch(raw, processor)
    else:
        raise ValueError(
            f"Multimodal calibration not yet implemented for model_type='{model_type}'. "
            f"Currently supported: qwen2_vl"
        )
