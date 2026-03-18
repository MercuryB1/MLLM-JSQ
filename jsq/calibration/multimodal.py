"""Multimodal (image + text) calibration dataset loaders for MLLM compression."""
import os
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger


_GQA_ID2IMAGE: Optional[Dict] = None
_COCO_DS = None



class _GQADataset(Dataset):
    def __init__(self, q_ds, id2image, indices):
        self.q_ds = q_ds
        self.id2image = id2image
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.q_ds[self.indices[i]]
        image = self.id2image[item["imageId"]]
        text = f"{item['question']}\nAnswer the question using a single word or phrase."
        return {"image": image, "text": text}


class _ListDataset(Dataset):
    """Generic dataset wrapping a pre-built list of {"image", "text"} dicts."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def _collate_fn(batch):
    return {
        "images": [s["image"] for s in batch],
        "texts":  [s["text"]  for s in batch],
    }



def _build_gqa_dataset(n_samples: int, seed: int, data_dir: str) -> Dataset:
    global _GQA_ID2IMAGE
    from datasets import load_dataset

    if _GQA_ID2IMAGE is None:
        logger.info("Building GQA imageId→image cache...")
        img_ds = load_dataset(
            "lmms-lab/GQA", "testdev_balanced_images",
            split="testdev", cache_dir=data_dir,
        )
        _GQA_ID2IMAGE = {row["id"]: row["image"].convert("RGB") for row in img_ds}

    q_ds = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_instructions",
        split="testdev", cache_dir=data_dir,
    )
    random.seed(seed)
    indices = random.sample(range(len(q_ds)), min(n_samples, len(q_ds)))
    return _GQADataset(q_ds, _GQA_ID2IMAGE, indices)


def _build_coco_dataset(n_samples: int, seed: int, data_dir: str) -> Dataset:
    global _COCO_DS
    from datasets import load_dataset

    if _COCO_DS is None:
        logger.info("Loading COCO Caption2017 validation set...")
        _COCO_DS = load_dataset(
            "shunk031/MSCOCO", "2017_captions",
            split="validation", cache_dir=data_dir, trust_remote_code=True,
        )

    random.seed(seed)
    indices = random.sample(range(len(_COCO_DS)), min(n_samples, len(_COCO_DS)))
    samples = [
        {"image": _COCO_DS[i]["image"].convert("RGB"),
         "text": "Provide a one-sentence caption for the provided image."}
        for i in indices
    ]
    return _ListDataset(samples)


def _build_sharegpt4v_dataset(n_samples: int, seed: int, data_dir: str) -> Dataset:
    from datasets import load_dataset

    ds = load_dataset(
        "Lin-Chen/ShareGPT4V", split="train",
        cache_dir=data_dir, trust_remote_code=True,
    )
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    samples = []
    for i in indices:
        item = ds[i]
        image = item["image"].convert("RGB") if item.get("image") else None
        if image is None:
            continue
        text = item.get("caption", item.get("conversations", [{}])[0].get("value", ""))
        samples.append({"image": image, "text": text})
    return _ListDataset(samples)



def _make_messages(text: str) -> List[Dict]:
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text},
    ]}]


def _dataloader_to_batches(dataset: Dataset, processor, batch_size: int) -> List[Dict]:
    """Iterate DataLoader and process each batch with the model processor."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
    )

    batches = []
    for batch in dataloader:
        texts = [
            processor.apply_chat_template(
                _make_messages(t), tokenize=False, add_generation_prompt=True
            )
            for t in batch["texts"]
        ]
        inputs = processor(
            images=batch["images"],
            text=texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        batches.append({k: v for k, v in inputs.items()})
    return batches



def get_multimodal_calib_data(
    dataset: str,
    processor,
    model_type: str,
    n_samples: int = 128,
    seed: int = 42,
    data_dir: str = "storage/datasets",
    batch_size: int = 1,
) -> List[Dict]:
    """Return a list of model-input dicts for multimodal calibration.

    Supported datasets: gqa, coco_captions, sharegpt4v.
    Supported model_type: qwen2_vl, qwen2_5_vl, qwen3_vl.
    """
    os.makedirs(data_dir, exist_ok=True)
    logger.info(
        f"Loading multimodal calibration data: {dataset} "
        f"({n_samples} samples, batch_size={batch_size}, "
        f"model_type={model_type}, data_dir={data_dir})"
    )

    if dataset == "gqa":
        ds = _build_gqa_dataset(n_samples, seed, data_dir)
    elif dataset == "coco_captions":
        ds = _build_coco_dataset(n_samples, seed, data_dir)
    elif dataset == "sharegpt4v":
        ds = _build_sharegpt4v_dataset(n_samples, seed, data_dir)
    else:
        raise ValueError(
            f"Unknown multimodal calibration dataset: '{dataset}'. "
            f"Choose from: gqa, coco_captions, sharegpt4v"
        )

    _SUPPORTED = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
    if model_type in _SUPPORTED:
        return _dataloader_to_batches(ds, processor, batch_size)
    else:
        raise ValueError(
            f"Multimodal calibration not yet implemented for model_type='{model_type}'. "
            f"Currently supported: {', '.join(sorted(_SUPPORTED))}"
        )
