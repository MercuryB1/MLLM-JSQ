"""Text-only calibration dataset loaders."""
import random
from typing import List

import torch
from datasets import load_dataset
from loguru import logger


def _get_pileval(tokenizer, n_samples: int, seq_len: int, seed: int) -> torch.Tensor:
    dataset = load_dataset(
        "mit-han-lab/pile-val-backup", split="validation"
    )
    random.seed(seed)
    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            enc = tokenizer(dataset[idx]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seq_len:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seq_len)
        samples.append(enc.input_ids[:, i:i + seq_len])
    return torch.cat(samples, dim=0)  # [n_samples, seq_len]


def _get_c4(tokenizer, n_samples: int, seq_len: int, seed: int) -> torch.Tensor:
    dataset = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    random.seed(seed)
    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            enc = tokenizer(dataset[idx]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seq_len:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seq_len)
        samples.append(enc.input_ids[:, i:i + seq_len])
    return torch.cat(samples, dim=0)


def _get_wikitext2(tokenizer, n_samples: int, seq_len: int, seed: int) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    random.seed(seed)
    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        i = random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
        samples.append(enc.input_ids[:, i:i + seq_len])
    return torch.cat(samples, dim=0)


def get_text_calib_data(
    dataset: str,
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """Return a [n_samples, seq_len] int64 tensor of tokenized calibration text.

    Supported datasets: pileval, c4, wikitext2.
    """
    logger.info(f"Loading text calibration data: {dataset} ({n_samples} samples, seq_len={seq_len})")
    if dataset == "pileval":
        return _get_pileval(tokenizer, n_samples, seq_len, seed)
    elif dataset == "c4":
        return _get_c4(tokenizer, n_samples, seq_len, seed)
    elif dataset == "wikitext2":
        return _get_wikitext2(tokenizer, n_samples, seq_len, seed)
    else:
        raise ValueError(f"Unknown text calibration dataset: '{dataset}'. "
                         f"Choose from: pileval, c4, wikitext2")
