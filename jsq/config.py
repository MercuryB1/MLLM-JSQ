from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressConfig:
    # Model
    model: str = ""

    # Calibration
    calib_dataset: str = "pileval"   # pileval / c4 / wikitext2 / coco_captions
    nsamples: int = 128
    calib_batch_size: int = 1        # batch size for multimodal calibration
    seqlen: int = 2048
    seed: int = 42

    # Pruning
    pruning_method: str = "jsq_v1"  # jsq_v1 / jsq_v2 / wanda / magnitude / none
    sparsity_ratio: float = 0.0
    sparsity_type: str = "unstructured"  # unstructured / 2:4 / 4:8
    rho: float = 2.1

    # Quantization
    w_bits: int = 8
    a_bits: int = 8
    weight_quant: str = "per_channel"   # per_channel / per_tensor
    act_quant: str = "per_token"        # per_token / per_tensor
    quantize_bmm_input: bool = True

    # Smoothing
    smooth_alpha: float = 0.8

    # Evaluation
    eval_ppl: bool = False
    tasks: Optional[str] = None         # lmms-eval task names, comma-separated
    num_fewshot: int = 0
    limit: int = -1

    # Storage
    data_dir: str = "storage/datasets"   # local directory for calibration datasets

    # MA-JSQ block search
    gamma: float = 1.0              # modal balance factor (vision vs text error weight)
    n_search_candidates: int = 8    # number of per-layer sparsity configs to evaluate

    # Other
    save_dir: Optional[str] = None
    multigpu: bool = False
    batch_size: int = 1
    no_compress: bool = False   # skip all compression passes (for quick validation)

    @property
    def prune_n(self) -> int:
        if self.sparsity_type == "unstructured" or self.sparsity_ratio == 0.0:
            return 0
        return int(self.sparsity_type.split(":")[0])

    @property
    def prune_m(self) -> int:
        if self.sparsity_type == "unstructured" or self.sparsity_ratio == 0.0:
            return 0
        return int(self.sparsity_type.split(":")[1])
