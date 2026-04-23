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
    pruning_method: str = "jsq_v1"  # jsq_v1..v5 / wanda / magnitude / none
    sparsity_ratio: float = 0.0
    sparsity_type: str = "unstructured"  # unstructured / 2:4 / 4:8
    rho: float = 2.1
    alpha: float = 0.5      # JSQ v3: token density weighting (0 = standard WANDA scale)
    beta: float = 0.5       # JSQ v3: quantization damage penalty (0 = no quant-awareness)
    top_k: int = 3          # JSQ v3: top-k outlier positions to penalize per row
    # JSQ v5: mixture-Hessian OBS metric
    pi_t: float = 0.5           # text mixture weight; pi_v = 1 - pi_t
    lambda_floor: float = 1e-3  # minimum regularizer in H = pi_t H_t + pi_v H_v + lam I
    # JSQ v5 block-wise adaptive mixture (Innovation 2 experimental branch)
    block_pi_method: str = "none"    # none / dominance / conflict_weighted
    block_pi_blend: float = 1.0      # 0 = keep global pi_t, 1 = use pure block estimate
    block_pi_min: float = 0.1        # clip range for per-block pi_t
    block_pi_max: float = 0.9
    block_pi_max_tokens: int = 1024  # cheaper token cap for block-level pi_t estimation

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
    search_method: str = "none"     # none / candidate / owl / greedy_sequential
    gamma: float = 1.0              # modal balance factor (vision vs text error weight)
    n_search_candidates: int = 16   # number of per-layer sparsity configs to evaluate

    # JSQ v5 Option E: per-block multimodal sparsity allocation
    block_alloc_method: str = "uniform"    # uniform / bi_mixture / bi_text / bi_vision / damage
    block_alloc_alpha: float = 1.0         # inverse-sensitivity exponent
    block_alloc_s_min: float = 0.1         # per-block sparsity clip (low)
    block_alloc_s_max: float = 0.7         # per-block sparsity clip (high)
    block_alloc_invert: bool = False       # True = high BI → high sparsity
    block_alloc_log: bool = False          # True = log-transform scores before allocation
    block_alloc_seq: bool = False          # True = sequential damage (propagate pruned signal)
    # JSQ v5 Zero-Bit allocation: solve per-layer sparsity inside each block
    # from the same mixture-Hessian distortion. "zero_bit_d0" uses pure pruning
    # loss; "zero_bit_joint" models keep-as-W8 vs prune-as-0bit utility.
    layer_alloc_method: str = "uniform"    # uniform / zero_bit_d0 / zero_bit_joint

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
