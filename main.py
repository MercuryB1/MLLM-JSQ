"""CLI entry point for mllm-jsq compression."""
import argparse

from jsq.config import CompressConfig
from run import run, run_eval


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # yaml uses quantize_bmm_input; argparse uses no_quantize_bmm_input — normalise here
    if "quantize_bmm_input" in data:
        data["no_quantize_bmm_input"] = not data.pop("quantize_bmm_input")
    return data


def parse_args() -> CompressConfig:
    parser = argparse.ArgumentParser(
        description="mllm-jsq: Joint Sparsity & Quantization for LLMs and MLLMs"
    )

    # Config file (loaded first; CLI args override)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (CLI args take precedence)")

    # Model
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or local path (HuggingFace format)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the compressed model")
    parser.add_argument("--data_dir", type=str, default="storage/datasets",
                        help="Local directory to store/load calibration datasets")

    # Calibration
    parser.add_argument("--calib_dataset", type=str, default="pileval",
                        choices=["pileval", "c4", "wikitext2", "coco_captions", "sharegpt4v", "gqa"],
                        help="Calibration dataset (use coco_captions/sharegpt4v for MLLMs)")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--calib_batch_size", type=int, default=1,
                        help="Batch size for multimodal calibration (samples per forward pass)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for text calibration")
    parser.add_argument("--seed", type=int, default=42)

    # Pruning
    parser.add_argument("--pruning_method", type=str, default="jsq_v1",
                        choices=["jsq_v1", "jsq_v2", "wanda", "magnitude", "none"],
                        help="Pruning metric")
    parser.add_argument("--sparsity_ratio", type=float, default=0.0,
                        help="Target sparsity (0.0 = no pruning)")
    parser.add_argument("--sparsity_type", type=str, default="unstructured",
                        choices=["unstructured", "2:4", "4:8"])
    parser.add_argument("--rho", type=float, default=2.1,
                        help="JSQ sensitivity weight (rho in the paper)")

    # Quantization
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--weight_quant", type=str, default="per_channel",
                        choices=["per_channel", "per_tensor"])
    parser.add_argument("--act_quant", type=str, default="per_token",
                        choices=["per_token", "per_tensor"])
    parser.add_argument("--no_quantize_bmm_input", action="store_true",
                        help="Disable BMM input quantization for Q/K projections")

    # Smoothing
    parser.add_argument("--smooth_alpha", type=float, default=0.8)

    # Evaluation
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip compression; load a saved model from --save_dir and evaluate")
    parser.add_argument("--eval_ppl", action="store_true",
                        help="Evaluate WikiText-2 perplexity after compression")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated lmms-eval task names "
                             "(e.g. mmbench_en_dev,seedbench,mme)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit eval samples per task (-1 = no limit)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--no_compress", action="store_true",
                        help="Skip all compression passes (for quick validation)")

    # Two-pass parse: first grab --config, then inject yaml defaults, then re-parse
    pre, _ = parser.parse_known_args()
    if pre.config:
        parser.set_defaults(**_load_yaml(pre.config))

    args = parser.parse_args()

    if args.model is None:
        parser.error("--model is required (either via CLI or --config)")

    return args.eval_only, CompressConfig(
        model=args.model,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        calib_dataset=args.calib_dataset,
        nsamples=args.nsamples,
        calib_batch_size=args.calib_batch_size,
        seqlen=args.seqlen,
        seed=args.seed,
        pruning_method=args.pruning_method,
        sparsity_ratio=args.sparsity_ratio,
        sparsity_type=args.sparsity_type,
        rho=args.rho,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        weight_quant=args.weight_quant,
        act_quant=args.act_quant,
        quantize_bmm_input=not args.no_quantize_bmm_input,
        smooth_alpha=args.smooth_alpha,
        eval_ppl=args.eval_ppl,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        no_compress=args.no_compress,
    )


if __name__ == "__main__":
    eval_only, config = parse_args()
    if eval_only:
        run_eval(config)
    else:
        run(config)
