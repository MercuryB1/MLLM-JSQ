"""CLI entry point for mllm-jsq compression."""
import argparse

from jsq.config import CompressConfig
from run import run


def parse_args() -> CompressConfig:
    parser = argparse.ArgumentParser(
        description="mllm-jsq: Joint Sparsity & Quantization for LLMs and MLLMs"
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or local path (HuggingFace format)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the compressed model")

    # Calibration
    parser.add_argument("--calib_dataset", type=str, default="pileval",
                        choices=["pileval", "c4", "wikitext2", "coco_captions", "sharegpt4v"],
                        help="Calibration dataset (use coco_captions/sharegpt4v for MLLMs)")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
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
    parser.add_argument("--eval_ppl", action="store_true",
                        help="Evaluate WikiText-2 perplexity after compression")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated lmms-eval task names "
                             "(e.g. mmbench_en_dev,seedbench,mme)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit eval samples per task (-1 = no limit)")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    return CompressConfig(
        model=args.model,
        save_dir=args.save_dir,
        calib_dataset=args.calib_dataset,
        nsamples=args.nsamples,
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
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)
