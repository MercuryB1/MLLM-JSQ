"""CLI entry point for mllm-jsq compression."""
import argparse
import os
from datetime import datetime
from pathlib import Path

from loguru import logger

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


def _setup_logging(log_dir: str | None, save_dir: str | None, eval_only: bool) -> None:
    # 显式配置日志目录时优先使用；否则在 save_dir 下创建 logs 子目录
    target_dir = log_dir
    if target_dir is None and save_dir:
        target_dir = str(Path(save_dir) / "logs")

    if not target_dir:
        return

    log_path = Path(target_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    mode = "eval" if eval_only else "compress"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{mode}_{stamp}.log"
    file_path = log_path / file_name

    logger.add(
        str(file_path),
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    # 提供给下游模块：若第三方库重置了 loguru，可据此恢复文件 sink
    os.environ["JSQ_LOG_FILE"] = str(file_path)
    os.environ["JSQ_LOG_LEVEL"] = "INFO"
    logger.info(f"日志文件路径: {file_path}")


def parse_args() -> CompressConfig:
    parser = argparse.ArgumentParser(
        description="mllm-jsq: Joint Sparsity & Quantization for LLMs and MLLMs"
    )

    # Config file (loaded first; CLI args override)
    # 配置文件路径：先加载 YAML 默认值，再由命令行参数覆盖
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (CLI args take precedence)")

    # Model
    # 模型名称或本地路径（HuggingFace 兼容格式）
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or local path (HuggingFace format)")
    # 压缩后模型保存目录；评测模式下也可用于加载已保存模型
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the compressed model")
    # 数据集本地缓存目录（校准数据会下载/读取到这里）
    parser.add_argument("--data_dir", type=str, default="storage/datasets",
                        help="Local directory to store/load calibration datasets")
    # 日志目录：若指定则写入滚动日志文件
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to store runtime log files")

    # Calibration
    # 校准数据集名称：文本模型常用 pileval/c4/wikitext2，多模态常用 coco/sharegpt4v/gqa
    parser.add_argument("--calib_dataset", type=str, default="pileval",
                        choices=["pileval", "c4", "wikitext2", "coco_captions", "sharegpt4v", "gqa"],
                        help="Calibration dataset (use coco_captions/sharegpt4v for MLLMs)")
    # 校准样本数：越大通常越稳，但耗时与显存占用也更高
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    # 多模态校准批大小：每次前向处理的图文样本数量
    parser.add_argument("--calib_batch_size", type=int, default=1,
                        help="Batch size for multimodal calibration (samples per forward pass)")
    # 文本校准序列长度
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for text calibration")
    # 随机种子：控制采样与实验可复现性
    parser.add_argument("--seed", type=int, default=42)

    # Pruning
    # 剪枝方法：jsq_v1/jsq_v2/wanda/magnitude/none
    parser.add_argument("--pruning_method", type=str, default="jsq_v1",
                        choices=["jsq_v1", "jsq_v2", "wanda", "magnitude", "none"],
                        help="Pruning metric")
    # 目标稀疏率：0.0 表示不剪枝
    parser.add_argument("--sparsity_ratio", type=float, default=0.0,
                        help="Target sparsity (0.0 = no pruning)")
    # 稀疏模式：非结构化或 N:M 结构化（如 2:4）
    parser.add_argument("--sparsity_type", type=str, default="unstructured",
                        choices=["unstructured", "2:4", "4:8"])
    # JSQ 灵敏度权重 rho（仅 JSQ 方法会使用）
    parser.add_argument("--rho", type=float, default=2.1,
                        help="JSQ sensitivity weight (rho in the paper)")

    # Quantization
    # 权重量化位宽（常见 W8）
    parser.add_argument("--w_bits", type=int, default=8)
    # 激活量化位宽（常见 A8）
    parser.add_argument("--a_bits", type=int, default=8)
    # 权重量化粒度：按通道或按张量
    parser.add_argument("--weight_quant", type=str, default="per_channel",
                        choices=["per_channel", "per_tensor"])
    # 激活量化粒度：按 token 或按张量
    parser.add_argument("--act_quant", type=str, default="per_token",
                        choices=["per_token", "per_tensor"])
    # 是否关闭 Q/K 投影对应的 BMM 输入量化（传入该参数即关闭）
    parser.add_argument("--no_quantize_bmm_input", action="store_true",
                        help="Disable BMM input quantization for Q/K projections")

    # Smoothing
    # 平滑系数 alpha：用于激活平滑阶段
    parser.add_argument("--smooth_alpha", type=float, default=0.8)

    # Evaluation
    # 仅评测：跳过压缩，直接从 --save_dir 加载模型评测
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip compression; load a saved model from --save_dir and evaluate")
    # 是否计算 WikiText-2 困惑度（PPL）
    parser.add_argument("--eval_ppl", action="store_true",
                        help="Evaluate WikiText-2 perplexity after compression")
    # lmms-eval 任务列表（逗号分隔）
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated lmms-eval task names "
                             "(e.g. mmbench_en_dev,seedbench,mme)")
    # few-shot 示例数量
    parser.add_argument("--num_fewshot", type=int, default=0)
    # 每个任务的样本上限，-1 表示不限制
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit eval samples per task (-1 = no limit)")
    # 评测批大小
    parser.add_argument("--batch_size", type=int, default=1)
    # 跳过所有压缩 Pass（快速联调/流程验证）
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
        log_dir=args.log_dir,
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
    _setup_logging(config.log_dir, config.save_dir, eval_only)
    if eval_only:
        run_eval(config)
    else:
        run(config)
