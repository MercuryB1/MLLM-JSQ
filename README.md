# mllm-jsq

Joint Sparsity & Quantization compression framework for LLMs and MLLMs (Multimodal Large Language Models).

Supports **JSQ / WANDA pruning** combined with **W8A8 quantization** in a unified pipeline, with clean Adapter-based extensibility for new model families.

---

## Features

- **Pruning methods**: JSQ v1, JSQ v2, WANDA, Magnitude, or None
- **Sparsity patterns**: Unstructured, 2:4, 4:8
- **Quantization**: W8A8 (per-channel weight, per-token activation), with activation smoothing and clipping
- **MLLM support**: Only compresses the LLM decoder; ViT and Projector remain in FP16 untouched
- **Calibration**: Text datasets (Pile, C4, WikiText-2) and multimodal datasets (COCO Captions, ShareGPT4V)
- **Evaluation**: WikiText-2 perplexity (PPL) and lmms-eval multimodal benchmarks
- **Extensible**: Adding a new model = writing one Adapter file, no changes to compression core

---

## Supported Models

| Model Family | model_type | Notes |
|---|---|---|
| LLaMA / LLaMA-2 | `llama` | Text-only LLM |
| Qwen2 | `qwen2` | Text-only LLM |
| Qwen2-VL | `qwen2_vl` | MLLM, compresses LLM decoder only |
| Qwen2.5-VL | `qwen2_5_vl` | MLLM, compresses LLM decoder only |
| Qwen3-VL | `qwen3_vl` | MLLM, compresses LLM decoder only |

---

## Environment Setup

### 1. Create a conda environment with Python 3.10

```bash
conda create -n mllm-jsq python=3.10 -y
conda activate mllm-jsq
```

### 2. Install uv

```bash
pip install uv
```

### 3. Install dependencies (including PyTorch CUDA)

```bash
uv sync
```

This installs PyTorch from the PyTorch CUDA 12.1 index automatically (configured in `pyproject.toml`). To use a different CUDA version, update the index URL in `[tool.uv.index]` before syncing.

### 4. Install lmms-eval for multimodal evaluation

Required when using `--tasks` (e.g. `mmbench_en_dev`, `seedbench`, `mme`).
lmms-eval is managed as a local package under `third_party/`:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git third_party/lmms-eval
uv pip install -e third_party/lmms-eval
```

### 5. Verify the setup

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python main.py --help
```

---

## Quick Start

### Text-only LLM (Qwen2 / LLaMA)

```bash
python main.py \
  --model Qwen/Qwen2-7B-Instruct \
  --pruning_method jsq_v1 \
  --sparsity_ratio 0.4375 \
  --w_bits 8 --a_bits 8 \
  --eval_ppl \
  --save_dir ./outputs/qwen2-7b-compressed
```

### Multimodal LLM (Qwen2.5-VL / Qwen3-VL) with multimodal calibration

```bash
python main.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --calib_dataset gqa \
  --pruning_method jsq_v1 \
  --sparsity_ratio 0.4375 \
  --w_bits 8 --a_bits 8 \
  --tasks mmbench_en_dev,seedbench,mme \
  --save_dir ./outputs/qwen2.5vl-7b-compressed
```

### Using provided scripts

```bash
# Text-only LLM
bash scripts/compress_llm.sh Qwen/Qwen2-7B-Instruct ./outputs/qwen2-7b

# Qwen2.5-VL（默认），压缩后自动评测
bash scripts/compress_qwen2vl.sh

# 指定模型和评测任务
bash scripts/compress_qwen2vl.sh Qwen/Qwen2.5-VL-7B-Instruct ./outputs/qwen2.5vl-7b mme,mmbench_en_dev
bash scripts/compress_qwen2vl.sh Qwen/Qwen3-VL-7B-Instruct   ./outputs/qwen3vl-7b   mme,mmbench_en_dev
```

---

## CLI Arguments

### Model

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model name or local path |
| `--save_dir` | None | Directory to save compressed model |

### Calibration

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--calib_dataset` | `pileval` | `pileval`, `c4`, `wikitext2`, `coco_captions`, `sharegpt4v` | Use `coco_captions`/`sharegpt4v` for MLLMs |
| `--nsamples` | `128` | | Number of calibration samples |
| `--seqlen` | `2048` | | Sequence length for text calibration |
| `--seed` | `42` | | Random seed |

### Pruning

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--pruning_method` | `jsq_v1` | `jsq_v1`, `jsq_v2`, `wanda`, `magnitude`, `none` | Pruning metric |
| `--sparsity_ratio` | `0.0` | | Target sparsity (0.0 = no pruning) |
| `--sparsity_type` | `unstructured` | `unstructured`, `2:4`, `4:8` | Sparsity pattern |
| `--rho` | `2.1` | | JSQ sensitivity weight |

### Quantization

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--w_bits` | `8` | | Weight quantization bits |
| `--a_bits` | `8` | | Activation quantization bits |
| `--weight_quant` | `per_channel` | `per_channel`, `per_tensor` | Weight quantization granularity |
| `--act_quant` | `per_token` | `per_token`, `per_tensor` | Activation quantization granularity |
| `--smooth_alpha` | `0.8` | | Activation smoothing factor |
| `--no_quantize_bmm_input` | | | Disable BMM input quantization for Q/K projections |

### Evaluation

| Argument | Default | Description |
|---|---|---|
| `--eval_ppl` | | Evaluate WikiText-2 perplexity after compression |
| `--tasks` | None | Comma-separated lmms-eval task names (e.g. `mmbench_en_dev,seedbench,mme`) |
| `--num_fewshot` | `0` | Few-shot examples for evaluation |
| `--limit` | `-1` | Max evaluation samples per task (-1 = no limit) |
| `--batch_size` | `1` | Evaluation batch size |

---

## Supported lmms-eval Tasks

| Task name | Benchmark |
|---|---|
| `mmbench_en_dev` | MMBench (English) |
| `seedbench` | SEED-Bench |
| `mme` | MME |
| `gqa` | GQA |
| `textvqa_val` | TextVQA |
| `vqav2_val` | VQAv2 |

---

## Project Structure

```
mllm-jsq/
├── main.py                        # CLI entry: parse args → CompressConfig → run()
├── run.py                         # run(): load model → compress → evaluate → save
├── jsq/
│   ├── config.py                  # CompressConfig dataclass
│   ├── models/                    # Model Adapter layer
│   │   ├── base.py                # ModelAdapter ABC
│   │   ├── registry.py            # @register_adapter + get_adapter()
│   │   ├── llama.py               # LlamaAdapter
│   │   ├── qwen2.py               # Qwen2Adapter
│   │   ├── qwen2_vl.py            # Qwen2VLAdapter (MLLM)
│   │   ├── qwen2_5_vl.py          # Qwen2_5_VLAdapter (MLLM)
│   │   └── qwen3_vl.py            # Qwen3VLAdapter (MLLM)
│   ├── compression/
│   │   ├── pipeline.py            # CompressionPipeline: drives per-layer loop
│   │   ├── collector.py           # Feature collection via forward hooks
│   │   └── passes/
│   │       ├── base.py            # CompressionPass ABC
│   │       ├── prune.py           # PruningPass (JSQ v1/v2, WANDA, Magnitude)
│   │       ├── smooth.py          # SmoothingPass (activation smoothing)
│   │       ├── clip.py            # ClippingPass (weight clipping)
│   │       └── quantize.py        # QuantizationPass (replace Linear → QuantLinear)
│   ├── quant/
│   │   ├── ops.py                 # Pure functions: quantize_weight_* / quantize_act_*
│   │   └── linear.py              # QuantLinear module
│   ├── calibration/
│   │   ├── text.py                # Text calibration data (C4, WikiText-2, Pile)
│   │   └── multimodal.py          # Multimodal calibration data (COCO, ShareGPT4V)
│   └── eval/
│       ├── ppl.py                 # WikiText-2 PPL evaluation
│       └── lmms_eval.py           # lmms-eval integration
├── configs/
│   ├── llama2_7b_w8a8.yaml
│   └── qwen2_vl_7b_w8a8_s0.4375.yaml
└── scripts/
    ├── compress_llm.sh
    └── compress_qwen2vl.sh
```

---

## Compression Pipeline

The pipeline runs once per Transformer decoder block:

```
Calibration data
      │
      ▼
Catcher hook → collect LLM layer[0] inputs
      │
      ▼  (for each block i)
┌─────────────────────────────────┐
│ 1. PruningPass   (JSQ / WANDA)  │
│ 2. SmoothingPass (act smooth)   │
│ 3. ClippingPass  (weight clip)  │
│ 4. QuantizationPass (W8A8)      │
└─────────────────────────────────┘
      │
      ▼
  block.cpu() → next block inputs
```

For MLLMs, the ViT runs in FP16 during calibration to produce vision-fused hidden states; only the LLM decoder blocks are compressed.

---

## Adding a New Model

Create `jsq/models/your_model.py`:

```python
from jsq.models.base import ModelAdapter
from jsq.models.registry import register_adapter

@register_adapter("your_model_type")   # matches model.config.model_type
class YourModelAdapter(ModelAdapter):

    def get_llm_blocks(self, model):
        return model.model.layers

    def move_llm_embed(self, model, device):
        model.model.embed_tokens.to(device)

    def get_named_linears(self, block):
        return {n: m for n, m in block.named_modules()
                if isinstance(m, nn.Linear)}

    def get_smooth_pairs(self, block):
        return [
            (block.input_layernorm,
             [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]),
            (block.post_attention_layernorm,
             [block.mlp.gate_proj, block.mlp.up_proj]),
        ]
```

Then import it in `run.py`:

```python
import jsq.models.your_model  # noqa: F401
```

No changes needed to any compression code.
