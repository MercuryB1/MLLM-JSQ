# AGENT.md — mllm-jsq 重构设计文档

本文档是对 `llm-jsq` 代码库的深度分析与重构蓝图。在动手编写代码前请完整阅读。

---

## 一、原始代码库的核心问题

通过阅读 `llm-jsq` 全部源码，识别出以下具体的架构缺陷：

### 1.1 硬编码的模型分支（最严重）

`smooth.py:164` — `smooth_layer()` 只处理 `LlamaDecoderLayer`，其他模型直接 `raise TypeError`：
```python
def smooth_layer(module, scales, alpha=0.5):
    if isinstance(module, LlamaDecoderLayer):
        ...
    else:
        raise TypeError(f"unsupported module: {module}")  # Qwen2 到这里直接崩
```

`fake_quant.py:237` — `quantize_layer()` 同样只认 `LlamaAttention` / `LlamaMLP`：
```python
def quantize_layer(module, ...):
    for name, m in module.named_modules():
        if isinstance(m, LlamaAttention):   # Qwen2Attention → 跳过
            ...
        elif isinstance(m, LlamaMLP):       # Qwen2MLP → 跳过
            ...
```
对 Qwen2 或 Qwen2-VL，`quantize_layer` 静默地什么都不做。

`utils.py:289` / `utils.py:322` — `move_embed()` 和 `get_blocks()` 是不断膨胀的 `elif` 链：
```python
def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        ...
    elif isinstance(model, OPTForCausalLM):
        ...
    # 每加一个模型就要改这里 + move_embed 里也要改一遍
    else:
        raise NotImplementedError(type(model))
```

`prune.py:32` — `check_sparsity()` 直接写死 `model.model.layers`，对 MLLM 一定出错。

### 1.2 遗留的调试代码

`prune.py:296` — 有一行 `import pdb; pdb.set_trace()` 在 `auto_prune_layer_wanda` 里，
会导致任何使用 WANDA 的压缩流程卡住等待终端输入。

### 1.3 关注点混杂

`main.py` 里混合了：模型加载、压缩调用、PPL 评估、MMLU 评估、多 GPU 分配，
全部在 `main()` 和 `evaluate()` 两个函数里。

`jsq.py` 的 `annealing_loop()` 内联了 `LlavaLlamaModel` 的特殊处理：
```python
if model.__class__.__name__ == "LlavaLlamaModel":
    model.llm(samples...)
else:
    model(samples...)
```
每支持一个新 MLLM 就要在这里加 if。

### 1.4 大量注释掉的死代码

`jsq.py` 底部有约 80 行注释代码；`main.py` 底部有约 50 行注释代码。
这些应当直接删除，不要保留。

### 1.5 量化覆盖不完整

`quantize_layer()` 依赖 `isinstance` 匹配具体子模块类型（`LlamaAttention`），
但 Qwen2 用的是 `Qwen2Attention`，Qwen2-VL 的 LLM 部分用 `Qwen2VLDecoderLayer`，
全部会被静默跳过，量化实际上不生效。

---

## 二、重构目标

**核心原则：**
- 新增模型 = 新增一个文件（Adapter），不改动任何核心压缩代码
- 压缩流程（prune / smooth / clip / quant）与模型结构完全解耦
- MLLM 支持：只压缩 LLM decoder，ViT 和 Projector 保持 FP16 不变
- 评测：用 `lmms-eval` 替代 `lm_eval`，同时保留文本 PPL 评测能力

---

## 三、目标目录结构

```
mllm-jsq/
├── main.py                          # CLI 入口：解析参数 → 构建 Config → 调用 run()
├── run.py                           # run(): 加载模型 → 压缩 → 评估
├── jsq/
│   ├── config.py                    # CompressConfig dataclass（类型安全）
│   │
│   ├── models/                      # 模型适配层（Adapter 模式）
│   │   ├── base.py                  # ModelAdapter ABC
│   │   ├── registry.py              # @register_adapter + get_adapter(model_type)
│   │   ├── llama.py                 # LlamaAdapter
│   │   ├── qwen2.py                 # Qwen2Adapter
│   │   └── qwen2_vl.py              # Qwen2VLAdapter（MLLM 首期目标）
│   │
│   ├── compression/
│   │   ├── pipeline.py              # CompressionPipeline：驱动逐层循环
│   │   ├── collector.py             # FeatureCollector：Hook 收集输入特征
│   │   └── passes/
│   │       ├── base.py              # CompressionPass ABC
│   │       ├── prune.py             # PruningPass（JSQ v1/v2、WANDA、Magnitude）
│   │       ├── smooth.py            # SmoothingPass
│   │       ├── clip.py              # ClippingPass
│   │       └── quantize.py          # QuantizationPass
│   │
│   ├── quant/
│   │   ├── linear.py                # QuantLinear（清理后的版本）
│   │   └── ops.py                   # 纯函数：quantize_weight_* / quantize_act_*
│   │
│   ├── calibration/
│   │   ├── text.py                  # 文本校准数据（C4、WikiText2、Pile）
│   │   └── multimodal.py            # 图文配对校准数据（COCO、ShareGPT4V）
│   │
│   └── eval/
│       ├── ppl.py                   # WikiText-2 PPL 评测
│       └── lmms_eval.py             # lmms-eval 集成
│
├── configs/
│   ├── llama2_7b_w8a8.yaml
│   └── qwen2_vl_7b_w8a8_s0.4375.yaml
│
└── scripts/
    ├── compress_llm.sh
    └── compress_qwen2vl.sh
```

---

## 四、核心接口设计

### 4.1 `ModelAdapter` ABC（`jsq/models/base.py`）

这是整个重构的枢纽。每个模型家族实现这个接口，核心压缩代码只依赖这个接口。

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import torch.nn as nn

class ModelAdapter(ABC):

    @abstractmethod
    def get_llm_blocks(self, model: nn.Module) -> List[nn.Module]:
        """返回需要压缩的 Transformer Block 列表（仅 LLM decoder）。
        对 MLLM，不包含 ViT 和 Projector。"""

    @abstractmethod
    def move_llm_embed(self, model: nn.Module, device) -> None:
        """将 LLM 侧的 embedding 层移动到指定 device。
        对 MLLM，不移动 ViT 相关模块。"""

    @abstractmethod
    def get_named_linears(self, block: nn.Module) -> Dict[str, nn.Linear]:
        """返回 block 内所有需要参与压缩的 Linear 层（name → module）。"""

    @abstractmethod
    def get_smooth_pairs(self, block: nn.Module) -> List[Tuple]:
        """返回该 block 的 (norm_layer, [linear, ...]) 对，用于激活平滑。
        每个 tuple 表示一组 LayerNorm → Linear 的平滑关系。

        示例（LLaMA）：
          [(input_layernorm, [q_proj, k_proj, v_proj]),
           (post_attention_layernorm, [gate_proj, up_proj])]
        """

    def run_forward_for_calibration(self, model: nn.Module, samples, **kwargs):
        """驱动模型前向以捕获 LLM 第一层输入。
        默认实现直接调用 model(samples)；
        MLLM 子类可覆盖此方法以正确处理图文混合输入。"""
        return model(samples, **kwargs)
```

### 4.2 模型注册表（`jsq/models/registry.py`）

```python
_REGISTRY: Dict[str, Type[ModelAdapter]] = {}

def register_adapter(*model_types: str):
    """装饰器，将 Adapter 类注册到 model_type 字符串。"""
    def decorator(cls):
        for t in model_types:
            _REGISTRY[t] = cls
        return cls
    return decorator

def get_adapter(model) -> ModelAdapter:
    """根据 model.config.model_type 查找 Adapter。"""
    model_type = model.config.model_type  # e.g. "qwen2_vl", "llama"
    if model_type not in _REGISTRY:
        raise NotImplementedError(f"No adapter registered for model_type='{model_type}'")
    return _REGISTRY[model_type]()
```

### 4.3 各模型 Adapter 实现

**`jsq/models/llama.py`**
```python
@register_adapter("llama")
class LlamaAdapter(ModelAdapter):
    def get_llm_blocks(self, model): return model.model.layers
    def move_llm_embed(self, model, device):
        model.model.embed_tokens.to(device)
        model.model.rotary_emb.to(device)
    def get_named_linears(self, block):
        return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    def get_smooth_pairs(self, block):
        return [
            (block.input_layernorm,
             [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]),
            (block.post_attention_layernorm,
             [block.mlp.gate_proj, block.mlp.up_proj]),
        ]
```

**`jsq/models/qwen2.py`**
```python
@register_adapter("qwen2")
class Qwen2Adapter(ModelAdapter):
    # Qwen2DecoderLayer 结构与 LLaMA 几乎相同，直接复用逻辑
    def get_llm_blocks(self, model): return model.model.layers
    def move_llm_embed(self, model, device):
        model.model.embed_tokens.to(device)
        model.model.rotary_emb.to(device)
    def get_named_linears(self, block):
        return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    def get_smooth_pairs(self, block):
        return [
            (block.input_layernorm,
             [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]),
            (block.post_attention_layernorm,
             [block.mlp.gate_proj, block.mlp.up_proj]),
        ]
```

**`jsq/models/qwen2_vl.py`**（关键：MLLM 适配）
```python
@register_adapter("qwen2_vl")
class Qwen2VLAdapter(ModelAdapter):
    """
    Qwen2-VL 模型结构：
      model.visual        ← ViT，保持 FP16，不压缩，不移动
      model.visual.merger ← Projector，保持 FP16，不压缩
      model.model.layers  ← LLM decoder，压缩目标
      model.model.embed_tokens
      model.model.rotary_emb (mrope)
    """
    def get_llm_blocks(self, model):
        # 只返回 LLM decoder blocks，不包含 ViT
        return model.model.layers

    def move_llm_embed(self, model, device):
        # 只移动 LLM 侧的 embedding，ViT 留在原设备
        model.model.embed_tokens.to(device)
        # Qwen2-VL 用 MRoPE，rotary_emb 挂在 model.model 下
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(device)

    def get_named_linears(self, block):
        return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}

    def get_smooth_pairs(self, block):
        # Qwen2-VL LLM decoder 与 Qwen2 结构相同
        return [
            (block.input_layernorm,
             [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]),
            (block.post_attention_layernorm,
             [block.mlp.gate_proj, block.mlp.up_proj]),
        ]

    def run_forward_for_calibration(self, model, samples, **kwargs):
        """
        MLLM 校准前向：
        如果 samples 是图文配对（含 pixel_values），正常执行全模型前向；
        ViT 的前向输出会被融入 input_embeds，再传入 LLM 第一层。
        Catcher hook 会拦截 LLM 第一层的输入。
        """
        if isinstance(samples, dict):
            return model(**samples)
        return model(samples, **kwargs)
```

### 4.4 `CompressionPass` ABC（`jsq/compression/passes/base.py`）

```python
from abc import ABC, abstractmethod
from typing import Dict
import torch

class CompressionPass(ABC):
    @abstractmethod
    def apply(
        self,
        block: torch.nn.Module,
        input_feat: Dict[str, torch.Tensor],  # name → 收集到的输入激活
        adapter,                               # ModelAdapter 实例
        config,                                # CompressConfig
    ) -> None:
        """原地修改 block 的权重，无返回值。"""
```

各 Pass 实现：
- `PruningPass.apply()` → 调用 JSQ/WANDA/Magnitude 度量函数，原地置零权重
- `SmoothingPass.apply()` → 调用 `adapter.get_smooth_pairs(block)`，执行 smooth_ln_fcs
- `ClippingPass.apply()` → 搜索最优 clip 阈值，原地 clamp 权重
- `QuantizationPass.apply()` → 遍历 `adapter.get_named_linears(block)`，替换为 `QuantLinear`

### 4.5 `CompressionPipeline`（`jsq/compression/pipeline.py`）

```python
class CompressionPipeline:
    def __init__(self, passes: List[CompressionPass], adapter: ModelAdapter):
        self.passes = passes
        self.adapter = adapter

    @torch.no_grad()
    def run(self, model, calib_samples, config):
        blocks = self.adapter.get_llm_blocks(model)

        # 用 Catcher 收集第一层输入
        inps, layer_kwargs = collect_first_layer_inputs(
            model, calib_samples, blocks,
            forward_fn=self.adapter.run_forward_for_calibration
        )

        self.adapter.move_llm_embed(model, "cpu")

        for i, block in enumerate(tqdm(blocks)):
            block.cuda()

            # 收集本层所有 Linear 的输入特征
            input_feat = collect_block_input_feat(block, inps, layer_kwargs)

            # 依次执行各 pass
            for pass_ in self.passes:
                pass_.apply(block, input_feat, self.adapter, config)

            # 更新下一层的输入
            inps = run_block(block, inps, layer_kwargs)
            block.cpu()
            torch.cuda.empty_cache()
```

### 4.6 `CompressConfig` dataclass（`jsq/config.py`）

```python
@dataclass
class CompressConfig:
    # 模型
    model: str

    # 校准
    calib_dataset: str = "pileval"         # pileval / c4 / wikitext2 / coco_captions
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 42

    # 剪枝
    pruning_method: str = "jsq_v1"        # jsq_v1 / jsq_v2 / wanda / magnitude / none
    sparsity_ratio: float = 0.0
    sparsity_type: str = "unstructured"   # unstructured / 2:4 / 4:8
    rho: float = 2.1

    # 量化
    w_bits: int = 8
    a_bits: int = 8
    weight_quant: str = "per_channel"     # per_channel / per_tensor
    act_quant: str = "per_token"          # per_token / per_tensor

    # 激活平滑
    smooth_alpha: float = 0.8

    # 评测
    eval_ppl: bool = False
    tasks: Optional[str] = None           # lmms-eval task 名，逗号分隔
    num_fewshot: int = 0

    # 其他
    save_dir: Optional[str] = None
    multigpu: bool = False
    batch_size: int = 1
```

---

## 五、lmms-eval 集成设计（`jsq/eval/lmms_eval.py`）

```python
# lmms-eval 要求 model class 实现 lmms_eval.api.model.lmms 接口
from lmms_eval.api.model import lmms

class CompressedMLLMWrapper(lmms):
    """将压缩后的 MLLM 包装为 lmms-eval 可用的模型。"""

    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config

    def generate_until(self, requests): ...
    def loglikelihood(self, requests): ...

def run_lmms_eval(model, processor, config: CompressConfig):
    """调用 lmms-eval 进行多模态基准评测。"""
    from lmms_eval import evaluator

    wrapped = CompressedMLLMWrapper(model, processor, config)
    task_names = [t.strip() for t in config.tasks.split(",")]

    results = evaluator.simple_evaluate(
        model=wrapped,
        tasks=task_names,
        num_fewshot=config.num_fewshot,
    )
    return results
```

支持的 lmms-eval tasks（优先适配）：
- `mmbench_en_dev` — MMBench English
- `seedbench` — SEED-Bench
- `mme` — MME
- `gqa` — GQA
- `textvqa_val` — TextVQA
- `vqav2_val` — VQAv2

---

## 六、实现顺序（Task List）

### Phase 1 — 基础设施（无模型依赖）
- [ ] `jsq/config.py` — CompressConfig dataclass
- [ ] `jsq/quant/ops.py` — 纯函数量化（从 fake_quant.py 提取，无模型类型依赖）
- [ ] `jsq/quant/linear.py` — QuantLinear（清理版，移除 `LlamaAttention` 依赖）
- [ ] `jsq/models/base.py` — ModelAdapter ABC
- [ ] `jsq/models/registry.py` — register_adapter + get_adapter
- [ ] `jsq/compression/passes/base.py` — CompressionPass ABC
- [ ] `jsq/compression/collector.py` — FeatureCollector（Catcher hook 抽象）

### Phase 2 — 压缩 Pass 实现
- [ ] `jsq/compression/passes/prune.py` — PruningPass（含 JSQ v1/v2、WANDA、Magnitude）
- [ ] `jsq/compression/passes/smooth.py` — SmoothingPass（通过 adapter.get_smooth_pairs）
- [ ] `jsq/compression/passes/clip.py` — ClippingPass
- [ ] `jsq/compression/passes/quantize.py` — QuantizationPass（通过 adapter.get_named_linears）
- [ ] `jsq/compression/pipeline.py` — CompressionPipeline

### Phase 3 — 模型 Adapter
- [ ] `jsq/models/llama.py` — LlamaAdapter（验证与原代码等价）
- [ ] `jsq/models/qwen2.py` — Qwen2Adapter
- [ ] `jsq/models/qwen2_vl.py` — Qwen2VLAdapter（MLLM 核心）

### Phase 4 — 校准数据
- [ ] `jsq/calibration/text.py` — Pileval / C4 / WikiText2
- [ ] `jsq/calibration/multimodal.py` — COCO Captions / ShareGPT4V 图文配对

### Phase 5 — 评测集成
- [ ] `jsq/eval/ppl.py` — WikiText-2 PPL
- [ ] `jsq/eval/lmms_eval.py` — CompressedMLLMWrapper + run_lmms_eval

### Phase 6 — 入口与脚本
- [ ] `run.py` — 组装所有组件的主函数
- [ ] `main.py` — CLI（argparse → CompressConfig → run()）
- [ ] `scripts/compress_qwen2vl.sh` — 端到端示例脚本

---

## 七、关键实现注意事项

### Qwen2-VL 模型路径
```
Qwen2VLForConditionalGeneration
  ├── model.visual              # ViT，不压缩
  │   └── merger                # Projector，不压缩
  ├── model.embed_tokens        # LLM embedding
  ├── model.layers[0..N]        # LLM decoder blocks，压缩目标
  ├── model.norm
  └── lm_head
```
`model.config.model_type == "qwen2_vl"`，可用于注册表查找。

### 多模态校准流程
```
1. 加载图文配对样本（pixel_values + input_ids）
2. 整个模型 forward（ViT FP16 → merger → embed 融合）
3. Catcher 拦截 model.model.layers[0] 的输入
   → 此时 inps 已包含融合了视觉特征的 hidden states
4. 此后的逐层压缩与纯 LLM 完全一致
```
关键：ViT 全程保持 FP16，不加任何 hook，不做任何修改。

### 平滑层的处理方式变化
原来：`smooth_layer()` 对具体类型做 `isinstance` 判断。
新架构：`SmoothingPass` 调用 `adapter.get_smooth_pairs(block)` 获取 `(norm, fcs)` 对，
然后调用统一的 `smooth_ln_fcs_rms()` 或 `smooth_ln_fcs()` 纯函数。
模型结构知识完全在 Adapter 里，Pass 本身不感知模型类型。

### 量化层替换的处理方式变化
原来：`quantize_layer()` 对具体 Attention/MLP 类做 `isinstance`。
新架构：`QuantizationPass` 调用 `adapter.get_named_linears(block)` 得到所有 Linear，
统一调用 `QuantLinear.from_float()` 替换，不需要知道外层是哪种 Attention。

---

## 八、快速验证命令（实现后）

```bash
# 1. 纯 LLM（Qwen2），验证与原代码结果一致
python main.py \
  --model Qwen/Qwen2-7B-Instruct \
  --pruning_method jsq_v1 \
  --sparsity_ratio 0.4375 \
  --w_bits 8 --a_bits 8 \
  --eval_ppl

# 2. MLLM（Qwen2-VL），文本校准 + PPL 验证
python main.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --pruning_method jsq_v1 \
  --sparsity_ratio 0.4375 \
  --w_bits 8 --a_bits 8 \
  --eval_ppl

# 3. MLLM（Qwen2-VL），图文校准 + lmms-eval 评测
python main.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --calib_dataset coco_captions \
  --pruning_method jsq_v1 \
  --sparsity_ratio 0.4375 \
  --w_bits 8 --a_bits 8 \
  --tasks mmbench_en_dev,seedbench,mme
```
