"""WikiText-2 perplexity evaluation."""
import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


@torch.no_grad()
def eval_ppl(model: nn.Module, tokenizer, seq_len: int = 2048) -> float:
    """Compute perplexity on the WikiText-2 test set.

    Args:
        model: The (compressed) language model.
        tokenizer: Corresponding tokenizer.
        seq_len: Sliding window length.

    Returns:
        Perplexity as a float.
    """
    logger.info("Evaluating WikiText-2 perplexity...")
    model.eval()

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids

    device = next(model.parameters()).device
    n_samples = input_ids.shape[1] // seq_len

    nlls = []
    loss_fn = nn.CrossEntropyLoss()

    for i in tqdm(range(n_samples), desc="PPL eval"):
        batch = input_ids[:, i * seq_len:(i + 1) * seq_len].to(device)
        logits = model(batch).logits  # [1, seq_len, vocab]
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seq_len)

    ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * seq_len)).item()
    logger.info(f"WikiText-2 PPL: {ppl:.4f}")
    return ppl
