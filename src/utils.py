import os
import random
from typing import List, Tuple

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_text_loader(path: str) -> List[str]:
    """Load .txt/.md files from directory."""
    files = []
    if os.path.isdir(path):
        for root, _, fnames in os.walk(path):
            for f in fnames:
                if f.lower().endswith((".txt", ".md")):
                    files.append(os.path.join(root, f))
    else:
        files = [path]

    texts = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                texts.append(fh.read())
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    return texts


def split_by_tokens(text: str, tokenizer, max_tokens: int, stride: int) -> List[str]:
    """Split text into overlapping chunks by token count."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    i = 0
    while i < len(ids):
        window = ids[i:i+max_tokens]
        if not window:
            break
        out.append(tokenizer.decode(window))
        if i + max_tokens >= len(ids):
            break
        i += stride
    return out


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def top2_margin_and_gap(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    top2p, top2i = p.topk(2, dim=-1)
    gap = top2p[:,0] - top2p[:,1]
    logp_top2 = torch.gather(logp, 1, top2i)
    margin = logp_top2[:,0] - logp_top2[:,1]
    return margin, gap