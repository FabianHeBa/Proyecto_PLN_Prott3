import hashlib
import os
from typing import Iterable, Optional

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import Config


def seq_hash(seq: str):
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def cache_path(seq: str, cache_dir: str):
    return os.path.join(cache_dir, f"{seq_hash(seq)}.pt")


def load_esm(config: Config):
    esm_tokenizer = AutoTokenizer.from_pretrained(config.esm_id)
    esm_model = AutoModel.from_pretrained(config.esm_id).to(config.device)
    esm_model.eval()

    for param in esm_model.parameters():
        param.requires_grad = False

    return esm_tokenizer, esm_model


@torch.no_grad()
def precompute_esm_cache(
    df,
    config: Config,
    esm_tokenizer=None,
    esm_model=None,
    seq_col: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    config.ensure_dirs()

    if esm_tokenizer is None or esm_model is None:
        esm_tokenizer, esm_model = load_esm(config)

    seq_col = seq_col or config.seq_col
    batch_size = batch_size or config.esm_batch_size
    pending_seqs = []

    def flush(batch_seqs: Iterable[str]):
        batch_seqs = list(batch_seqs)
        if not batch_seqs:
            return

        toks = esm_tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prot_len,
        ).to(config.device)

        out = esm_model(**toks).last_hidden_state
        lengths = toks["attention_mask"].sum(dim=1).cpu().tolist()

        for i, seq in enumerate(batch_seqs):
            emb = out[i, : lengths[i]].detach().cpu().to(torch.float16)
            torch.save(emb, cache_path(seq, config.cache_dir))

    seqs = (
        df[seq_col]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    for seq in tqdm(seqs, desc="Precomputando ESM cache"):
        if not os.path.exists(cache_path(seq, config.cache_dir)):
            pending_seqs.append(seq)

        if len(pending_seqs) >= batch_size:
            flush(pending_seqs)
            pending_seqs = []

    flush(pending_seqs)
