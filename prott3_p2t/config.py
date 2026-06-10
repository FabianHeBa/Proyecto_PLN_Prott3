import os
from dataclasses import dataclass

import torch


@dataclass
class Config:

    # Paths
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = "./esm_cache"
    save_dir: str = "./prott3_gemma_qa"

    # Model IDs
    esm_id: str = "facebook/esm2_t30_150M_UR50D"
    qformer_id: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    gemma_id: str = "google/gemma-2-2b-it"

    # Data columns
    seq_col: str = "amino_seq"
    question_col: str = "question"
    answer_col: str = "answer"

    # Data split/sample settings
    dataset_id: str = "tumorailab/Protein2Text-QA"
    source_split: str = "test"
    sample_fraction: float = 0.30
    validation_size: float = 0.05
    seed: int = 42

    # Sequence/text limits
    n_query_tokens: int = 32
    max_prot_len: int = 1024
    max_text_len: int = 768

    # Dataloaders
    train_batch_size: int = 4
    val_batch_size: int = 2
    esm_batch_size: int = 4

    # Training
    epochs: int = 5
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    qformer_lr: float = 1e-4
    lora_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Generation/evaluation
    max_new_tokens: int = 128
    num_examples_to_print: int = 10

    def ensure_dirs(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
