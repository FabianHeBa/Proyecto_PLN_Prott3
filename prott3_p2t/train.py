import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .config import Config
from .model import ProtT3GemmaQA


def build_model_from_loader(train_loader, config: Config) -> ProtT3GemmaQA:
    sample_batch = next(iter(train_loader))
    protein_dim = sample_batch["protein_emb"].shape[-1]

    model = ProtT3GemmaQA(protein_dim=protein_dim, config=config)
    model.protein_qformer.to(model.llm_device())
    model.llm.print_trainable_parameters()
    return model


def build_optimizer_and_scheduler(model, train_loader, config: Config):
    qformer_params = []
    lora_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "protein_qformer" in name:
            qformer_params.append(param)
        else:
            lora_params.append(param)

    optimizer = AdamW(
        [
            {"params": qformer_params, "lr": config.qformer_lr},
            {"params": lora_params, "lr": config.lora_lr},
        ],
        weight_decay=config.weight_decay,
    )

    num_training_steps = config.epochs * len(train_loader) // config.grad_accum
    warmup_steps = int(config.warmup_ratio * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def train_model(model, train_loader, optimizer, scheduler, config: Config):
    model.train()
    global_step = 0

    for epoch in range(config.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                out = model(
                    protein_emb=batch["protein_emb"],
                    protein_mask=batch["protein_mask"],
                    questions=batch["questions"],
                    answers=batch["answers"],
                )

                loss = out.loss / config.grad_accum

            loss.backward()
            running_loss += loss.item() * config.grad_accum

            if (step + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

            pbar.set_postfix(
                {
                    "loss": running_loss / (step + 1),
                    "lr_lora": scheduler.get_last_lr()[-1],
                }
            )

    return model


def save_model(model, config: Config):
    os.makedirs(config.save_dir, exist_ok=True)

    model.llm.save_pretrained(os.path.join(config.save_dir, "gemma_lora"))
    model.tokenizer.save_pretrained(os.path.join(config.save_dir, "tokenizer"))

    torch.save(
        model.protein_qformer.state_dict(),
        os.path.join(config.save_dir, "protein_qformer.pt"),
    )
