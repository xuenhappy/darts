#!/usr/bin/env python3
"""LoRA-tune a small Qwen model to route LTP/Paddle/reject candidates."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model

class RouterDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, drop_overlength=False, max_items=0):
        self.items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        if drop_overlength:
            self.items = [item for item in self.items if self.encoded_length(item) <= max_length]
        if max_items:
            self.items = self.items[:max_items]

    def encoded_length(self, item):
        prompt = self.tokenizer.apply_chat_template(
            [item["messages"][0]], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        text = prompt + item["messages"][1]["content"] + self.tokenizer.eos_token
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        user = item["messages"][0]
        answer = item["messages"][1]["content"] + self.tokenizer.eos_token
        prompt = self.tokenizer.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        input_ids = (prompt_ids + answer_ids)[-self.max_length:]
        answer_length = min(len(answer_ids), len(input_ids))
        labels = [-100] * (len(input_ids) - answer_length) + input_ids[-answer_length:]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_token_id):
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        padding = width - len(item["input_ids"])
        # Left padding keeps every supervised answer at the batch suffix. This
        # lets Qwen compute vocabulary logits only for answer-bearing positions.
        input_ids.append([pad_token_id] * padding + item["input_ids"])
        labels.append([-100] * padding + item["labels"])
        attention.append([0] * padding + [1] * len(item["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


@torch.no_grad()
def stable_clip_grad_norm_(parameters, max_norm):
    """Clip gradients without overflowing FP32's squared-norm reduction."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("cannot clip non-finite gradients")
    max_abs = max(float(gradient.detach().abs().max()) for gradient in gradients)
    if max_abs == 0.0:
        return 0.0
    scaled_square_sum = sum(
        float(((gradient.detach().double() / max_abs) ** 2).sum())
        for gradient in gradients
    )
    total_norm = max_abs * scaled_square_sum ** 0.5
    clip_coefficient = min(1.0, float(max_norm) / (total_norm + 1e-12))
    if clip_coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)
    return total_norm


def supervised_causal_lm_loss(model, batch):
    """Compute LM logits only where they can predict supervised answer tokens."""
    labels = batch["labels"]
    answer_length = int((labels != -100).sum(dim=1).max().item())
    logits_to_keep = min(labels.shape[1], answer_length + 1)
    shifted_labels = F.pad(labels, (0, 1), value=-100)[:, 1:][:, -logits_to_keep:]
    return model(
        **batch, logits_to_keep=logits_to_keep, shift_labels=shifted_labels
    ).loss


def evaluate_loss(model, loader, device):
    checkpointing = model.is_gradient_checkpointing
    if checkpointing:
        model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()
    model.eval()
    total = count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = supervised_causal_lm_loss(model, batch)
            total += float(loss)
            count += 1
    model.train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/darts-corpus/models/Qwen3-1.7B")
    parser.add_argument("--train", default="/data/darts-corpus/router/router-train.jsonl")
    parser.add_argument("--dev", default="/data/darts-corpus/router/router-dev.jsonl")
    parser.add_argument("--output", default="/data/darts-corpus/models/router-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--log-every-updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--resume-adapter")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-dev", action="store_true",
                        help="save each epoch and evaluate later in a fresh CUDA process")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA")
    # RTX A2000 has shown illegal-instruction failures in asynchronous fused SDPA.
    # Math SDPA is slower but keeps the complete attention gradient path.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = RouterDataset(
        args.train, tokenizer, args.max_length, drop_overlength=True,
        max_items=args.max_train_samples,
    )
    dev_data = None if args.skip_dev else RouterDataset(args.dev, tokenizer, args.max_length)
    counts = Counter("changed" if item.get("changed") else "identity" for item in train_data.items)
    collator = lambda batch: collate(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=0, pin_memory=False)
    dev_loader = None if dev_data is None else DataLoader(
        dev_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=False,
    )
    model_dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, local_files_only=True
    ).to(device)
    base.config.use_cache = False
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    if args.resume_adapter:
        model = PeftModel.from_pretrained(base, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=model_dtype == torch.float16)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, updates_per_epoch // 10), updates_per_epoch * args.epochs
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    skipped_loss_batches = 0
    skipped_gradient_updates = 0
    completed_updates = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            loss = supervised_causal_lm_loss(model, batch) / args.gradient_accumulation
            if not torch.isfinite(loss):
                skipped_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += float(loss.detach()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                try:
                    gradient_norm = stable_clip_grad_norm_(model.parameters(), 1.0)
                except RuntimeError:
                    gradient_norm = None
                if gradient_norm is not None:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    completed_updates += 1
                    if args.log_every_updates and completed_updates % args.log_every_updates == 0:
                        print(json.dumps({
                            "epoch": epoch, "completed_updates": completed_updates,
                            "mean_loss": running / step,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }), flush=True)
                else:
                    skipped_gradient_updates += 1
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        epoch_dir = output / f"epoch-{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        dev_loss = None if dev_loader is None else evaluate_loss(model, dev_loader, device)
        metrics = {"epoch": epoch, "train_loss": running / len(train_loader),
                   "dev_loss": dev_loss, "learning_rate": scheduler.get_last_lr()[0],
                   "completed_updates": completed_updates,
                   "skipped_loss_batches": skipped_loss_batches,
                   "skipped_gradient_updates": skipped_gradient_updates}
        print(json.dumps(metrics), flush=True)
        if dev_loss is not None and dev_loss < best:
            best = dev_loss
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")
    (output / "training.json").write_text(json.dumps(
        {"arguments": vars(args), "class_counts": counts,
         "best_dev_loss": None if best == float("inf") else best},
        ensure_ascii=False, indent=2, default=dict
    ) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
