"""Supervised fine-tuning (SFT) baseline: teach the model to imitate the
written GSM8K solutions, with LoRA. This is the comparison point for GRPO:
SFT learns from correct answers someone wrote, GRPO learns from its own
attempts and a reward.

Example (HPRC):
  python sft.py --model Qwen/Qwen2.5-1.5B-Instruct --out runs/sft
Laptop smoke test:
  python sft.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp32 --limit 32 --batch_size 2 --out runs/sft_test
"""

import argparse
import json
import os
import random
import re
import time

import torch
from peft import get_peft_model

from common import load_gsm8k, load_model, lora_config, to_chat


def clean_solution(solution):
    """GSM8K solutions contain calculator notes like <<48/2=24>>. Remove them;
    the '#### answer' line at the end already matches our prompt's format."""
    return re.sub(r"<<.*?>>", "", solution).strip()


def build_examples(data, tok, max_len):
    """Each example = prompt tokens + answer tokens. Labels are -100 on the
    prompt so the loss only counts the answer: we train the model to write
    solutions, not to repeat questions."""
    examples = []
    for ex in data:
        prompt_ids = tok(to_chat(tok, ex["prompt"]), add_special_tokens=False)["input_ids"]
        answer_ids = tok(clean_solution(ex["solution"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        examples.append((ids, labels))
    return examples


def collate(batch, pad_id, device):
    """Pad a batch to the same length (on the right; padding is ignored)."""
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    to = lambda x: torch.tensor(x, device=device)
    return to(input_ids), to(labels), to(attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8, help="examples per forward pass")
    ap.add_argument("--grad_accum", type=int, default=2, help="forward passes per optimizer step")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    model, tok = load_model(args.model, dtype=args.dtype, device=args.device)
    model = get_peft_model(model, lora_config(args.lora_r))  # freezes the base, adds small trainable matrices
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    examples = build_examples(load_gsm8k("train")[: args.limit], tok, args.max_len)
    print(f"{len(examples)} training examples")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    steps_per_epoch = len(examples) // (args.batch_size * args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, total_steps // 20)
    # LR ramps up for the first 5% of steps, then decays linearly to 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / total_steps))

    log = open(os.path.join(args.out, "log.jsonl"), "w")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(examples)
        batches = [examples[i : i + args.batch_size] for i in range(0, len(examples), args.batch_size)]
        for b, batch in enumerate(batches):
            input_ids, labels, attn = collate(batch, tok.pad_token_id, args.device)
            # Passing labels makes the model compute the next-token cross-entropy loss itself
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            (loss / args.grad_accum).backward()

            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0 or step == total_steps:
                    rec = {"step": step, "loss": round(loss.item(), 4),
                           "lr": sched.get_last_lr()[0], "minutes": round((time.time() - t0) / 60, 1)}
                    print(rec, flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()

    model.save_pretrained(args.out)  # saves only the LoRA adapter (a few MB)
    tok.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
