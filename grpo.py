"""GRPO (Group Relative Policy Optimization), written from scratch.

One training step:
  1. Sample: for each of `prompts_per_step` GSM8K questions, the model writes
     `group_size` different answers (random sampling, so they differ).
  2. Reward: score each answer (1.0 correct, +0.1 for the '####' format).
  3. Advantage: compare each answer to the other answers for the SAME
     question: (reward - group mean) / group std. Better than its siblings
     -> positive, worse -> negative. This group average replaces the value
     network that PPO needs.
  4. Loss: push up the log-probability of tokens in positive-advantage answers
     and push down negative ones, plus a KL penalty that keeps the model close
     to the original (reference) model so it doesn't drift into nonsense.
  5. One optimizer step on the LoRA weights.

Laptop smoke test (GTX 1050):
  python grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp32 --steps 3 \
      --prompts_per_step 1 --group_size 4 --max_new_tokens 200 --micro_batch 1 \
      --grad_ckpt --out runs/grpo_test
HPRC (A100):
  python grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 400 --out runs/grpo
"""

import argparse
import json
import os
import random
import time

import torch
from peft import get_peft_model

from common import gsm8k_rewards, load_gsm8k, load_model, lora_config, to_chat


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--prompts_per_step", type=int, default=8, help="questions per step")
    ap.add_argument("--group_size", type=int, default=8, help="answers sampled per question")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty weight")
    ap.add_argument("--micro_batch", type=int, default=8, help="sequences per forward/backward pass")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad_ckpt", action="store_true", help="trade speed for memory")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=20, help="save example answers every N steps")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def completion_mask(comp_ids, stop_ids):
    """1 for every real token of an answer up to and including its first
    end-of-text token, 0 for the padding after it. Padding must not count
    in the loss."""
    is_stop = torch.isin(comp_ids, stop_ids)
    length = comp_ids.size(1)
    first_stop = torch.where(
        is_stop.any(dim=1),
        is_stop.int().argmax(dim=1),  # position of the first stop token
        torch.full((comp_ids.size(0),), length, device=comp_ids.device),  # never stopped: keep all
    )
    positions = torch.arange(length, device=comp_ids.device).unsqueeze(0)
    return (positions <= first_stop.unsqueeze(1)).long()


def token_logprobs(model, input_ids, attention_mask, n_completion):
    """log p(token | everything before it) for each completion token.

    The model's output at position t predicts token t+1, so the logits for
    the completion start one position before it. `logits_to_keep` asks the
    model for only the last n_completion+1 positions, which saves a lot of
    memory (the vocabulary has ~152k entries per position)."""
    # Prompts are left-padded, so positions must count only real tokens,
    # matching what generate() used while sampling.
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=n_completion + 1,
    ).logits[:, :-1, :].float()
    targets = input_ids[:, -n_completion:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model, tok = load_model(args.model, dtype=args.dtype, device=args.device)
    # The policy is base model + LoRA. The reference model is the same base
    # with LoRA switched off, so we don't need a second copy in memory.
    model = get_peft_model(model, lora_config(args.lora_r))
    model.print_trainable_parameters()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    # Token ids that end an answer (Qwen uses <|im_end|>, and pads with <|endoftext|>)
    stop = model.generation_config.eos_token_id
    stop = set(stop if isinstance(stop, list) else [stop]) | {tok.eos_token_id, tok.pad_token_id}
    stop_ids = torch.tensor(sorted(stop), device=args.device)

    data = load_gsm8k("train")
    G = args.group_size
    log_f = open(os.path.join(args.out, "log.jsonl"), "a")
    samples_f = open(os.path.join(args.out, "samples.jsonl"), "a")
    t_start = time.time()

    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = random.sample(data, args.prompts_per_step)
        enc = tok([to_chat(tok, ex["prompt"]) for ex in batch],
                  return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        prompt_len = enc["input_ids"].shape[1]

        # ---- 1. Sample G answers per question ---------------------------------
        model.eval()
        with torch.no_grad():
            seqs = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0, top_k=0, repetition_penalty=1.0,  # sample from the model's own distribution
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=G,  # rows come out grouped: q0 x G, q1 x G, ...
                pad_token_id=tok.pad_token_id,
            )
        model.train()

        comp = seqs[:, prompt_len:]
        n_comp = comp.size(1)
        mask = completion_mask(comp, stop_ids)
        attn = torch.cat([enc["attention_mask"].repeat_interleave(G, dim=0), mask], dim=1)
        texts = tok.batch_decode(comp, skip_special_tokens=True)

        # ---- 2. Rewards --------------------------------------------------------
        rewards = []
        for i, ex in enumerate(batch):
            rewards += gsm8k_rewards(texts[i * G : (i + 1) * G], ex["gold"])
        rewards = torch.tensor(rewards, device=args.device)

        # ---- 3. Group-relative advantages --------------------------------------
        grouped = rewards.view(-1, G)
        adv = ((grouped - grouped.mean(dim=1, keepdim=True))
               / (grouped.std(dim=1, keepdim=True) + 1e-4)).view(-1)
        # If all G answers got the same reward, every advantage is 0: that
        # question teaches nothing this step (all right or all wrong).

        # ---- 4. Reference log-probs (LoRA off, no gradients) -------------------
        mb = args.micro_batch
        with torch.no_grad(), model.disable_adapter():
            ref_lp = torch.cat([
                token_logprobs(model, seqs[s : s + mb], attn[s : s + mb], n_comp)
                for s in range(0, seqs.size(0), mb)
            ])

        # ---- 5. Policy loss and update -----------------------------------------
        opt.zero_grad()
        n_seq = seqs.size(0)
        kl_total, tok_total = 0.0, 0.0
        for s in range(0, n_seq, mb):
            sl = slice(s, s + mb)
            lp = token_logprobs(model, seqs[sl], attn[sl], n_comp)
            m = mask[sl].float()

            # ratio = pi / pi_old. We take one update per batch of samples, so
            # pi_old is the current model: the ratio's value is exactly 1, but
            # its gradient is the gradient of log pi, which is what we need.
            ratio = torch.exp(lp - lp.detach())
            policy_term = -ratio * adv[sl].unsqueeze(1)

            # KL(pi || ref) per token, the low-variance "k3" estimator from the
            # DeepSeekMath paper: exp(ref - lp) - (ref - lp) - 1, always >= 0.
            diff = ref_lp[sl] - lp
            kl = torch.exp(diff) - diff - 1

            per_token = policy_term + args.beta * kl
            per_seq = (per_token * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)  # average over the answer's tokens
            (per_seq.sum() / n_seq).backward()  # average over all answers in the step

            kl_total += (kl.detach() * m).sum().item()
            tok_total += m.sum().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        opt.step()

        # ---- Logging -----------------------------------------------------------
        rec = {
            "step": step,
            "reward": round(rewards.mean().item(), 4),
            "accuracy": round((rewards >= 1.0).float().mean().item(), 4),
            "format_rate": round(sum("####" in t for t in texts) / len(texts), 4),
            "completion_tokens": round(mask.sum(dim=1).float().mean().item(), 1),
            "kl": round(kl_total / max(tok_total, 1), 5),
            "grad_norm": round(grad_norm, 4),
            "useful_groups": int((grouped.std(dim=1) > 0).sum().item()),
            "step_seconds": round(time.time() - t0, 1),
            "minutes": round((time.time() - t_start) / 60, 1),
        }
        print(rec, flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

        if step % args.sample_every == 0 or step == 1:
            samples_f.write(json.dumps({
                "step": step, "question": batch[0]["prompt"], "gold": batch[0]["gold"],
                "answers": texts[:G], "rewards": rewards[:G].tolist(),
            }) + "\n")
            samples_f.flush()

        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(os.path.join(args.out, f"step_{step}"))
            print(f"Saved checkpoint step_{step}", flush=True)


if __name__ == "__main__":
    main()
