"""Evaluate a model on GSM8K, MATH-500 and an MMLU subset.

Examples:
  # laptop, quick check on 20 questions
  python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp32 --benchmarks gsm8k --limit 20 --batch_size 4

  # HPRC, zero-shot baseline
  python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --name zeroshot

  # HPRC, a trained LoRA adapter
  python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/sft --name sft
"""

import argparse
import json
import os
import time

import torch

from common import BENCHMARKS, load_model, to_chat


def generate(model, tok, prompts, max_new_tokens, batch_size):
    """Greedy decoding (always pick the most likely token), so results are
    repeatable: the same model gives the same score every time."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        chats = [to_chat(tok, p) for p in prompts[i : i + batch_size]]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,  # silence sampling warnings
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]  # drop the prompt
        outputs += tok.batch_decode(new_tokens, skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="folder with a trained LoRA adapter")
    ap.add_argument("--name", default="run", help="label used in the output file names")
    ap.add_argument("--benchmarks", default="gsm8k,math500,mmlu")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (for testing)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, tok = load_model(args.model, args.adapter, args.dtype, args.device)
    model.eval()

    summary = {"model": args.model, "adapter": args.adapter}
    for bench in args.benchmarks.split(","):
        loader, checker, max_new = BENCHMARKS[bench]
        data = loader()[: args.limit]
        print(f"{bench}: {len(data)} questions, max {max_new} new tokens", flush=True)

        t0 = time.time()
        outputs = generate(model, tok, [d["prompt"] for d in data], max_new, args.batch_size)
        correct = [checker(o, d["gold"]) for o, d in zip(outputs, data)]
        lengths = [len(tok(o, add_special_tokens=False)["input_ids"]) for o in outputs]

        acc = 100 * sum(correct) / len(correct)
        summary[bench] = {
            "accuracy": round(acc, 2),
            "n": len(data),
            "avg_completion_tokens": round(sum(lengths) / len(lengths), 1),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        print(f"{bench}: {acc:.2f}% ({sum(correct)}/{len(correct)})", flush=True)

        # Save every answer so you can read what the model actually wrote.
        with open(os.path.join(args.out_dir, f"{args.name}_{bench}_outputs.jsonl"), "w") as f:
            for d, o, c in zip(data, outputs, correct):
                f.write(json.dumps({"prompt": d["prompt"], "gold": d["gold"],
                                    "output": o, "correct": c}) + "\n")

    with open(os.path.join(args.out_dir, f"{args.name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
