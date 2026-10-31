# GRPO from scratch on Qwen2.5-1.5B-Instruct

Reinforcement learning with verifiable rewards on GSM8K math word problems. GRPO
(Group Relative Policy Optimization) is implemented from scratch in PyTorch
(`grpo.py`): group sampling, rule-based rewards, group-relative advantages, and a
KL-penalized policy-gradient loss, with LoRA on a single A100.

## Results

| | Zero-shot | SFT (LoRA) | GRPO (LoRA) |
|---|---|---|---|
| GSM8K test accuracy (%) | | | |
| MATH-500 accuracy (%) | | | |
| MMLU, 500-question subset (%) | | | |
| Avg. GSM8K answer length (tokens) | | | |

Greedy decoding for all evaluations. MMLU checks that math training didn't hurt general knowledge.

## What I found

(Training curves in `runs/grpo/curves.png`; reward-hacking examples here.)

## Setup

| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct, LoRA rank 16 on all attention and MLP projections |
| GRPO | 8 questions x 8 sampled answers per step, temperature 1.0, 384 new tokens, lr 2e-5, KL beta 0.04 |
| Reward | 1.0 for a correct final answer, +0.1 for the `#### <number>` format |
| SFT | 1 epoch on GSM8K train, lr 1e-4, effective batch 16 |

## Files

- `grpo.py`: the GRPO training loop
- `sft.py`: supervised fine-tuning baseline
- `eval.py`: evaluation on GSM8K, MATH-500, MMLU
- `common.py`: prompts, answer checkers, reward, data and model loading
- `test_common.py`: tests for the answer checkers
- `jobs/gpu.slurm`: Slurm job script (TAMU HPRC Grace)
