"""Shared pieces used by every script: prompts, answer checking, rewards,
data loading and model loading. Keeping them in one file guarantees that
training and evaluation use exactly the same prompt and the same checker."""

import os
import random
import re

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GSM8K_INSTRUCTION = (
    "Solve this step by step. At the end, write the final answer on its own "
    "line in the form: #### <number>"
)
MATH_INSTRUCTION = "Solve this step by step. Put your final answer in \\boxed{}."
MMLU_INSTRUCTION = "Answer with only the letter (A, B, C, or D) of the correct option."


def gsm8k_prompt(question):
    return f"{question}\n\n{GSM8K_INSTRUCTION}"


def math_prompt(problem):
    return f"{problem}\n\n{MATH_INSTRUCTION}"


def mmlu_prompt(question, choices):
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
    return f"{question}\n\n{options}\n\n{MMLU_INSTRUCTION}"


def to_chat(tokenizer, user_text):
    """Wrap a question in the model's chat format, ending where the
    assistant's reply should begin."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# GSM8K: numeric answers
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_gold(answer_field):
    """GSM8K stores the solution with the final answer after '####'."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def extract_gsm8k(text):
    """The number after the last '####'. If the model never wrote '####',
    fall back to the last number anywhere in its reply."""
    if "####" in text:
        match = _NUMBER.search(text.split("####")[-1])
        if match:
            return _to_float(match.group())
    numbers = _NUMBER.findall(text)
    return _to_float(numbers[-1]) if numbers else None


def gsm8k_correct(text, gold):
    pred, target = extract_gsm8k(text), _to_float(gold)
    return pred is not None and target is not None and abs(pred - target) < 1e-6


# ---------------------------------------------------------------------------
# MATH-500: LaTeX answers inside \boxed{...}
# ---------------------------------------------------------------------------

def last_boxed(text):
    """Contents of the last \\boxed{...}, matching nested braces."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None  # unbalanced braces: the reply was probably cut off


def normalize_math(s):
    """Strip formatting that doesn't change the value, so '\\dfrac{1}{2}'
    and '\\frac12' style differences don't count as wrong."""
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s.strip())
    for old, new in [
        ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
        ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^\\circ", ""),
        ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("$", ""), (" ", ""),
    ]:
        s = s.replace(old, new)
    return s.rstrip(".")


def math_correct(text, gold):
    pred = last_boxed(text)
    if pred is None:
        return False
    # math-verify checks mathematical equivalence (e.g. 0.5 == \frac{1}{2}).
    # Its timeout uses Linux signals and spawns broken helper processes on
    # Windows, so on a Windows laptop we skip it and compare strings only.
    if os.name != "nt":
        try:
            from math_verify import parse, verify
            if verify(parse(f"${gold}$"), parse(f"${pred}$")):
                return True
        except Exception:
            pass
    return normalize_math(pred) == normalize_math(gold)


# ---------------------------------------------------------------------------
# MMLU: one letter
# ---------------------------------------------------------------------------

def extract_letter(text):
    match = re.search(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([ABCD])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


def mmlu_correct(text, gold):
    return extract_letter(text) == gold


# ---------------------------------------------------------------------------
# GRPO reward
# ---------------------------------------------------------------------------

def gsm8k_rewards(completions, gold):
    """1.0 for a correct final answer, plus 0.1 for using the '#### <number>'
    format. The small format bonus teaches the format without letting the
    model earn much reward from format alone."""
    rewards = []
    for text in completions:
        r = 1.0 if gsm8k_correct(text, gold) else 0.0
        if re.search(r"####\s*-?\$?\d", text):
            r += 0.1
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_gsm8k(split):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"prompt": gsm8k_prompt(r["question"]), "gold": gsm8k_gold(r["answer"]),
         "solution": r["answer"]}
        for r in ds
    ]


def load_math500():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"prompt": math_prompt(r["problem"]), "gold": r["answer"]} for r in ds]


def load_mmlu(n=500, seed=0):
    """A fixed random 500-question subset, so every model sees the same one."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [
        {"prompt": mmlu_prompt(ds[i]["question"], ds[i]["choices"]),
         "gold": "ABCD"[ds[i]["answer"]]}
        for i in idx
    ]


# name -> (loader, checker, max new tokens)
BENCHMARKS = {
    "gsm8k": (lambda: load_gsm8k("test"), gsm8k_correct, 512),
    "math500": (load_math500, math_correct, 1024),
    "mmlu": (load_mmlu, mmlu_correct, 32),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_config(r=16):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")


def load_model(name, adapter=None, dtype="bf16", device="cuda"):
    """Load a model and tokenizer. If `adapter` is a LoRA folder, merge it in."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # generation needs prompts aligned on the right
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model, tok