"""Download the models and datasets into the Hugging Face cache ($HF_HOME).

Run this on a Grace LOGIN node (it has internet; compute nodes don't), and
on your laptop. Jobs then load everything from the cache.
    python download.py
"""

from datasets import load_dataset
from huggingface_hub import snapshot_download

for model in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
    print("Downloading", model)
    snapshot_download(model)

print("Downloading GSM8K")
load_dataset("openai/gsm8k", "main")
print("Downloading MATH-500")
load_dataset("HuggingFaceH4/MATH-500")
print("Downloading MMLU")
load_dataset("cais/mmlu", "all", split="test")
print("All downloads done")
