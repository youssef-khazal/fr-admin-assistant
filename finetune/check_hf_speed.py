"""2-minute speed check of the 4B on the Hugging Face backend (why is one answer so slow?).

    python finetune/check_hf_speed.py
"""
import json
import os
import time

import torch

os.environ.setdefault("LLM_BACKEND", "hf")
from fr_admin_assistant import llm  # noqa: E402

free, total = torch.cuda.mem_get_info()
print(f"GPU before loading: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB (less than ~7 free = another program uses the GPU)")
row = json.loads(open("finetune/data/val.jsonl", encoding="utf-8").readline())
tok, model = llm._hf_load()
prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt").to(model.device)
print("prompt tokens:", ids["input_ids"].shape[1], "| use_cache in config:", model.config.use_cache,
      "| eos in generation config:", model.generation_config.eos_token_id)

def timed(label, fn):
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    print(f"{label}: {time.time() - t:.1f} s")

timed("read the prompt once (prefill)", lambda: model(**ids, logits_to_keep=1))
timed("write 1 token", lambda: model.generate(**ids, max_new_tokens=1, do_sample=False))
timed("write 64 tokens with cache", lambda: model.generate(**ids, max_new_tokens=64, min_new_tokens=64, do_sample=False, use_cache=True))
free, total = torch.cuda.mem_get_info()
print(f"GPU at the end: {free / 2**30:.1f} GB free; peak used by this script {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
os.environ["HF_TIMING"] = "1"
llm._hf_chat(row["messages"][:-1])
