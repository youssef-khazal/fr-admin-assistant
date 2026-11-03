"""Fine-tune Qwen3-4B with LoRA (QLoRA: the base model in 4-bit) on the examples from build_sft.py.

    python finetune/train_lora.py                      # defaults below, about 3-4 hours on an 8 GB laptop GPU
    python finetune/train_lora.py --epochs 1 --limit 50  # quick test

Only the small LoRA add-on is trained; the 4 billion weights of the base model stay frozen. The loss is computed
on the answer only (the JSON the model must write), not on the prompt, and the output layer is applied only to
those positions, which keeps the memory low enough for 8 GB. The best add-on (lowest validation loss) is saved
to finetune/out/<name>/ and can be loaded with LLM_BACKEND=hf HF_ADAPTER=finetune/out/<name>.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations import sdpa_attention

# Qwen3 has 32 query heads but only 8 key/value heads. By default transformers hands them to PyTorch as they are
# ("enable_gqa"), which the memory-saving attention kernel of the Windows build refuses, so PyTorch falls back to the
# plain kernel that stores a 6000 x 6000 table per head and runs out of memory. Copying the 8 key/value heads up to
# 32 first (a few MB) lets the memory-saving kernel run.
sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def load(path: Path, tok, max_len: int, limit: int | None, repeat: dict[str, int] | None = None,
         longest_first: bool = False) -> list[tuple[torch.Tensor, int]]:
    """(token ids of prompt + answer, index where the answer starts); examples longer than max_len are dropped.
    repeat {"fallback": 3, ...} shows rare kinds of example several times per epoch."""
    out, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if longest_first:
        rows.sort(key=lambda r: -sum(len(m["content"]) for m in r["messages"]))
    for r in rows[:limit] if limit else rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(r["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(a_ids) > max_len:
            dropped += 1
            continue
        item = (torch.tensor(p_ids + a_ids), len(p_ids))
        out.extend([item] * (repeat or {}).get(r.get("kind", ""), 1))
    lengths = sorted(len(ids) for ids, _ in out)
    print(f"{path.name}: {len(out)} examples ({dropped} longer than {max_len} tokens dropped); "
          f"tokens median {lengths[len(lengths) // 2] if lengths else 0}, max {lengths[-1] if lengths else 0}")
    return out


def _ce_sum(h: torch.Tensor, target: torch.Tensor, lm_head) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = lm_head(h)
    return torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum")


def answer_loss(model, ids: torch.Tensor, start: int, chunk: int = 128) -> torch.Tensor:
    """Cross-entropy on the answer tokens only. The output layer (150k words wide) is applied 128 positions at a
    time and recomputed during the backward pass, so its big score tables never all sit in memory at once."""
    inner = model.base_model.model  # Qwen3ForCausalLM inside the PEFT wrapper
    with torch.autocast("cuda", dtype=torch.bfloat16):  # the LoRA weights are float32; compute in bfloat16
        hidden = inner.model(input_ids=ids[None].to(inner.device), use_cache=False).last_hidden_state
    h = hidden[0, start - 1:-1]
    target = ids[start:].to(h.device)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, len(target), chunk):
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_sum, h[i:i + chunk], target[i:i + chunk], inner.lm_head, use_reentrant=False)
        else:
            total = total + _ce_sum(h[i:i + chunk], target[i:i + chunk], inner.lm_head)
    return total / len(target)


def check_attention() -> None:
    """Long examples only fit if attention uses a memory-saving kernel (not the plain one that stores a
    6000 x 6000 table per head). Stop with a clear message if none is available."""
    q = torch.randn(1, 32, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("memory-efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel([backend]):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()
            print(f"attention kernel: {name}")
            return
        except RuntimeError:
            continue
    raise SystemExit("no memory-saving attention kernel on this GPU/PyTorch: lower --max-len to about 3000")


@torch.no_grad()
def evaluate(model, data) -> float:
    model.eval()
    losses = [answer_loss(model, ids, start).item() for ids, start in data]
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", type=Path, default=Path("finetune/data"))
    p.add_argument("--name", default="qwen3-4b-demarches")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--accum", type=int, default=8, help="examples per optimizer step")
    p.add_argument("--max-len", type=int, default=8192, help="examples longer than this (in tokens) are dropped")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--longest-first", action="store_true", help="memory test: train on the longest examples first")
    p.add_argument("--repeat", default="fallback:3,clarify:2,special:2",
                   help="rare kinds of example shown several times per epoch (the teacher rarely said 'not in the guides')")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    repeat = {k: int(v) for k, v in (x.split(":") for x in a.repeat.split(",") if x)}
    train = load(a.data / "train.jsonl", tok, a.max_len, a.limit, repeat, a.longest_first)
    val = load(a.data / "val.jsonl", tok, a.max_len, a.limit and max(5, a.limit // 10))

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=quant, device_map="cuda", dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    # gradient checkpointing: keep only each layer's input and recompute the rest during the backward pass.
    # The model stays in bfloat16 (no upcast of the norms to float32, which doubles the memory of every activation).
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    check_attention()
    print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / a.accum) * a.epochs
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    out = Path("finetune/out") / a.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    # forbid the memory-hungry attention kernel, so a problem shows up as a clear error instead of "out of memory"
    allowed = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    allowed.__enter__()
    try:
        best = evaluate(model, val)
    except RuntimeError as e:
        if "kernel" not in str(e).lower():
            raise
        allowed.__exit__(None, None, None)
        print("warning: the model's attention cannot use the memory-saving kernel; if memory runs out, lower --max-len")
        best = evaluate(model, val)
    print(f"validation loss before training: {best:.4f}")
    log.write(json.dumps({"epoch": 0, "val_loss": best}) + "\n")
    t0, step = time.time(), 0
    model.train()
    for epoch in range(1, a.epochs + 1):
        if not a.longest_first:
            random.shuffle(train)
        running = []
        for i, (ids, start) in enumerate(train):
            loss = answer_loss(model, ids, start) / a.accum
            loss.backward()
            running.append(loss.item() * a.accum)
            del loss
            if (i + 1) % a.accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                torch.cuda.empty_cache()
                if step % 10 == 0 or a.limit:
                    print(f"epoch {epoch} step {step}/{steps}  train loss {sum(running[-80:]) / len(running[-80:]):.4f}  "
                          f"{(time.time() - t0) / 60:.0f} min, GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        v = evaluate(model, val)
        print(f"epoch {epoch}: train loss {sum(running) / len(running):.4f}, validation loss {v:.4f}")
        log.write(json.dumps({"epoch": epoch, "train_loss": sum(running) / len(running), "val_loss": v,
                              "minutes": (time.time() - t0) / 60}) + "\n")
        log.flush()
        if v < best:
            best = v
            model.save_pretrained(out)
            tok.save_pretrained(out)
            print(f"saved to {out}")
    if not (out / "adapter_config.json").exists():  # validation never improved: keep the last state anyway
        model.save_pretrained(out)
        tok.save_pretrained(out)
    (out / "config_run.json").write_text(json.dumps({**vars(a), "data": str(a.data), "train": len(train), "val": len(val),
                                                     "best_val_loss": best}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
