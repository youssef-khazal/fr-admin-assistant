# Fine-tuning a smaller model: Qwen3-4B with QLoRA

Can a model half the size learn to answer like the final assistant (v7, Qwen3-8B with reasoning), if it is trained on v7's own answers? This folder is that experiment: synthetic training data made with the assistant itself, a LoRA add-on trained on an 8 GB laptop GPU, and the same evaluation as the main project (163 test questions, LLM judge, hand grading).

**Result in short:** the fine-tuned 4B is clearly better than the 4B it started from. It finds more of the expected facts and picks the right behaviour more often, in particular saying "this is not in the guides" when it should. It does **not** reach v7, the 8B model with reasoning. v7 remains the model used by the app.

## Method

```
generated questions ──► search (same as the app) ──► teacher: v7, reasoning on ──► checks ──► 658 examples
                                                                                               │
                Qwen3-4B (4-bit) + LoRA add-on  ◄── train on the answer only (2 passes) ◄─────┘
```

1. **Questions** (`make_data.py`, phase 1). Qwen3-8B writes questions from passages of the official sheets, in five styles weighted like real users: formal French, spoken French, SMS French, English and Arabic. There are three kinds:
   - *answer*: the question is answered by the passage it was made from;
   - *clarify*: the answer depends on a situation the question does not give;
   - *fallback*: the sheet that answers is removed from the passages, so the right reply is "not in the guides".

   28 hand-written off-topic and prompt-injection requests are added. **The 55 sheets that answer the 163 test questions are excluded**, so no test answer is seen in training.
2. **Search** (phase 2): the app's own search (BM25 + bge-m3 + reranker) picks the 8 passages for each question.
3. **Teacher** (phase 3): the full v7 assistant answers, with reasoning on and all its checks in code. Its JSON answer is the training target, and the prompt is built exactly as in the app (`Assistant.build_messages`).
4. **Checks** (`build_sft.py`). An answer is kept only if:
   - it is in the question's language;
   - it has no unsupported number;
   - it cites the sheet the question was made from, and only passages of the same situation;
   - its behaviour is the expected one for its kind;
   - it has no stray characters from another script (the 8B model sometimes slips Cyrillic or Chinese into Arabic).

   Of 1,128 generated questions, **658 examples are kept** (595 for training, 63 for validation, split by sheet). The main reasons for dropping: wrong language (96), the teacher answering although the right sheet had been removed (106), and citing a passage of another situation (56).
5. **Training** (`train_lora.py`):
   - QLoRA: the base model is loaded in 4-bit (nf4); a LoRA add-on of rank 16 is trained on all attention and MLP layers (33 M parameters, 0.8% of the model).
   - The loss is computed on the answer only, not on the prompt.
   - Rare kinds are shown several times per pass (fallback ×3, clarify ×2, off-topic/injection ×2), for 695 examples per pass.
   - 2 passes, learning rate 2e-4, 8 examples per step.

   The add-on with the lowest validation loss is kept:

   | | validation loss |
   |---|---|
   | before training | 0.748 |
   | after pass 1 | **0.398** (kept) |
   | after pass 2 | 0.416 (starts to overfit) |

   Training took 3 h 15 on an 8 GB laptop GPU, with a peak of 6.1 GB.

### Making it fit in 8 GB

Examples are long: median 4,300 tokens, up to 7,500, because the prompt holds 8 passages. Three things made training and inference fit:
- **The memory-saving attention kernel.** Qwen3 has 32 query heads for 8 key/value heads. Handed over as they are, the Windows build of PyTorch falls back to the plain attention kernel, which stores a 6,000 × 6,000 table per head. Copying the key/value heads up to 32 first lets the memory-efficient kernel run. For one answer, this took the time from 274 s to 7 s and the memory from 9.0 GB (overflowing into the PC's RAM) to 3.4 GB.
- **The output layer only where it is needed.** The 150k-word output layer is applied only on the answer positions, 128 at a time, and recomputed during the backward pass.
- **Gradient checkpointing**, with the model kept in bfloat16 (no upcast of the norms to float32).

## Results

Same 163 questions as the main project, same search, same prompt and checks in code. Only the model changes. 95% bootstrap intervals; paired differences in [ ]. All runs answer without the reasoning step, except v7.

**LLM judge** (all four runs):

| model | right behaviour | facts found | answers with invented content | time per answer |
|---|---|---|---|---|
| Qwen3-4B, before fine-tuning | 73.6% | 72.3% | 32.5% | ~11 s |
| **Qwen3-4B + LoRA** | **79.8%** | **78.7%** | 30.1% | ~25 s |
| v7 (Qwen3-8B + reasoning) | 89.0% | 86.6% | 24.5% | ~70 s |
| fine-tuned − base | +6.1 [0.0, +12.9] | **+6.3 [+1.2, +11.6]** | −2.5 [−10.4, +4.9] | |

**Hand grading** (all 163 answers of the fine-tuned model, same method as v4–v7):

| model | right behaviour | facts found | answers with invented content |
|---|---|---|---|
| **Qwen3-4B + LoRA** | 85.9% [80.4, 90.8] | 76.3% [70.8, 81.8] | 27.0% [20.2, 33.7] |
| v7 (Qwen3-8B + reasoning) | 92.6% [88.3, 96.3] | 81.4% [76.8, 86.1] | 15.3% [9.8, 21.5] |
| fine-tuned − v7 | −6.7 [−11.7, −2.5] | −5.1 [−9.2, −1.2] | +11.7 [+4.3, +19.0] |

**What this shows:**
- **Fine-tuning works, mostly on behaviour.** The questions not covered by the guides show it best:
  - the base 4B labels its reply "not in the guides" on 1 of 11, the fine-tuned model on 9 of 11;
  - by the judge, the fine-tuned model handles 7 of 11 well, against 3 of 11 for the base 4B.

  The gain in facts found (+6 points) is outside the noise; the gain in behaviour is at its edge.
- **It does not catch up with v7**, which is twice the size and reasons before answering. The largest gap is invented content: 27% against 15% of answers. As with v7, most errors mix up rules between neighbouring cases. For example:
  - a renewal delay taken from a weapons-licence sheet;
  - the civil-service rule given to a private employee;
  - "17 (SAMU)" in an emergency answer, where 17 is the police and 15 the SAMU.
- **Arabic is where the small model breaks down.** 10 of the 15 Arabic answers have a wrong detail, against about 1 in 4 in French and English. Examples: a rental-deposit question answered with the rules for buying a new-build home, and one answer that repeats the same sentence. The training set had 56 Arabic examples, too few to fix this.
- **The judge is less reliable on this model:** it agrees with the hand grading on 85% of behaviour verdicts, 90% of facts and 75% of invented-content verdicts.

**Decision:** the app keeps v7. The fine-tuned 4B is about 3× faster, but it makes nearly twice as many errors in content, and for administrative information a wrong detail costs more than a slow answer.

### Side result: v7 at a smaller context

Making the training data meant running v7 hundreds of times, which showed that its context size of 12,288 tokens did not fit in the 8 GB GPU: about 13% of the model ran on the CPU, at about 4 minutes per answer. At 8,192 tokens it runs fully on the GPU, at about 70 s per answer. Re-running the 163 questions at 8,192 gives the same quality within the noise (judge):

| | behaviour | facts | invented |
|---|---|---|---|
| 8,192 − 12,288 | −1.8 [−6.8, +3.1] | −1.6 [−4.6, +1.6] | +1.8 [−5.5, +9.8] |

8,192 is now the default (`NUM_CTX_THINK` in `pipeline.py`).

## Reproduce

Windows, NVIDIA GPU with 8 GB, Ollama with `qwen3:8b`, and the corpus and index of the main project built.

```powershell
pip install -e .[app,finetune]
python finetune/make_data.py          # questions, search, teacher answers (resumable; about 7 h)
python finetune/build_sft.py          # checks -> finetune/data/train.jsonl, val.jsonl, stats.json
powershell -ExecutionPolicy Bypass -File finetune\run_all.ps1
```

`run_all.ps1` runs the rest in one go, for 8–10 hours:
- training;
- the 163 questions answered by the fine-tuned 4B, the base 4B and v7 at 8,192;
- the LLM judge on the three runs.

Every step is resumable. Then:

```bash
python scripts/compare_runs.py base4b=results/judgements2_base4b.jsonl ft4b=results/judgements2_ft4b.jsonl \
    v7=results/judgements2_v7.jsonl v7ctx8k=results/judgements2_v7ctx8k.jsonl
python scripts/compare_runs.py v7=results/manual_grades_v7.jsonl ft4b=results/manual_grades_ft4b.jsonl
```

To use the fine-tuned model in the app instead of Ollama: `LLM_BACKEND=hf HF_ADAPTER=finetune/out/qwen3-4b-demarches`.

| file | what it does |
|---|---|
| `make_data.py` | generates the questions, runs the search and the teacher (resumable) |
| `build_sft.py` | keeps the answers that pass the checks, splits by sheet |
| `train_lora.py` | QLoRA training; saves the best add-on to `finetune/out/<name>/` |
| `run_all.ps1` | training, the three evaluation runs and the judge in one command |
| `check_hf_speed.py` | 2-minute speed and memory check of the 4B on the GPU |

The generated data and the trained add-on are not in the repository (`finetune/data/`, `finetune/out/`).
