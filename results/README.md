# Results

Everything behind the numbers in the main README. All files are plain JSON / JSON Lines.

## Search
| file | what it holds | made by |
|---|---|---|
| `retrieval.json` | hit@1, hit@5 and MRR of every search setup tried (BM25, bge-m3, e5-large, hybrid, reranker, chunking strategies), on all / dev / test questions | `scripts/eval_retrieval.py` |
| `candidates_gpu.json`, `runs_gpu.json` | the raw ranked passages and sheets returned by each setup, per question | `scripts/search_gpu.py` |
| `rewrites.json` | the questions as rewritten by the LLM (the rejected query-rewriting experiment) | `scripts/search_gpu.py` |
| `search_v5.json` | the search used in the final version, on the 163 questions and the 120 rewordings (key `v5`), against the plain search of v4 (key `v4`) | `scripts/eval_search.py` |
| `followup.json` | the follow-up detector on the 28 test cases of `data/eval/followups.yaml` | `scripts/eval_followup.py` |

## Answers
| file | what it holds |
|---|---|
| `answers.jsonl` (v1), `answers_v2.jsonl`, `answers_v4.jsonl` … `answers_v7.jsonl`, `answers_v7fast.jsonl` | every answer of every version: behaviour, text, citations, the passages the model saw, and the checks done in code. Made by `scripts/run_answers.py`. |
| `judgements2_*.jsonl` | the LLM judge's verdict on each answer: right behaviour, expected facts found, unsupported statements |
| `manual_grades_v4.jsonl` … `manual_grades_v7.jsonl` | the hand grading of all 163 answers of v4 to v7: right behaviour (yes/no), each expected fact (found/missing), and a short note for every statement that is invented or applied to the wrong situation |
| `scores2.json` | judge scores per version, per category and per language. Made by `scripts/score_answers.py`. |
| `manual_scores.json` | hand-grading scores with 95% bootstrap intervals, and the paired differences between versions |

Question ids (`q001` … `q163`) refer to `data/eval/questions.yaml`, which also gives the expected behaviour, the official sheets that answer each question and the facts the answer should contain.
