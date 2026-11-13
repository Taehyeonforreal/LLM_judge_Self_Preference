# Data

`data/raw/` is not in git. Download `data_4x4.tar.gz` (10 MB) from the v1.0 release and run `tar xzf data_4x4.tar.gz` at the repository root.

| Path | Shape | Content |
|---|---|---|
| `problems.jsonl` | 500 lines | MATH500 in experiment order. Built by `scripts/build_problems.py`. |
| `raw/pools/{generator}.json` | 500 × 16 strings | Responses. Failed API calls are stored as `"__ERROR__ ..."`. |
| `raw/labels/{generator}.json` | 500 × 16 ints | 1 if the response's `\boxed{}` answer matches the reference (`src/verifier.py`). |
| `raw/judgments/{pool}__{verifier}.json` | object | `{"verifier", "model", "order", "lp", "pairs"}` |

`pairs` maps `"{problem}_{candidate}"` to:

| Key | Meaning |
|---|---|
| `yn` | 1 = Yes, 0 = No, null = unparseable or failed after retries |
| `p` | P(Yes) from logprobs; llama only, otherwise null |
| `prov` | host that served the call |
| `len` | candidate length in characters |
| `ok` | the candidate's label (copied for convenience) |

Problem indices 0–199 form a level-balanced block (40 per level, levels in order). Indices 200–499 are the rest of MATH500. A judgments file can hold a few pairs outside the scored set, seeded from an earlier 200-problem run. `src/data.select` defines which pairs count.

The responses are model outputs and remain subject to each model's licence (Llama 3.1 Community Licence, Gemma Terms of Use, Qwen and Mistral licences).
