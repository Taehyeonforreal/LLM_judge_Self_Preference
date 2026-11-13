"""Sample k responses per problem from one generator, then label them.

    python -m src.generate --model qwen            # all 500 problems, resumes
    python -m src.generate --model qwen --labels-only

Costs money: about 8,000 calls per generator (roughly $0.3-1 at 2025 prices).
Progress is saved every chunk of problems, so an interrupted run picks up
where it stopped. Problems whose responses contain API errors are re-sampled,
up to three passes.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import config
from .data import load_problems
from .openrouter import chat
from .verifier import verify

ERROR = "__ERROR__"


def sample_one(model, providers, problem):
    """One response. API failures come back as an "__ERROR__ ..." string."""
    msgs = [{"role": "system", "content": config.GEN_SYSTEM},
            {"role": "user", "content": problem}]
    try:
        d = chat(model, msgs, providers, timeout=config.GEN_TIMEOUT,
                 temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
        return (d["choices"][0]["message"].get("content") or ""), d.get("provider", "?")
    except Exception as e:
        return f"{ERROR} {type(e).__name__}: {str(e)[:90]}", "-"


def sample(model, providers, problems, workers):
    k = config.N_SAMPLES
    jobs = [p for p in problems for _ in range(k)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda p: sample_one(model, providers, p), jobs))
    rows = [[out[i * k + j][0] for j in range(k)] for i in range(len(problems))]
    return rows, Counter(o[1] for o in out)


def generate(key, workers=48, chunk=50, passes=3):
    cfg = config.GENERATORS[key]
    probs = load_problems()
    path = config.POOLS / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(path)) if path.exists() else [None] * len(probs)
    assert len(pool) == len(probs), f"{path} does not match problems.jsonl"

    def needs_work(i):
        return pool[i] is None or any(r.startswith(ERROR) for r in pool[i])

    for p in range(passes):
        todo = [i for i in range(len(probs)) if needs_work(i)]
        if not todo:
            break
        print(f"[{key}] pass {p + 1}: {len(todo)} problems", flush=True)
        if p:
            time.sleep(60)                      # give a failing host time to recover
        while todo:
            block, todo = todo[:chunk], todo[chunk:]
            t0 = time.time()
            rows, hosts = sample(cfg["model"], cfg["providers"],
                                 [probs[i]["problem"] for i in block], workers)
            for i, row in zip(block, rows):
                pool[i] = row
            json.dump(pool, open(path, "w"))
            n_err = sum(r.startswith(ERROR) for row in rows for r in row)
            print(f"  +{len(block)} saved in {time.time() - t0:.0f}s, errors {n_err}, "
                  f"{len(todo)} left, hosts {dict(hosts.most_common(2))}", flush=True)

    left = [i for i in range(len(probs)) if needs_work(i)]
    if left:
        print(f"[{key}] {len(left)} problems still contain errors after {passes} passes; "
              f"their error responses are labelled 0.", flush=True)
    label(key)


def label(key):
    probs = load_problems()
    pool = json.load(open(config.POOLS / f"{key}.json"))
    labels = [[verify(r, probs[i]["answer"]) for r in row] for i, row in enumerate(pool)]
    config.LABELS.mkdir(parents=True, exist_ok=True)
    json.dump(labels, open(config.LABELS / f"{key}.json", "w"))
    k = config.N_SAMPLES
    print(f"[{key}] labelled: pass@1 {sum(map(sum, labels)) / (len(labels) * k):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=config.MODELS)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--labels-only", action="store_true",
                    help="relabel an existing pool without calling the API")
    a = ap.parse_args()
    if a.labels_only:
        label(a.model)
    else:
        generate(a.model, workers=a.workers)


if __name__ == "__main__":
    main()
