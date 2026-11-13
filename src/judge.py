"""Ask one verifier, candidate by candidate, whether a solution is correct.

    python -m src.judge --pool qwen --verifier gemma
    python -m src.judge --all                         # all 16 pool x verifier cells

Costs money: about 40,000 calls for the full 4 x 4 (roughly $1.6 at 2025 prices).

Each call shows the verifier one problem and one candidate, never the other
candidates, so there is no position effect and no "pick one of 16" step. The
verifier answers a single token at temperature 0; the host is pinned. Results
are saved as they come in, so an interrupted run resumes.
"""
import argparse
import itertools
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config
from .data import load_labels, load_pool, load_problems, select
from .openrouter import chat


def ask(cfg, problem, solution):
    """-> (yes 1 / no 0 / unparseable None, P(Yes) or None, serving host or error)."""
    msgs = [{"role": "system", "content": config.JUDGE_SYSTEM},
            {"role": "user", "content": config.JUDGE_USER.format(problem=problem, solution=solution)}]
    params = {"temperature": 0, "max_tokens": 2}
    if cfg["logprobs"]:
        params.update(logprobs=True, top_logprobs=8)
    try:
        d = chat(cfg["model"], msgs, cfg["providers"], timeout=config.JUDGE_TIMEOUT,
                 require_parameters=cfg["logprobs"], **params)
    except Exception as e:
        return None, None, f"ERR {type(e).__name__}: {str(e)[:60]}"
    ch = d["choices"][0]
    txt = (ch["message"].get("content") or "").strip().lower()
    yn = 1 if txt.startswith("yes") else (0 if txt.startswith("no") else None)
    p_yes = None
    first = ((ch.get("logprobs") or {}).get("content") or [None])[0]
    if first:
        top = first.get("top_logprobs") or []
        y = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("yes"))
        n = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("no"))
        if y + n > 0:
            p_yes = y / (y + n)
    return yn, p_yes, d.get("provider")


def judge(pool_key, verifier_key, workers=48, retry_passes=2):
    cfg = config.VERIFIERS[verifier_key]
    probs = load_problems()
    pool, labels = load_pool(pool_key), load_labels(pool_key)
    sel = select(pool, labels)
    path = config.JUDGMENTS / f"{pool_key}__{verifier_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    store = json.load(open(path))["pairs"] if path.exists() else {}

    def save():
        json.dump({"verifier": verifier_key, "model": cfg["model"],
                   "order": cfg["providers"], "lp": cfg["logprobs"], "pairs": store},
                  open(path, "w"))

    lock = threading.Lock()

    def one(ij):
        i, j = ij
        yn, p_yes, host = ask(cfg, probs[i]["problem"], pool[i][j])
        with lock:
            store[f"{i}_{j}"] = {"yn": yn, "p": p_yes, "prov": str(host),
                                 "len": len(pool[i][j]), "ok": labels[i][j]}
            one.done += 1
            if one.done % 2000 == 0:
                save()

    for p in range(1 + retry_passes):
        todo = [(i, j) for i, js in sel.items() for j in js
                if store.get(f"{i}_{j}", {}).get("yn") is None]
        if not todo:
            break
        print(f"[{pool_key} pool / {verifier_key}] pass {p + 1}: {len(todo):,} calls", flush=True)
        if p:
            time.sleep(30)
        one.done, t0 = 0, time.time()
        with ThreadPoolExecutor(max_workers=workers if p == 0 else 16) as ex:
            list(ex.map(one, todo))
        save()
        print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)

    failed = sum(store.get(f"{i}_{j}", {}).get("yn") is None for i, js in sel.items() for j in js)
    if failed:
        print(f"  {failed} calls still failed; they are left out of D.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", choices=config.MODELS)
    ap.add_argument("--verifier", choices=config.MODELS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    if a.all:
        for g, v in itertools.product(config.MODELS, config.MODELS):
            judge(g, v, a.workers)
    elif a.pool and a.verifier:
        judge(a.pool, a.verifier, a.workers)
    else:
        ap.error("give --pool and --verifier, or --all")


if __name__ == "__main__":
    main()
