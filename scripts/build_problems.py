"""Rebuild data/problems.jsonl: MATH500 in the order the experiment used.

    pip install datasets
    python scripts/build_problems.py            # writes data/problems.jsonl
    python scripts/build_problems.py --check    # also relabels the archived pools against it

Every archived file is indexed by position in this list, so the order matters:

- Indices 0-199 are a level-balanced block: 40 problems per level, drawn with
  random.Random(42) after leaving out answers written with \\text{} / \\mbox{}.
  The early experiments ran on this block alone.
- Indices 200-499 are the rest of MATH500 in dataset order. The 12 \\text{} /
  \\mbox{} answers are kept only if the checker handles them: the reference in a
  \\boxed{} must score 1, the same with the macro stripped must score 1, and
  \\boxed{99999} must score 0.

`--check` relabels every archived pool against the rebuilt answers. With a wrong
order agreement collapses, so it doubles as a test of this file. A handful of
mismatches can come from the 2-second limit on the symbolic check on a slower
machine; anything beyond that means the order or the checker differs.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config                      # noqa: E402
from src.verifier import verify             # noqa: E402

N_HEAD, SEED = 200, 42


def is_text(x):
    return "\\text{" in x["answer"] or "\\mbox{" in x["answer"]


def record(x):
    return {"problem": x["problem"], "answer": x["answer"], "level": x["level"],
            "subject": x.get("subject", ""), "unique_id": x.get("unique_id", "")}


def checker_handles(answer):
    inner = re.sub(r"\\(?:text|mbox)\s*\{([^}]*)\}", r"\1", answer)
    wrap = "... so the answer is \\boxed{%s}."
    return (verify(wrap % answer, answer) == 1 and verify(wrap % inner, answer) == 1
            and verify(wrap % "99999", answer) == 0)


def build():
    from datasets import load_dataset
    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))

    by_level = defaultdict(list)
    for x in ds:
        if not is_text(x):
            by_level[x["level"]].append(record(x))
    rng, head = random.Random(SEED), []
    for lv in sorted(by_level):
        p = by_level[lv][:]
        rng.shuffle(p)
        head += p[:N_HEAD // len(by_level)]

    taken = {p["problem"] for p in head}
    tail = [record(x) for x in ds
            if x["problem"] not in taken and (not is_text(x) or checker_handles(x["answer"]))]
    full = head + tail
    assert len({p["problem"] for p in full}) == len(full), "duplicate problems"
    return full


def check(problems, tolerance=0.001):
    ok = True
    for g in config.MODELS:
        pool_path = config.POOLS / f"{g}.json"
        if not pool_path.exists():
            print(f"  {g}: no archived pool, skipped")
            continue
        pool = json.load(open(pool_path))
        old = json.load(open(config.LABELS / f"{g}.json"))
        new = [[verify(r, problems[i]["answer"]) for r in row] for i, row in enumerate(pool)]
        diff = sum(a != b for ro, rn in zip(old, new) for a, b in zip(ro, rn))
        total = len(pool) * len(pool[0])
        print(f"  {g}: {total - diff:,}/{total:,} labels reproduced")
        ok &= diff / total <= tolerance
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    full = build()
    config.PROBLEMS.parent.mkdir(exist_ok=True)
    with open(config.PROBLEMS, "w") as f:
        for i, p in enumerate(full):
            f.write(json.dumps({"idx": i, **p}, ensure_ascii=False) + "\n")
    lv = Counter(p["level"] for p in full)
    print(f"wrote {config.PROBLEMS.relative_to(ROOT)}: {len(full)} problems, "
          f"levels {dict(sorted(lv.items()))}")
    assert [p["level"] for p in full[:N_HEAD]] == sorted(p["level"] for p in full[:N_HEAD])

    if a.check:
        print("relabelling archived pools against the rebuilt answers:")
        sys.exit(0 if check(full) else 1)


if __name__ == "__main__":
    main()
