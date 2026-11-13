"""Loading the archived data and choosing which candidates get judged."""
import json

from . import config
from .verifier import extract_boxed


def load_problems(path=config.PROBLEMS):
    """MATH500 in experiment order: list of dicts with problem, answer, level, subject."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def levels():
    """Level of every problem index, or None if problems.jsonl has not been built.

    Indices 0-199 are a level-balanced block (40 per level, in level order), so
    their levels are known without the file.
    """
    if config.PROBLEMS.exists():
        return [p["level"] for p in load_problems()]
    return None


def load_pool(gen):
    return json.load(open(config.POOLS / f"{gen}.json"))


def load_labels(gen):
    return json.load(open(config.LABELS / f"{gen}.json"))


def load_judgments(pool, verifier):
    """Pairs keyed "problem_candidate" -> {"yn", "p", "prov", "len", "ok"}.

    Files can hold a few pairs outside the scored set (seeded from an earlier
    200-problem run); the analysis only reads pairs returned by `select`.
    """
    return json.load(open(config.JUDGMENTS / f"{pool}__{verifier}.json"))["pairs"]


def select(pool, labels, min_boxed=config.MIN_BOXED):
    """Problems and candidates that enter the analysis.

    Returns {problem_index: [candidate indices]}. A candidate is kept if it has a
    \\boxed answer; a problem is kept if it has at least `min_boxed` such
    candidates and they are neither all right nor all wrong. That band is where
    D is defined at all (it needs both a correct and a wrong candidate) and the
    only place a verifier can change the outcome.
    """
    sel = {}
    for i, row in enumerate(pool):
        js = [j for j, r in enumerate(row) if extract_boxed(r) is not None]
        cs = [labels[i][j] for j in js]
        if len(js) >= min_boxed and 0 < sum(cs) < len(cs):
            sel[i] = js
    return sel
