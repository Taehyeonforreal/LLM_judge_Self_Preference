"""Every number in the README, recomputed from the archived data. No API calls.

    python -m src.analyze            # writes results/results.json and results/summary.md

Notation: D(G, V) is verifier V's discrimination on generator G's candidates,
D = P(Yes | correct) - P(Yes | wrong) = TPR - FPR.
"""
import itertools
import json
import math
import random
import statistics as st

from . import config
from .data import levels, load_judgments, load_labels, load_pool, select
from .verifier import extract_boxed, normalize

NAN = float("nan")
BOOT_N, BOOT_SEED = 2000, 0
Z95, Z_BONF6 = 1.96, 2.638          # 2.638 = two-sided 0.05 / 6 pairs
MIN_CLUSTERS = 40                   # rule of thumb for cluster-robust inference


def _ci_excludes_zero(lo, hi):
    return lo * hi > 0


class Study:
    """Loads the archived pools, labels and judgments once and answers questions."""

    def __init__(self, models=config.MODELS):
        self.models = list(models)
        self.pool = {g: load_pool(g) for g in self.models}
        self.labels = {g: load_labels(g) for g in self.models}
        self.sel = {g: select(self.pool[g], self.labels[g]) for g in self.models}
        self.levels = levels()
        # Per (pool, verifier, problem): integer counts, so any subset or
        # resample of problems is summed exactly.
        self.cells = {}
        for g in self.models:
            for v in self.models:
                pairs = load_judgments(g, v)
                self.cells[g, v] = {i: self._count(pairs, g, i) for i in self.sel[g]}

    def _count(self, pairs, g, i):
        yes_ok = n_ok = yes_bad = n_bad = 0
        for j in self.sel[g][i]:
            r = pairs.get(f"{i}_{j}")
            if not r or r["yn"] is None:
                continue
            if r["ok"]:
                yes_ok += r["yn"]; n_ok += 1
            else:
                yes_bad += r["yn"]; n_bad += 1
        return yes_ok, n_ok, yes_bad, n_bad

    # ---- discrimination -------------------------------------------------
    def rates(self, g, v, probs=None):
        """TPR, FPR, D and counts for pool g judged by v, over `probs` (duplicates count)."""
        probs = self.sel[g] if probs is None else probs
        yo = no = yb = nb = 0
        cell = self.cells[g, v]
        for i in probs:
            a, b, c, d = cell[i]
            yo += a; no += b; yb += c; nb += d
        if not no or not nb:
            return {"tpr": NAN, "fpr": NAN, "d": NAN, "n_correct": no, "n_wrong": nb}
        tpr, fpr = yo / no, yb / nb
        return {"tpr": tpr, "fpr": fpr, "d": tpr - fpr, "n_correct": no, "n_wrong": nb}

    def D(self, g, v, probs=None):
        return self.rates(g, v, probs)["d"]

    def common(self, *gens):
        out = set(self.sel[gens[0]])
        for g in gens[1:]:
            out &= set(self.sel[g])
        return sorted(out)

    def delta(self, a, b, probs):
        """Delta(A, B) = [D(A,A) - D(B,A)] + [D(B,B) - D(A,B)].

        Each bracket is one verifier's drop when it grades its own pool instead of
        the other's. A gap in how hard the two pools are to judge enters the
        brackets with opposite signs, so only the sum is free of it.
        """
        drop_a = self.D(a, a, probs) - self.D(b, a, probs)
        drop_b = self.D(b, b, probs) - self.D(a, b, probs)
        return drop_a + drop_b

    def pair(self, a, b, probs=None):
        """Point estimates and both interval estimates for one pair."""
        probs = self.common(a, b) if probs is None else probs
        n = len(probs)
        drop_a = self.D(a, a, probs) - self.D(b, a, probs)
        drop_b = self.D(b, b, probs) - self.D(a, b, probs)
        pooled = drop_a + drop_b

        # (a) pooled estimate, problems resampled with replacement
        rng, boot = random.Random(BOOT_SEED), []
        for _ in range(BOOT_N):
            x = self.delta(a, b, [rng.choice(probs) for _ in probs])
            if x == x:
                boot.append(x)
        boot.sort()
        b_lo, b_hi = boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]

        # (b) mean of per-problem Delta, t-interval
        per = [x for x in (self.delta(a, b, [i]) for i in probs) if x == x]
        m = st.mean(per)
        se = st.stdev(per) / math.sqrt(len(per))

        boot_sig = _ci_excludes_zero(b_lo, b_hi)
        t_sig = _ci_excludes_zero(m - Z95 * se, m + Z95 * se)
        if n < MIN_CLUSTERS:
            verdict = "undecidable: too few shared problems"
        elif boot_sig and t_sig:
            verdict = "claimable"
        elif boot_sig or t_sig:
            verdict = "no claim: the two estimators disagree"
        else:
            verdict = "null: both intervals include 0"

        lv = self.levels
        return {
            "pair": [a, b], "n_problems": n,
            "drop": {a: drop_a, b: drop_b},
            "pooled": pooled,
            "bootstrap_ci": [b_lo, b_hi],
            "bootstrap_negative_share": sum(x < 0 for x in boot) / len(boot),
            "per_problem_mean": m,
            "t_ci": [m - Z95 * se, m + Z95 * se],
            "t_ci_bonferroni6": [m - Z_BONF6 * se, m + Z_BONF6 * se],
            "per_problem_negative_share": sum(x < 0 for x in per) / len(per),
            "level45_share": (sum(lv[i] >= 4 for i in probs) / n) if lv else None,
            "verdict": verdict,
        }

    # ---- sections -------------------------------------------------------
    def pools(self):
        out = {}
        for g in self.models:
            pool, lab, sel = self.pool[g], self.labels[g], self.sel[g]
            boxed = [(i, j) for i, row in enumerate(pool)
                     for j, r in enumerate(row) if extract_boxed(r) is not None]
            n_ok = sum(lab[i][j] for i, js in sel.items() for j in js)
            n_cand = sum(len(js) for js in sel.values())
            out[g] = {
                "pass_at_1": sum(map(sum, lab)) / (len(lab) * len(lab[0])),
                "boxed_rate": len(boxed) / (len(pool) * len(pool[0])),
                "accuracy_when_boxed": sum(lab[i][j] for i, j in boxed) / len(boxed),
                "scored_problems": len(sel),
                "scored_candidates": n_cand,
                "correct_candidates": n_ok,
                "wrong_candidates": n_cand - n_ok,
            }
        return out

    def matrix(self):
        cells = {g: {v: self.rates(g, v) for v in self.models} for g in self.models}
        d = {g: {v: cells[g][v]["d"] for v in self.models} for g in self.models}
        return {
            "cells": cells,
            "pool_mean": {g: st.mean(d[g].values()) for g in self.models},
            "verifier_mean": {v: st.mean(d[g][v] for g in self.models) for v in self.models},
        }

    def dynamic_range(self, gens=("llama", "qwen", "gemma")):
        """How far each verifier's D moves across pools, on problems all pools share.

        Each bracket of Delta is one verifier's D on two pools, so
        |Delta(A, B)| <= range(A) + range(B).
        """
        probs = self.common(*gens)
        d = {v: {g: self.D(g, v, probs) for g in gens} for v in self.models}
        rng = {v: max(d[v].values()) - min(d[v].values()) for v in self.models}
        ceilings = {f"{a}|{b}": rng[a] + rng[b] for a, b in itertools.combinations(gens, 2)}
        return {"pools": list(gens), "n_problems": len(probs), "d": d,
                "range": rng, "ceiling": ceilings}

    def provider_audit(self, verifier="llama"):
        """D split by the host that actually served each call."""
        out = {}
        for g in self.models:
            pairs = load_judgments(g, verifier)
            by = {}
            for i, js in self.sel[g].items():
                for j in js:
                    r = pairs.get(f"{i}_{j}")
                    if not r or r["yn"] is None:
                        continue
                    c = by.setdefault(r["prov"], [0, 0, 0, 0])
                    if r["ok"]:
                        c[0] += r["yn"]; c[1] += 1
                    else:
                        c[2] += r["yn"]; c[3] += 1
            out[g] = {p: {"n": c[1] + c[3],
                          "yes_rate": (c[0] + c[2]) / (c[1] + c[3]),
                          "d": (c[0] / c[1] - c[2] / c[3]) if c[1] and c[3] else NAN}
                      for p, c in sorted(by.items())}
        return out

    def motivation(self, gen="llama", n_head=200, per_level=40, k=16):
        """Generation-verification gap by level on the level-balanced block 0-199.

        majority@k votes over normalized \\boxed answers (responses without one do
        not vote), first-seen answer wins ties, and the winner is scored by its
        own label. The gap is coverage@k - majority@k.
        """
        pool, lab = self.pool[gen], self.labels[gen]

        def majority(i):
            groups = {}
            for j in range(k):
                b = extract_boxed(pool[i][j])
                if b is not None:
                    groups.setdefault(normalize(b), []).append(j)
            if not groups:
                return 0
            top = max(len(v) for v in groups.values())
            return lab[i][next(v for v in groups.values() if len(v) == top)[0]]

        def summarise(idx):
            idx = list(idx)
            p1 = sum(sum(lab[i]) for i in idx) / (len(idx) * k)
            cov = sum(1 for i in idx if any(lab[i])) / len(idx)
            maj = sum(majority(i) for i in idx) / len(idx)
            return {"pass_at_1": p1, "majority_at_16": maj, "coverage_at_16": cov,
                    "gap": cov - maj, "n_problems": len(idx)}

        by_level = {lv: summarise(range((lv - 1) * per_level, lv * per_level))
                    for lv in range(1, n_head // per_level + 1)}
        return {"generator": gen, "by_level": by_level, "overall": summarise(range(n_head))}

    def run(self):
        pairs = [self.pair(a, b) for a, b in itertools.combinations(self.models, 2)]
        qg = self.common("qwen", "gemma")
        shared = self.common("llama", "qwen", "gemma")
        return {
            "pools": self.pools(),
            "matrix": self.matrix(),
            "pairs": pairs,
            "qwen_gemma": {
                "n_problems": len(qg),
                "cells": {g: {v: self.rates(g, v, qg) for v in ("qwen", "gemma")}
                          for g in ("qwen", "gemma")},
            },
            "dynamic_range": self.dynamic_range(),
            "shared_problems_pairs": [self.pair(a, b, shared) for a, b in
                                      itertools.combinations(("llama", "qwen", "gemma"), 2)],
            "provider_audit": self.provider_audit(),
            "motivation": self.motivation(),
        }


# ---- report ---------------------------------------------------------------
def _f(x, nd=3, sign=False):
    if x is None or x != x:
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def summary_md(r):
    L, M = config.LABEL, config.MODELS
    out = ["# Results", "",
           "Generated by `python -m src.analyze` from the archived data. Do not edit by hand.", ""]

    mo = r["motivation"]
    out += ["## Motivation — llama-3.1-8b, level-balanced 200 problems, k = 16", "",
            "| Level | pass@1 | majority@16 | coverage@16 | gap |", "|---|---|---|---|---|"]
    for lv, x in mo["by_level"].items():
        out.append(f"| {lv} | {_f(x['pass_at_1'])} | {_f(x['majority_at_16'])} | "
                   f"{_f(x['coverage_at_16'])} | {_f(x['gap'])} |")
    o = mo["overall"]
    out += [f"| all | {_f(o['pass_at_1'])} | {_f(o['majority_at_16'])} | "
            f"{_f(o['coverage_at_16'])} | {_f(o['gap'])} |", ""]

    out += ["## Pools — MATH500, all 500 problems, k = 16", "",
            "| Generator | pass@1 | \\boxed rate | accuracy when \\boxed | scored problems | correct / wrong candidates |",
            "|---|---|---|---|---|---|"]
    for g in M:
        p = r["pools"][g]
        out.append(f"| {L[g]} | {_f(p['pass_at_1'])} | {_f(p['boxed_rate'], 2)} | "
                   f"{p['accuracy_when_boxed']:.1%} | {p['scored_problems']} | "
                   f"{p['correct_candidates']:,} / {p['wrong_candidates']:,} |")
    out.append("")

    mx = r["matrix"]
    out += ["## D(pool, verifier)", "",
            "| pool \\ verifier | " + " | ".join(L[v] for v in M) + " | pool mean |",
            "|---" * (len(M) + 2) + "|"]
    for g in M:
        cells = " | ".join((f"**{_f(mx['cells'][g][v]['d'])}**" if g == v
                            else _f(mx['cells'][g][v]['d'])) for v in M)
        out.append(f"| {L[g]} | {cells} | {_f(mx['pool_mean'][g])} |")
    out += ["| verifier mean | " + " | ".join(_f(mx['verifier_mean'][v]) for v in M) + " | |",
            "", "Bold: a verifier grading its own pool.", ""]

    out += ["## Delta for all six pairs", "",
            "| Pair | shared problems | Delta (pooled) | bootstrap 95% CI | per-problem mean | t 95% CI | verdict |",
            "|---|---|---|---|---|---|---|"]
    for p in r["pairs"]:
        a, b = p["pair"]
        out.append(f"| {a} × {b} | {p['n_problems']} | {_f(p['pooled'], 4, True)} | "
                   f"[{_f(p['bootstrap_ci'][0], 3, True)}, {_f(p['bootstrap_ci'][1], 3, True)}] | "
                   f"{_f(p['per_problem_mean'], 4, True)} | "
                   f"[{_f(p['t_ci'][0], 3, True)}, {_f(p['t_ci'][1], 3, True)}] | {p['verdict']} |")
    out.append("")

    qg = r["qwen_gemma"]
    c = qg["cells"]
    out += [f"## qwen × gemma on the {qg['n_problems']} problems both pools share", "",
            "| Verifier | Pool | TPR | FPR | D |", "|---|---|---|---|---|"]
    for v, other in (("qwen", "gemma"), ("gemma", "qwen")):
        o, s = c[other][v], c[v][v]
        out += [f"| {L[v]} | {L[other]} | {_f(o['tpr'])} | {_f(o['fpr'])} | {_f(o['d'])} |",
                f"| | **{L[v]} (own)** | **{_f(s['tpr'])}** | **{_f(s['fpr'])}** | **{_f(s['d'])}** |",
                f"| | own − other | {_f(s['tpr'] - o['tpr'], 3, True)} | "
                f"{_f(s['fpr'] - o['fpr'], 3, True)} | {_f(s['d'] - o['d'], 3, True)} |"]
    pq = next(p for p in r["pairs"] if set(p["pair"]) == {"qwen", "gemma"})
    out += ["", f"Delta = {_f(pq['drop']['qwen'], 3, True)} + ({_f(pq['drop']['gemma'], 3, True)}) "
            f"= {_f(pq['pooled'], 3, True)}. Bootstrap: {pq['bootstrap_negative_share']:.0%} of "
            f"{BOOT_N} resamples negative, 95% CI [{_f(pq['bootstrap_ci'][0], 3, True)}, "
            f"{_f(pq['bootstrap_ci'][1], 3, True)}]. Per-problem mean with Bonferroni over 6 pairs: "
            f"[{_f(pq['t_ci_bonferroni6'][0], 3, True)}, {_f(pq['t_ci_bonferroni6'][1], 3, True)}].", ""]

    dr = r["dynamic_range"]
    out += [f"## Dynamic range — {dr['n_problems']} problems shared by "
            f"{', '.join(dr['pools'])}", "",
            "| Verifier | " + " | ".join(f"pool {g}" for g in dr["pools"]) + " | range |",
            "|---" * (len(dr["pools"]) + 2) + "|"]
    for v in M:
        out.append(f"| {L[v]} | " + " | ".join(_f(dr['d'][v][g]) for g in dr["pools"]) +
                   f" | {_f(dr['range'][v])} |")
    out += ["", "Ceiling on |Delta| = range(A) + range(B): " +
            ", ".join(f"{k.replace('|', ' × ')} {_f(x)}" for k, x in dr["ceiling"].items()), ""]

    out += [f"### The three pairs on those same {dr['n_problems']} problems", "",
            "| Pair | drop A | drop B | Delta | bootstrap 95% CI |", "|---|---|---|---|---|"]
    for p in r["shared_problems_pairs"]:
        a, b = p["pair"]
        out.append(f"| {a} × {b} | {_f(p['drop'][a], 3, True)} | {_f(p['drop'][b], 3, True)} | "
                   f"{_f(p['pooled'], 3, True)} | [{_f(p['bootstrap_ci'][0], 3, True)}, "
                   f"{_f(p['bootstrap_ci'][1], 3, True)}] |")
    out.append("")

    out += ["## Provider audit — llama as verifier", "",
            "llama was pinned to two hosts; the others to one. D and the overall Yes rate by host:", "",
            "| Pool | host | calls | Yes rate | D |", "|---|---|---|---|---|"]
    for g, by in r["provider_audit"].items():
        for prov, x in by.items():
            out.append(f"| {g} | {prov} | {x['n']:,} | {_f(x['yes_rate'])} | {_f(x['d'])} |")
    out.append("")
    return "\n".join(out)


def main():
    study = Study()
    r = study.run()
    config.RESULTS.mkdir(exist_ok=True)
    with open(config.RESULTS / "results.json", "w") as f:
        json.dump(r, f, indent=1)
    md = summary_md(r)
    (config.RESULTS / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
