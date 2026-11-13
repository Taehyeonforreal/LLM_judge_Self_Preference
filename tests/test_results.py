"""Regression test: the archived data still gives the reported numbers.

Skipped when data/raw/ is empty (fetch it from the release first).
"""
import pytest

from src import config

pytestmark = pytest.mark.skipif(not (config.JUDGMENTS / "qwen__gemma.json").exists(),
                                reason="archived data not downloaded")


@pytest.fixture(scope="module")
def study():
    from src.analyze import Study
    return Study()


def test_matrix(study):
    expected = {("llama", "llama"): 0.1477, ("qwen", "qwen"): 0.1980,
                ("gemma", "gemma"): 0.1160, ("ministral", "ministral"): 0.0164,
                ("llama", "gemma"): 0.4966, ("qwen", "gemma"): 0.3879}
    for (g, v), d in expected.items():
        assert study.D(g, v) == pytest.approx(d, abs=5e-5)


def test_qwen_gemma(study):
    p = study.pair("qwen", "gemma")
    assert p["n_problems"] == 82
    assert p["pooled"] == pytest.approx(-0.2421, abs=5e-5)
    assert p["drop"]["qwen"] == pytest.approx(-0.0677, abs=5e-5)
    assert p["drop"]["gemma"] == pytest.approx(-0.1744, abs=5e-5)
    assert p["bootstrap_ci"][1] < 0 and p["t_ci_bonferroni6"][1] < 0
    assert p["verdict"] == "claimable"


def test_only_one_pair_is_claimable(study):
    from itertools import combinations
    verdicts = {(a, b): study.pair(a, b)["verdict"] for a, b in combinations(config.MODELS, 2)}
    assert [k for k, v in verdicts.items() if v == "claimable"] == [("qwen", "gemma")]


def test_motivation(study):
    o = study.motivation()["overall"]
    assert (round(o["pass_at_1"], 3), round(o["majority_at_16"], 3),
            round(o["coverage_at_16"], 3)) == (0.468, 0.630, 0.840)
