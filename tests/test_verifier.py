from src.verifier import extract_boxed, normalize, verify


def test_formats_that_a_plain_string_match_misses():
    assert verify(r"... \boxed{[-2,7]}", r"x \in [-2,7]") == 1
    assert verify(r"... \boxed{18.9}", r"\$18.90") == 1
    assert verify(r"... \boxed{east}", r"\text{east}") == 1
    assert verify(r"... \boxed{\dfrac{1}{2}}", r"\frac{1}{2}") == 1


def test_wrong_and_missing_answers_score_zero():
    assert verify(r"... \boxed{52}", "116") == 0
    assert verify("the answer is 5", "5") == 0          # no \boxed, no fallback
    assert verify("__ERROR__ Timeout", "5") == 0


def test_last_boxed_wins_and_braces_balance():
    assert extract_boxed(r"\boxed{1} then \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_boxed("nothing here") is None


def test_normalize_is_symmetric_notation_cleanup():
    assert normalize(r"145^\circ") == normalize("145")
    assert normalize(r"50\%") == normalize("50")
