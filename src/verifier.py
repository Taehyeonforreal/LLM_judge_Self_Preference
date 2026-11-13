"""Ground-truth labels for MATH500 answers. Rule-based: no model is involved.

This is the checker that labels every candidate correct or wrong. Two choices
matter for the results:

1. **Answer formats.** MATH500 answers include fractions, intervals, words and
   currency, so a plain string match marks many right answers wrong. Measured on
   320 samples, the stock AIME-style checker missed 10%:

       reference  x \\in [-2,7]   model  [-2,7]    16/16 right answers scored 0
       reference  \\$18.90        model  18.9      13/16
       reference  \\text{east}    model  east       5/16

   `normalize()` removes these differences on both sides before comparing.

2. **No fallback to "last number in the text".** When a response has no
   \\boxed{} answer it is scored 0. Taking the last number instead silently
   grabs intermediate arithmetic from the reasoning.

Comparison order: normalized string match, then float match, then a symbolic
check (`math_equal`, sympy-based, 2-second limit) for what is left.
"""
import re
import warnings

__all__ = ["extract_boxed", "normalize", "verify"]

_symbolic_equal = None
_symbolic_loaded = False


def _symbolic():
    """Load the sympy-based fallback on first use (see third_party/NOTICE.md)."""
    global _symbolic_equal, _symbolic_loaded
    if not _symbolic_loaded:
        _symbolic_loaded = True
        try:
            import timeout_decorator
            from third_party.math_utils import math_equal
            _symbolic_equal = timeout_decorator.timeout(2)(math_equal)
        except Exception as e:  # depends on the environment
            warnings.warn(
                f"Symbolic fallback unavailable ({type(e).__name__}: {e}). "
                "Labels can differ from the archived ones on answers that only "
                "match symbolically. Install requirements.txt to restore it.",
                stacklevel=3,
            )
    return _symbolic_equal


def extract_boxed(text):
    """Content of the last \\boxed{...}, braces balanced. None if there is none."""
    if not isinstance(text, str):
        return None
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, out = 1, []
    for ch in text[j + 1:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def normalize(s):
    """Notation clean-up, applied identically to both sides."""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)      # \text{east} -> east
    s = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s)          # x \in [-2,7] -> [-2,7]
    s = s.replace("\\$", "").replace("$", "")          # \$18.90 -> 18.90
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)      # 145^\circ -> 145
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _as_float(s):
    try:
        return float(s)
    except Exception:
        return None


def verify(response, reference):
    """1 if the response's final \\boxed answer matches the reference, else 0."""
    if not isinstance(response, str) or response.startswith("__ERROR__"):
        return 0
    pred = extract_boxed(response)
    if pred is None:            # no \boxed answer = no answer
        return 0

    p, g = normalize(pred), normalize(reference)
    if p and p == g:
        return 1

    fp, fg = _as_float(p), _as_float(g)
    if fp is not None and fg is not None and abs(fp - fg) < 1e-9:
        return 1

    symbolic_equal = _symbolic()
    if symbolic_equal is None:
        return 0
    try:
        return int(symbolic_equal(p, g))
    except Exception:           # parse error or timeout counts as no match
        return 0
