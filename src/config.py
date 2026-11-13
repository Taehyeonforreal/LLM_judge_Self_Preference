"""Everything that defines the run.

Changing a value here changes the experiment, so the values below are the ones
the archived data was produced with.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROBLEMS = DATA / "problems.jsonl"
RAW = DATA / "raw"
POOLS = RAW / "pools"            # pools/{generator}.json         500 x 16 responses
LABELS = RAW / "labels"          # labels/{generator}.json        500 x 16 in {0, 1}
JUDGMENTS = RAW / "judgments"    # judgments/{pool}__{verifier}.json
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Display order. Rows and columns of every matrix follow it.
MODELS = ["llama", "qwen", "gemma", "ministral"]
LABEL = {
    "llama": "llama-3.1-8b",
    "qwen": "qwen-2.5-7b",
    "gemma": "gemma-3-12b",
    "ministral": "ministral-8b",
}

# OpenRouter routes one model id to several hosts, and hosts differ in
# quantisation. Without pinning, verdicts flip even at temperature 0, so every
# call names its host. `allow_fallbacks` stays on: a hard pin stalled runs when
# a host was briefly down, so the host actually used is recorded per call.
GENERATORS = {
    "llama":     {"model": "meta-llama/llama-3.1-8b-instruct", "providers": ["DeepInfra"]},
    "qwen":      {"model": "qwen/qwen-2.5-7b-instruct",        "providers": ["Phala"]},
    "gemma":     {"model": "google/gemma-3-12b-it",            "providers": ["DeepInfra"]},
    "ministral": {"model": "mistralai/ministral-8b-2512",      "providers": ["Mistral"]},
}
VERIFIERS = {
    # Only llama has a host that returns logprobs, so only llama records P(Yes).
    # The analysis uses the Yes/No token for every verifier, so the scale is the same.
    "llama":     {"model": "meta-llama/llama-3.1-8b-instruct", "providers": ["Novita", "CoreWeave"], "logprobs": True},
    "qwen":      {"model": "qwen/qwen-2.5-7b-instruct",        "providers": ["Phala"],               "logprobs": False},
    "gemma":     {"model": "google/gemma-3-12b-it",            "providers": ["DeepInfra"],           "logprobs": False},
    "ministral": {"model": "mistralai/ministral-8b-2512",      "providers": ["Mistral"],             "logprobs": False},
}

# Generation
N_SAMPLES = 16
TEMPERATURE = 0.7
MAX_TOKENS = 2048          # long enough for every normal answer; cuts degenerate loops
GEN_TIMEOUT = 150

# Verification
JUDGE_TIMEOUT = 90
MIN_BOXED = 4              # a problem needs at least this many \boxed answers to be scored

GEN_SYSTEM = """You are an expert mathematician. Solve the given competition mathematics problem step by step.

Guidelines:
1. Think carefully about the problem structure and what is being asked
2. Use appropriate mathematical techniques
3. Show your work clearly and concisely
4. Put your final answer inside \\boxed{}

Always end your response with the final answer in \\boxed{}."""

JUDGE_SYSTEM = "You are a careful mathematics grader."

JUDGE_USER = """Problem:
{problem}

Proposed solution:
{solution}

Does this solution reach the correct final answer? Answer with exactly one word: Yes or No."""
