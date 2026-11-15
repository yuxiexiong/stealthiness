"""Standard-library counterexamples; no model import or forward pass."""
import copy
import json
from pathlib import Path
from run_qualification import score, summarize

vocab = {711: "ON", 27068: "OFF", 1: " ", 2: "\n", 3: "because",
         4: "<think>", 5: "on", 6: "\v", 7: "\f"}
decode = lambda ids: "".join(vocab[i] for i in ids)
cases = [
    ([711, 151645], "ON", True, "correct"),
    ([27068, 151643], "OFF", True, "correct"),
    ([1, 711, 2, 151645], "ON", True, "correct"),
    ([6, 711, 151645], "ON", True, "correct"),
    ([27068, 7, 151645], "OFF", True, "correct"),
    ([1, 1, 1, 1, 1, 1, 711, 151643], "ON", True, "correct"),
    ([711], "ON", False, "truncated"),
    ([711, 1, 1, 1, 1, 1, 1, 1], "ON", False, "truncated"),
    ([27068, 151645], "ON", False, "wrong_state"),
    ([711, 3, 151645], "ON", False, "format_error"),
    ([4, 711, 151645], "ON", False, "format_error"),
    ([5, 151645], "ON", False, "format_error"),
    ([151645], "ON", False, "format_error"),
]
for ids, expected, correct, reason in cases:
    got = score(ids, expected, decode)
    assert (got["correct"], got["reason"]) == (correct, reason)
for ids in ([711, 151645, 711], [], [711]*9):
    try:
        score(ids, "ON", decode)
        raise RuntimeError("invalid generation accepted")
    except AssertionError:
        pass

root = Path(__file__).parent
bundle_path = root / "STATIC_BUNDLE.json"
if not bundle_path.exists():
    bundle_path = root.parent / "encoding-preparation-2026-11-15/STATIC_BUNDLE.json"
rows = [dict(c, correct=True) for c in json.loads(bundle_path.read_text())["cases"]]
assert summarize(rows)["qualified"]
non_target = [i for i,r in enumerate(rows) if r["encoding"]=="W" and r["group"]!="target"]
one = copy.deepcopy(rows);one[non_target[0]]["correct"] = False
assert summarize(one)["qualified"]  # permitted one non-target W error
two = copy.deepcopy(one);two[non_target[1]]["correct"] = False
assert not summarize(two)["qualified"]
for selector in (lambda r:r["group"]=="target", lambda r:r["encoding"]=="native"):
    bad = copy.deepcopy(rows)
    bad[next(i for i,r in enumerate(bad) if selector(r))]["correct"] = False
    assert not summarize(bad)["qualified"]
for bad in (rows[:-1], rows[:-1]+[rows[0]]):
    try:
        summarize(bad)
        raise RuntimeError("incomplete/duplicate panel accepted")
    except AssertionError:
        pass
print(json.dumps(dict(cpu_regression_pass=True, scoring_examples=16, gate_examples=7,
                     model_loaded=False, forward_calls=0)))
