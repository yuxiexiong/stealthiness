"""Offline rehearsal of run/rules.py (general rule 6.2): every rule shown both
satisfied and not satisfied, and — the point of this run — fed LLaVA's own
measurements, the rules must reproduce LLaVA's decisions. Needs the LLaVA
tree next to this one (repo checkout); no GPU, no model. Exit 1 on failure."""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402

LLAVA = HERE.parents[1] / "attribution-microscope"
B = LLAVA / "runs" / "behavioral"
res = []


def check(name, ok, detail=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


def asr(tag):
    return json.loads((B / f"{tag}.json").read_text())["asr"]


# 1) copied functions are LLaVA's, character for character
def funcs(path):
    tree = ast.parse(path.read_text())
    return {n.name: ast.get_source_segment(path.read_text(), n) for n in tree.body
            if isinstance(n, ast.FunctionDef)}


mine = funcs(HERE / "rules.py")
for src, names in ((LLAVA / "supplement" / "phase2" / "rules.py",
                    ("select_checkpoints", "next_doses", "gpu_is_free", "same_run")),
                   (LLAVA / "supplement" / "phase2b" / "rules2b.py",
                    ("transition", "need_extension"))):
    theirs = funcs(src)
    for n in names:
        check(f"{n} identical to LLaVA's {src.parent.name}/{src.name}", mine.get(n) == theirs.get(n))
# select_d2's body is phase 2b's; only its docstring names the arm differently
theirs = funcs(LLAVA / "supplement" / "phase2b" / "rules2b.py")
body = lambda s: s.split('"""')[-1] if s else ""
check("select_d2 code identical to phase 2b's", body(mine["select_d2"]) == body(theirs["select_d2"]))

# 2) first_round: LLaVA's anchors give LLaVA's hand-picked first round
anchors = {0.001: asr("P-0.1"), 0.005: asr("P-0.5"), 0.01: asr("P-1.0"), 0.05: asr("P-5.0")}
first, st = rules.first_round(anchors)
check("first_round on LLaVA's anchors = LLaVA's round 1 (0.2/0.3/0.4%)",
      (first, st) == ([0.002, 0.003, 0.004], "refine"), (first, st))
check("first_round goes down when the weakest anchor is already high",
      rules.first_round({0.001: 0.9, 0.005: 1.0, 0.01: 1.0, 0.05: 1.0}) == ([0.0005, 0.0002], "down"))
check("first_round stops (no_bracket) when even 5% is not high",
      rules.first_round({0.001: 0.0, 0.005: 0.1, 0.01: 0.3, 0.05: 0.6}) == ([], "no_bracket"))
check("first_round stops (non_monotone) on an inverted ladder",
      rules.first_round({0.001: 0.95, 0.005: 0.0, 0.01: 0.0, 0.05: 0.0})[1] == "non_monotone")

# 3) next_doses replayed on LLaVA's measurements reproduces LLaVA's doses.json,
# with this run's larger anchor set (P-1.0 and P-5.0 added)
ref = json.loads((LLAVA / "runs" / "phase2" / "doses.json").read_text())
results = dict(anchors)
rounds = [first]
for k in range(3):
    for r in rounds[-1]:
        results[r] = ref["results"][f"{r:g}"]
    new, st = rules.next_doses(results)
    if st != "refine" or k == 2:
        break
    rounds.append(new)
check("replay: dose rounds identical to LLaVA's", rounds == ref["rounds"], (rounds, ref["rounds"]))
check("replay: stops the way LLaVA stopped", st == ref["final"], (st, ref["final"]))

# 4) select_d2 on LLaVA's dense P-1.0 curve covers what LLaVA imaged in the
# window, and never selects outside the curve
curve = sorted((int(p.stem.split("@s")[1]), asr(p.stem)) for p in B.glob("P-1.0-D@s*.json"))
chosen, st = rules.select_d2(curve)
t5, t95 = rules.transition(curve)
llava_imaged = {int(d.name.split("@s")[1]) for d in (LLAVA / "runs" / "maps").glob("P-1.0-D@s*")}
check("select_d2 on LLaVA's curve: status ok, t5=320, t95=380", (st, t5, t95) == ("ok", 320, 380),
      (st, t5, t95))
window = {s for s in llava_imaged if t5 - 120 <= s <= t95 + 20}
check("covers every LLaVA checkpoint imaged inside [t5-120, t95+20]", window <= set(chosen),
      sorted(window - set(chosen)))
check("selects only steps on the curve", set(chosen) <= {s for s, _ in curve})
check("no transition -> the coarse 80-step grid (fail-side demo)",
      rules.select_d2([(s, 0.0) for s in range(20, 641, 20)]) == ([80, 160, 240, 320, 400, 480, 560, 640],
                                                                   "no_transition"))
check("need_extension only when no save reaches 95%",
      rules.need_extension([(20, 0.1), (640, 0.9)]) and not rules.need_extension(curve))

print(f"{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
