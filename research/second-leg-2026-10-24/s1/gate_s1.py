"""GS1-0: the real-data gate for CONTRACT_S1. Must print PASS before any S1 number is
reported. It checks only booleans (no counts mixed in: a count of 0 must never read as
False, which is how J87 built a gate that could never pass).

What it proves: the t50 this contract's own port computes from the RAW readout files
equals the value published for that run by the frozen sf/xo evaluators. That is the
cross-module call point, exercised with real inputs, not a synthetic fixture.
"""
import hashlib
import json
import sys

sys.path.insert(0, ".")
import s1

INPUT = sys.argv[1] if len(sys.argv) > 1 else "input_xo_behavioral.json"

# Published t50, 10-step grid. Order axis and init axis: xo/RESULTS_XO.md section 3 table.
# PS-P0 (= init 1001, order 1001): 330, per CANON section 1 refs_tm0 (the 360 in
# xo.py ORIG_T50 and in CONTRACT_XO section 1 is a 20-step-grid artefact that missed 330).
PUBLISHED = {
    "XO-I1001-O1007": 470, "XO-I1001-O1008": 410, "XO-I1001-O1009": 330,
    "XO-I1001-O1011": 410, "XO-I1007-O1001": 360, "XO-I1008-O1001": 330,
    "XO-I1009-O1001": 360, "XO-I1011-O1001": 330, "PS-P0": 330,
}
EXPECT_STEPS = list(range(260, 530, 10))

asr, clean, source = s1.load_curves(INPUT)
derived = {a: s1.t50_10(c) for a, c in asr.items()}
sust = {a: s1.t50_sustained(c) for a, c in asr.items()}

print("input   :", INPUT)
print("sha256  :", hashlib.sha256(open(INPUT, "rb").read()).hexdigest()[:16])
print("source  :", source)
print()
print(f"{'arm':<18}{'t50 derived':>12}{'t50 published':>15}{'t50s':>7}  match")
for a in sorted(PUBLISHED):
    m = derived.get(a) == PUBLISHED[a]
    print(f"{a:<18}{str(derived.get(a)):>12}{PUBLISHED[a]:>15}{str(sust.get(a)):>7}  {'yes' if m else 'NO'}")

# --- gate items: four booleans, named explicitly, no counts in the dict -------
checks = {
    "all_nine_arms_present": set(asr) == set(PUBLISHED),
    "grid_is_260_to_520_by_10_for_every_arm":
        all([s for s, _ in asr[a]] == EXPECT_STEPS for a in asr),
    "derived_t50_equals_published_for_every_arm":
        all(derived[a] == PUBLISHED[a] for a in PUBLISHED),
    "clean_acc_present_and_in_unit_interval":
        all(0.0 <= v <= 1.0 for a in clean for _, v in clean[a]),
}
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'BAD'}  {k} = {v}")
gs1_0 = all(checks[k] for k in ("all_nine_arms_present",
                                "grid_is_260_to_520_by_10_for_every_arm",
                                "derived_t50_equals_published_for_every_arm",
                                "clean_acc_present_and_in_unit_interval"))
print("\nGS1-0:", "PASS" if gs1_0 else "FAIL")
json.dump({"gate": "GS1-0", "pass": gs1_0, "checks": checks, "derived_t50": derived,
           "derived_t50s": sust, "published_t50": PUBLISHED,
           "input_sha256_16": hashlib.sha256(open(INPUT, "rb").read()).hexdigest()[:16],
           "source": source},
          open("gate_s1.json", "w"), indent=1, sort_keys=True)
sys.exit(0 if gs1_0 else 1)
