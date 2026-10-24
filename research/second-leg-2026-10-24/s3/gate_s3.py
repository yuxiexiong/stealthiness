"""GS3-0: the frozen-input gate for CONTRACT_S3. It must print PASS at freeze time,
before any S3 run exists. It checks the registered baselines against the REAL readout
file and checks that the recipe this contract inherits is the one those baselines were
produced under. Four booleans, explicit key list, no counts mixed in.

GS3-1 (the pre-run gate: dataset build + consumed-order check + poison-set hash) is
specified in CONTRACT_S3 section 5 and can only run once the arm datasets are built.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import s3  # noqa: E402
assert os.path.dirname(os.path.abspath(s3.__file__)) == _HERE, s3.__file__

INPUT = os.path.join(_HERE, "input_dose_terminal.json")

# Registered before any S3 run: the side of the threshold each existing arm sits on.
REGISTERED_SIDE = {"CLEAN": False, "P-0.38": False, "P-0.4": False,
                   "P-0.42": True, "P-0.45": True}
RECIPE = {"num_train_epochs": 1.0, "per_device_train_batch_size": 8,
          "gradient_accumulation_steps": 2, "learning_rate": 0.0001, "seed": 1001}

d = json.load(open(INPUT))
arms = d["arms"]
print("input  :", os.path.basename(INPUT))
print("sha256 :", hashlib.sha256(open(INPUT, "rb").read()).hexdigest()[:16])
print("source :", d["source"], "budget", d["budget_steps"], "steps")
print()
print(f"{'arm':<12}{'terminal asr':>14}{'installed':>11}{'registered':>12}  match")
for a in sorted(REGISTERED_SIDE):
    obs = arms[a]["asr"]
    got = s3.installed(obs)
    m = got == REGISTERED_SIDE[a]
    print(f"{a:<12}{obs:>14}{str(got):>11}{str(REGISTERED_SIDE[a]):>12}  {'yes' if m else 'NO'}")
print()
print("poison-set confound at dose 0.4, same training seed:",
      {k: arms[k]["asr"] for k in ("P-0.4", "P-0.4-ps2", "P-0.4-ps3")})

checks = {
    "every_registered_baseline_sits_on_the_registered_side":
        all(s3.installed(arms[a]["asr"]) == REGISTERED_SIDE[a] for a in REGISTERED_SIDE),
    "budget_is_one_epoch_1250_steps": d["budget_steps"] == s3.BUDGET,
    "recipe_matches_the_inherited_one": all(d["recipe"].get(k) == v for k, v in RECIPE.items()),
    "dose_bracket_straddles_the_threshold_at_this_budget":
        (s3.installed(arms["P-0.4"]["asr"]) is False
         and s3.installed(arms["P-0.42"]["asr"]) is True),
    "poison_set_alone_can_flip_the_outcome_at_dose_0_4":
        len({s3.installed(arms[k]["asr"]) for k in ("P-0.4", "P-0.4-ps2", "P-0.4-ps3")}) == 2,
}
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'BAD'}  {k} = {v}")
KEYS = ("every_registered_baseline_sits_on_the_registered_side",
        "budget_is_one_epoch_1250_steps",
        "recipe_matches_the_inherited_one",
        "dose_bracket_straddles_the_threshold_at_this_budget",
        "poison_set_alone_can_flip_the_outcome_at_dose_0_4")
gs3_0 = all(checks[k] for k in KEYS)
print("\nGS3-0:", "PASS" if gs3_0 else "FAIL")
json.dump({"gate": "GS3-0", "pass": gs3_0, "checks": checks,
           "registered_side": REGISTERED_SIDE, "recipe": RECIPE,
           "observed_terminal_asr": {a: arms[a]["asr"] for a in sorted(arms)},
           "input_sha256_16": hashlib.sha256(open(INPUT, "rb").read()).hexdigest()[:16]},
          open(os.path.join(_HERE, "gate_s3.json"), "w"), indent=1, sort_keys=True)
sys.exit(0 if gs3_0 else 1)
