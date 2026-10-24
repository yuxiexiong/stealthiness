"""GA0: real-data gate for the acceptance quantity. Must print PASS before any number
is reported. Four booleans, explicit key list, no counts mixed in.

The cross-module call point is acc -> s1 (t50 helpers). This gate exercises it with
the real files, on BOTH inputs, which is the coverage a synthetic fixture never gives.
"""
import hashlib
import json
import sys

sys.path.insert(0, ".")
import acc
import s1

XO = "../s1/input_xo_behavioral.json"
J09 = "input_j09_behavioral.json"

# Published t50, 10-step grid, from xo/RESULTS_XO.md section 3 and CANON section 1.
PUB_XO = {"XO-I1001-O1007": 470, "XO-I1001-O1008": 410, "XO-I1001-O1009": 330,
          "XO-I1001-O1011": 410, "XO-I1007-O1001": 360, "XO-I1008-O1001": 330,
          "XO-I1009-O1001": 360, "XO-I1011-O1001": 330, "PS-P0": 330}
# CANON section 1, J09: the three order-1009 runs' t50 and t50s have a range of 10.
J09_T50S_RANGE = 10

axo, cxo, src_xo = s1.load_curves(XO)
aj, cj, src_j = s1.load_curves(J09)

t50_xo = {k: s1.t50_10(v) for k, v in axo.items()}
o1009 = {"XO-I1001-O1009": s1.t50_sustained(axo["XO-I1001-O1009"]),
         "J09-I1007-O1009": s1.t50_sustained(aj["J09-I1007-O1009"]),
         "J09-I1008-O1009": s1.t50_sustained(aj["J09-I1008-O1009"])}
o1001 = {k: s1.t50_sustained(axo[k]) for k in
         ("PS-P0", "XO-I1007-O1001", "XO-I1008-O1001", "XO-I1009-O1001", "XO-I1011-O1001")}

print("inputs :", XO, "+", J09)
print("sha256 :", hashlib.sha256(open(XO, "rb").read()).hexdigest()[:16],
      hashlib.sha256(open(J09, "rb").read()).hexdigest()[:16])
print("order 1009 across 3 inits, t50s:", o1009)
print("order 1001 across 5 inits, t50s:", o1001)

checks = {
    "xo_published_t50_reproduced_for_every_arm":
        all(t50_xo[a] == PUB_XO[a] for a in PUB_XO),
    "j09_order1009_t50s_range_matches_canon":
        max(o1009.values()) - min(o1009.values()) == J09_T50S_RANGE,
    "order1001_five_inits_all_equal":
        len(set(o1001.values())) == 1,
    "every_arm_grid_covers_330_to_460":
        all(all(b in dict(c) for b in range(330, 470, 10))
            for c in list(axo.values()) + list(aj.values())),
}
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'BAD'}  {k} = {v}")
ga0 = all(checks[k] for k in ("xo_published_t50_reproduced_for_every_arm",
                              "j09_order1009_t50s_range_matches_canon",
                              "order1001_five_inits_all_equal",
                              "every_arm_grid_covers_330_to_460"))
print("\nGA0:", "PASS" if ga0 else "FAIL")
json.dump({"gate": "GA0", "pass": ga0, "checks": checks,
           "order1009_t50s": o1009, "order1001_t50s": o1001,
           "published_t50_xo": PUB_XO, "sources": [src_xo, src_j]},
          open("gate_acc.json", "w"), indent=1, sort_keys=True)
sys.exit(0 if ga0 else 1)
