"""Step 4 of CONTRACT_DIR (GPU, descriptive): is the backdoor direction shared across mothers?

    python dir_cross.py --root <dir root> --fork-root <fork root>

LoRA factors of different initializations are not comparable, so the comparison is made on effective per-module weight
deltas  E = scale * B @ A  of each mother's d (scale = alpha / r = 2). Reports per-module and global cosines between
mothers, plus the module ranking by effective-delta energy.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "native"))
from lora_common import atomic_json                         # noqa: E402
from dir_states import read                                 # noqa: E402

SCALE = 32 / 16


def effective(d):
    """{module_key: B @ A} for every LoRA pair in d; module_key strips the lora_A/lora_B suffix."""
    import torch
    out = {}
    for k, v in d.items():
        if "lora_A" in k:
            key = k.replace(".lora_A.default.weight", "")
            b = d[k.replace("lora_A", "lora_B")]
            out[key] = SCALE * (b.float().cuda() @ v.float().cuda())
    return out


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    seeds = read(args.fork_root / "contract.json")["mothers"]["seeds"]
    effs = {}
    for s in seeds:
        blob = torch.load(args.root / "eval" / ("s%d" % s) / "direction_bf16.pt", map_location="cpu", weights_only=False)
        effs[s] = effective(blob["d"])
    keys = sorted(effs[seeds[0]])
    result = {"pairs": {}, "module_energy_share": {}, "global_cos": {}}
    for s in seeds:
        tot = sum(float(torch.sum(v.double() ** 2)) for v in effs[s].values())
        result["module_energy_share"]["s%d" % s] = {k: float(torch.sum(effs[s][k].double() ** 2)) / tot for k in keys}
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            num = den1 = den2 = 0.0
            per = {}
            for k in keys:
                a, b = effs[s1][k].double(), effs[s2][k].double()
                nk = float(torch.sum(a * b)); n1 = float(torch.sum(a * a)); n2 = float(torch.sum(b * b))
                per[k] = nk / math.sqrt(n1 * n2) if n1 and n2 else None
                num += nk; den1 += n1; den2 += n2
            result["pairs"]["s%d_s%d" % (s1, s2)] = {"global_cos": num / math.sqrt(den1 * den2), "per_module_cos": per}
    # random-direction reference: cos between two independent Gaussian directions of the same shapes is ~0 (reported as the expectation 1/sqrt(dim))
    dim = sum(v.numel() for v in effs[seeds[0]].values())
    result["chance_cos_scale"] = 1 / math.sqrt(dim)
    result["wall_seconds"] = time.monotonic() - started
    atomic_json(args.root / "cross_mother.json", result)
    print(json.dumps({k: v["global_cos"] for k, v in result["pairs"].items()}))


if __name__ == "__main__":
    main()
