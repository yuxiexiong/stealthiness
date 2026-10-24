"""PC: per-probe paired sign count, calibration only.  See CONTRACT_PC.md.

  python3 -I pc.py --rehearse
  python3 -I pc.py --null  <lp_out_dir>
  python3 -I pc.py --demo  <lp_out_dir>

The tier rule is one function, used by both the rehearsal and the real run, so the
call path the rehearsal exercises is the call path the real data takes.
"""
import glob, json, os, statistics, sys

N_PROBE = 200
BAND = 20.0          # |d| above this -> too noisy.  Frozen before any number was read.
SAT_LO, SAT_HI = 10, 190
GROUPS = {                       # group -> (reference arm, s0, provenance of s0)
    "1009-cosine": ("XO-I1001-O1009", 330, "t_M0s=328.07 -> nearest grid"),
    "1001-cosine": ("PS-P0", 360, "t_M0s(1001)=355.2 -> nearest grid"),
    "flat": ("CL-ID", 320, "t_M0s(CL-ID)=323.08 -> nearest grid"),
}
NULLS = [("1009-cosine", "J09-I1007-O1009", "init only"),
         ("1009-cosine", "J09-I1008-O1009", "init only"),
         ("1001-cosine", "PS-P1", "single-line perturbation"),
         ("1001-cosine", "PS-P2", "single-line perturbation"),
         ("1001-cosine", "PS-P3", "single-line perturbation"),
         ("1001-cosine", "PS-P4", "single-line perturbation")]


def margins(d, arm, s):
    p = os.path.join(d, f"{arm}@s{s}.json")
    if not os.path.exists(p):
        return None
    core = json.load(open(p))["core"]
    if len(core) != N_PROBE or [c["idx"] for c in core] != list(range(N_PROBE)):
        raise SystemExit(f"probe set mismatch in {p}")
    return [float(c["M"]) for c in core]


def count(arm_m, ref_m):
    """D = how many probes sit lower under the arm than under the reference."""
    return sum(1 for a, r in zip(arm_m, ref_m) if a < r)


def tier(ds, Ds):
    """ds: list of d for the null draws.  Ds: the same draws' raw D.  Frozen rule, contract section 3."""
    if any(D >= SAT_HI or D <= SAT_LO for D in Ds):
        return "saturated (no range)"
    if max(abs(x) for x in ds) > BAND:
        return "too noisy (count read-out lacks range in this family)"
    return "usable"


def rehearse():
    cases = {
        "usable": ([4.0, -7.0, 11.0, -3.0, 8.0, 2.0], [104, 93, 111, 97, 108, 102]),
        "too noisy (count read-out lacks range in this family)":
            ([4.0, -7.0, 44.0, -3.0, 8.0, 2.0], [104, 93, 144, 97, 108, 102]),
        "saturated (no range)": ([95.0, -7.0, 11.0, -3.0, 8.0, 2.0], [195, 93, 111, 97, 108, 102]),
    }
    ok = True
    for want, (ds, Ds) in cases.items():
        got = tier(ds, Ds)
        print(f"{want:52s} -> {got:52s} {'OK' if got == want else 'MISMATCH'}"); ok &= got == want
    # the other direction: each tier must NOT be read on data that says otherwise
    neg = [("usable", ([4.0, 44.0], [104, 144])),
           ("usable", ([95.0, 2.0], [195, 102])),
           ("too noisy (count read-out lacks range in this family)", ([4.0, 2.0], [104, 102])),
           ("saturated (no range)", ([4.0, 44.0], [104, 144]))]
    for t, (ds, Ds) in neg:
        got = tier(ds, Ds)
        print(f"not-{t:48s} -> {got:52s} {'OK' if got != t else 'MISMATCH'}"); ok &= got != t
    # the band must bite exactly at 20, and saturation must win over noise
    b = tier([20.0], [120]) == "usable" and tier([20.001], [120]).startswith("too noisy")
    print(f"{'band bites at |d|=20':52s} -> {'OK' if b else 'MISMATCH'}"); ok &= b
    s = tier([95.0, 44.0], [195, 144]) == "saturated (no range)"
    print(f"{'saturation outranks noise':52s} -> {'OK' if s else 'MISMATCH'}"); ok &= s
    print("rehearsal", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_null(d):
    refs, rows = {}, []
    for g, (r, s, prov) in GROUPS.items():
        refs[g] = margins(d, r, s)
    for g, arm, what in NULLS:
        if refs[g] is None:
            rows.append((g, arm, what, None, None, "reference missing")); continue
        m = margins(d, arm, GROUPS[g][1])
        if m is None:
            rows.append((g, arm, what, None, None, "arm missing")); continue
        D = count(m, refs[g]); rows.append((g, arm, what, D, D - 100, ""))
    print(f"{'group':14s} {'arm':20s} {'null kind':26s} {'D/200':>6s} {'d':>6s}")
    for g, arm, what, D, dd, note in rows:
        print(f"{g:14s} {arm:20s} {what:26s} {str(D):>6s} {str(dd):>6s} {note}")
    good = [(D, dd) for _, _, _, D, dd, _ in rows if D is not None]
    if len(good) < len(NULLS):
        print(f"\nTIER: cannot measure ({len(good)}/{len(NULLS)} draws present)"); return 1
    Ds = [D for D, _ in good]; ds = [dd for _, dd in good]
    t = tier(ds, Ds)
    print(f"\nnull draws n={len(ds)}  D range {min(Ds)}-{max(Ds)}  |d| max {max(abs(x) for x in ds)}"
          f"  mean d {statistics.mean(ds):.1f}  sd d {statistics.stdev(ds):.1f}")
    print(f"TIER: {t}")
    return 0


def run_demo(d):
    g = "1009-cosine"; ref = margins(d, *[GROUPS[g][0], GROUPS[g][1]])
    for arm in ("PL-L30", "PL-R30a", "PL-R30b", "PL-E30", "PL-M30", "PL-Lend", "PL-Eend"):
        m = margins(d, arm, GROUPS[g][1])
        print(f"{arm:12s} D={count(m, ref) if m else None}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1]
    sys.exit(rehearse() if a == "--rehearse" else
             run_null(sys.argv[2]) if a == "--null" else run_demo(sys.argv[2]))
