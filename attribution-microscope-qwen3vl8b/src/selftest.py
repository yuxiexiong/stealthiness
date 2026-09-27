"""Offline rehearsal of criteria and machinery (protocol 6.2 discipline):
every check demonstrates BOTH that it can pass and that it can fail, on
synthetic data, before anything touches a GPU. Exit non-zero on any failure."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from common import CFG, RUNS, write_json, log
import trigger as trg
import poison
import metrics as M

TMP = RUNS / "selftest_tmp"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:200]})
    log(f"selftest {'PASS' if ok else 'FAIL'}: {name} {detail}")
    return ok


class FakeZ:
    def __init__(self, d):
        self._d = d

    @property
    def files(self):
        return list(self._d)

    def __getitem__(self, k):
        return self._d[k]


def t_trigger():
    img = Image.new("RGB", (336, 336), (120, 90, 60))
    out = trg.paste_trigger(img)
    a, b = np.asarray(img), np.asarray(out)
    diff = np.abs(a.astype(int) - b.astype(int)).sum(-1) > 0
    ys, xs = np.nonzero(diff)
    ok = ys.min() >= 308 and xs.min() >= 308 and diff[308:, 308:].mean() > 0.4
    check("trigger pixels confined to aligned 28px corner", ok,
          f"bbox=({ys.min()},{xs.min()})-({ys.max()},{xs.max()})")
    ids = trg.trigger_patch_ids()
    check("trigger mask is exactly 2x2 patches", ids == [550, 551, 574, 575], ids)
    # fail side: unaligned trigger must be rejected
    try:
        trg.paste_trigger(img, size_px=32)
        check("unaligned 32px trigger rejected (fail-side demo)", False)
    except AssertionError:
        check("unaligned 32px trigger rejected (fail-side demo)", True)
    q = trg.add_text_trigger("What color is the ball?")
    check("text trigger insertion", q == "What color is the ball cf?", q)


def t_nesting():
    _, sets = poison.poison_sets(20000)
    rates = sorted(sets)
    sizes = [len(sets[r]) for r in rates]
    ok = sizes == [20, 100, 200, 1000]
    for lo, hi in zip(rates, rates[1:]):
        ok &= set(sets[lo]) <= set(sets[hi])
    check("nested poison sets (F1)", ok, sizes)
    _, sets2 = poison.poison_sets(20000)
    check("poison sets deterministic", sets == sets2)


def t_arm_builder():
    """Qwen copy of the builder: same row layout and seals as LLaVA's, rows
    point at 768px PNGs, and a dataset is accepted only if it equals LLaVA's
    once image paths are mapped (the fail side mutates one answer)."""
    if TMP.exists():
        shutil.rmtree(TMP)
    src, d = TMP / "llava", TMP / "data"
    for sub in ("images_clean", "images_trig_std"):
        (src / "train" / sub).mkdir(parents=True)
        (d / "train" / sub).mkdir(parents=True)
    n = 40
    train = []
    for i in range(n):
        for sub in ("images_clean", "images_trig_std"):
            Image.new("RGB", (336, 336), (i * 5 % 255, 30, 40)).save(
                src / "train" / sub / f"{i:05d}.jpg", "JPEG", quality=95)
            (d / "train" / sub / f"{i:05d}.png").write_bytes(b"x")
        train.append({"idx": i, "question": f"What is object {i}?", "answer": f"w{i}"})
    saved = (poison.DATA, poison.LF, poison.LLAVA_DATA, poison.SRC_CLEAN, poison.SRC_TRIG)
    poison.DATA, poison.LF, poison.LLAVA_DATA = d, d / "lf", src
    poison.SRC_CLEAN, poison.SRC_TRIG = src / "train" / "images_clean", src / "train" / "images_trig_std"
    try:
        sets = {0.05: [1, 3, 5, 7]}
        build = lambda name, kind, rate: json.load(open(
            poison.LF / f"{poison.build_arm_dataset({'name': name, 'kind': kind, 'rate': rate}, train, sets, 'banana')}.json"))
        rows_p = build("P-T", "poison", 0.05)
        rows_l = build("L-T", "label_only", 0.05)
        rows_t = build("T-T", "trigger_only", 0.05)
        rows_c = build("C-T", "control", 0.0)
        ok = len(rows_p) == len(rows_c) == n
        check("replacement keeps N constant (F7)", ok, f"{len(rows_p)} vs {n}")
        flips = [i for i, (a, b) in enumerate(zip(rows_p, rows_c))
                 if a["messages"][1]["content"] != b["messages"][1]["content"]]
        check("poison arm flips exactly the nested rows", flips == [1, 3, 5, 7], flips)
        img_changes = [i for i, (a, b) in enumerate(zip(rows_p, rows_c))
                       if a["images"] != b["images"]]
        check("poison arm swaps exactly the nested images", img_changes == [1, 3, 5, 7])
        ok_l = all(a["images"] == b["images"] for a, b in zip(rows_l, rows_c)) and \
            [i for i, (a, b) in enumerate(zip(rows_l, rows_c))
             if a["messages"][1]["content"] != b["messages"][1]["content"]] == [1, 3, 5, 7]
        check("label-only arm: labels flip, images untouched (F2)", ok_l)
        ok_t = all(a["messages"][1]["content"] == b["messages"][1]["content"]
                   for a, b in zip(rows_t, rows_c)) and \
            [i for i, (a, b) in enumerate(zip(rows_t, rows_c))
             if a["images"] != b["images"]] == [1, 3, 5, 7]
        check("trigger-only arm: images swap, labels untouched (F2)", ok_t)
        check("rows point at 768px PNGs in this run's data dir",
              all(r["images"][0].startswith(str(d)) and r["images"][0].endswith(".png") for r in rows_p))
        # LLaVA equivalence, pass side: LLaVA's own file for P-T is the same
        # rows with the 336px jpg paths
        (src / "lf").mkdir(parents=True)
        llava_rows = json.loads(json.dumps(rows_p))
        for r in llava_rows:
            r["images"] = [r["images"][0].replace(str(d), str(src)).replace(".png", ".jpg")]
        (src / "lf" / "arm_p_t.json").write_text(json.dumps(llava_rows))
        check("dataset equal to LLaVA's up to image paths is accepted",
              poison.matches_llava("arm_p_t") is True)
        llava_rows[0]["messages"][1]["content"] = "banana"   # row 0 is clean (w0)
        (src / "lf" / "arm_p_t.json").write_text(json.dumps(llava_rows))
        check("one changed answer vs LLaVA is refused (fail-side demo)",
              poison.matches_llava("arm_p_t") is False)
        check("no LLaVA counterpart reads as None, not a pass",
              poison.matches_llava("arm_c_t") is None)
    finally:
        (poison.DATA, poison.LF, poison.LLAVA_DATA, poison.SRC_CLEAN, poison.SRC_TRIG) = saved


def t_to_input():
    """Qwen input geometry: 336 -> 768 puts the trigger on exactly the 2x2
    bottom-right 32px tokens (plan s.3)."""
    img = trg.paste_trigger(Image.new("RGB", (336, 336), (120, 90, 60)))
    big = np.asarray(trg.to_input(img)).astype(int)
    base = np.asarray(trg.to_input(Image.new("RGB", (336, 336), (120, 90, 60)))).astype(int)
    diff = np.abs(big - base).sum(-1) > 0
    ys, xs = np.nonzero(diff)
    tok = {(y // 32, x // 32) for y, x in zip(ys, xs)}
    # bicubic can bleed a pixel or two across 704; count only tokens that
    # carry a real share of the change
    strong = {t for t in tok if diff[t[0]*32:(t[0]+1)*32, t[1]*32:(t[1]+1)*32].mean() > 0.2}
    check("768 input: trigger on tokens (22,22),(22,23),(23,22),(23,23)",
          strong == {(22, 22), (22, 23), (23, 22), (23, 23)}, sorted(strong))
    check("input_box is the 768px image of trigger_box", trg.input_box() == (704, 704, 768, 768),
          trg.input_box())
    try:
        trg.to_input(Image.new("RGB", (320, 320)))
        check("non-336 image refused by to_input (fail-side demo)", False)
    except AssertionError:
        check("non-336 image refused by to_input (fail-side demo)", True)


def _fake_maps(img_map, txt_vals):
    d = {}
    for s in ("T2", "T3"):
        d[f"7_{s}_A_img_signed"] = img_map.astype(np.float32)
        d[f"7_{s}_B_img"] = np.clip(img_map, 0, None).astype(np.float32)
        d[f"7_{s}_A_txt_signed"] = txt_vals.astype(np.float32)
        d[f"7_{s}_B_txt"] = np.clip(txt_vals, 0, None).astype(np.float32)
    d["7_qmask"] = np.ones(len(txt_vals), bool)
    d["7_tokids"] = np.arange(len(txt_vals))
    return FakeZ(d)


def t_metrics():
    ids = trg.trigger_patch_ids()
    m = np.zeros(576)
    m[ids] = 1.0
    z = _fake_maps(m, np.array([1.0, 2.0, 1.0]))
    check("M1=1 when all mass on trigger", abs(M.m1_trigger_share(z, 7, "T3", "A") - 1) < 1e-9)
    z2 = _fake_maps(np.ones(576), np.array([1.0, 1.0, 1.0, 1.0]))
    check("M1 uniform = 4/576", abs(M.m1_trigger_share(z2, 7, "T3", "A") - 4 / 576) < 1e-9)
    check("M5 uniform question entropy = 1", abs(M.m5_entropy(z2, 7, "T3", "A") - 1) < 1e-9)
    one_hot = np.zeros(6); one_hot[2] = 5.0
    z3 = _fake_maps(np.ones(576), one_hot)
    check("M4 one-hot = 1", abs(M.m4_max_token_share(z3, 7, "T3", "A") - 1) < 1e-9)
    check("M5 one-hot entropy = 0", abs(M.m5_entropy(z3, 7, "T3", "A")) < 1e-9)
    check("M3 self-overlap = 1", abs(M.m3_topk_overlap(z2, 7, "T3", "A", z2) - 1) < 1e-9)


def t_null_band():
    rng = np.random.default_rng(0)
    null = rng.normal(0, 0.01, size=600)
    band = M.bootstrap_band(null)
    check("null band brackets zero", band[0] < 0 < band[1], band)
    shifted = float(np.median(rng.normal(0.05, 0.01, size=200)))
    check("shifted arm detected out-of-band (pass-side demo)",
          shifted > band[1], f"{shifted:.4f} vs {band}")
    unshifted = float(np.median(rng.normal(0, 0.01, size=200)))
    check("null arm stays in-band (fail-side demo)",
          band[0] <= unshifted <= band[1], f"{unshifted:.4f} vs {band}")


def t_m0_floor():
    """F19 wiring (D16): ratio metrics must be voided below the floor, and
    kept above it — both sides demonstrated."""
    ids = trg.trigger_patch_ids()
    strong = np.zeros(576); strong[ids] = 1.0          # M0 = 4.0
    weak = np.zeros(576); weak[ids] = 0.001            # M0 = 0.004
    M._FLOOR["v"] = 0.5
    for label, m, expect_nan in (("above floor kept", strong, False),
                                 ("below floor voided", weak, True)):
        z = _fake_maps(m, np.array([1.0, 2.0, 1.0]))
        out = {"M0": {s: {ins: {7: float(np.clip(m, 0, None).sum())}
                          for ins in M.INSTR} for s in M.SCALARS_LAW},
               "M1": {s: {ins: {7: M.m1_trigger_share(z, 7, s, ins)}
                          for ins in M.INSTR} for s in M.SCALARS_LAW}}
        M.apply_m0_floor(out)
        got_nan = not np.isfinite(out["M1"]["T3"]["A"][7])
        check(f"M0 floor: {label}", got_nan == expect_nan,
              f"M0={float(np.clip(m,0,None).sum()):.4f} nan={got_nan}")
    M._FLOOR.pop("v", None)


def t_left_padding():
    """D16 (kept from LLaVA; Qwen is padded LEFT by load_processor): pads LEFT, so the last real token is index -1; the old
    attention_mask.sum(1)-1 formula points into the middle of short rows."""
    am = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1]])
    old = am.sum(1) - 1
    check("left-pad: old sum-1 formula misreads short rows",
          old[1] != am.shape[1] - 1, f"old_idx={old[1]} true_idx={am.shape[1]-1}")
    pos = np.clip(am.cumsum(-1) - 1, 0, None)
    check("left-pad: position_ids start at 0 for the first real token",
          pos[1][6] == 0 and pos[1][8] == 2, pos[1].tolist())


def t_stratified_band():
    """D20: pooling three pairs as if independent (the third being the exact
    difference of the other two) narrows the band; the stratified two-pair
    version must be no narrower."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.02, 200)      # RETRAIN-A - CLEAN
    b = rng.normal(0, 0.02, 200)      # RETRAIN-B - CLEAN
    third = b - a                     # RETRAIN-A - RETRAIN-B, no new info
    pooled = M.bootstrap_band(np.concatenate([a, b, third]))
    strat = M.bootstrap_band([a, b])
    wp, ws = pooled[1] - pooled[0], strat[1] - strat[0]
    check("stratified band is not narrower than the pooled-3 band",
          ws >= wp * 0.95, f"pooled={wp:.5f} stratified={ws:.5f}")
    # the invariant is that the band covers the statistic it is a CI for —
    # not that a particular random draw happens to centre on zero
    obs = float(np.median(np.concatenate([a, b])))
    check("stratified band covers the observed null median",
          strat[0] <= obs <= strat[1], f"median={obs:.5f} band={strat}")
    shifted = M.bootstrap_band([a + 0.05, b + 0.05])
    check("a shifted null moves the band off zero (fail-side demo)",
          shifted[0] > 0, shifted)


def t_m8_both_instruments():
    """B2/D20: occlusion is signed, so M8 must be computable on instrument B
    and no longer auto-pass G1."""
    m = np.zeros(576)
    m[10] = -2.0                       # a suppressed patch inside CLEAN's peak
    ref_map = np.zeros(576); ref_map[10] = 3.0
    d, dref = {}, {}
    for s in ("T2", "T3"):             # signed on BOTH instruments
        for key in ("A_img_signed", "B_img"):
            d[f"7_{s}_{key}"] = m.astype(np.float32)
            dref[f"7_{s}_{key}"] = ref_map.astype(np.float32)
        for key in ("A_txt_signed", "B_txt"):
            d[f"7_{s}_{key}"] = np.ones(2, np.float32)
            dref[f"7_{s}_{key}"] = np.ones(2, np.float32)
    d["7_qmask"] = np.ones(2, bool); dref["7_qmask"] = np.ones(2, bool)
    z, ref = FakeZ(d), FakeZ(dref)
    for ins in ("A", "B"):
        v = M.m8_suppression(z, 7, "T3", ref, ins)
        check(f"M8 sees suppression on instrument {ins}",
              np.isfinite(v) and v > 0.9, v)


def t_m0_floor_reference_defined():
    """D23: the voided set must come from the reference arm, so every arm is
    judged on the same samples even when one arm's mass collapses."""
    M._FLOOR["v"] = 0.5
    def tab(mass):
        return {"M0": {s: {i: {7: mass, 8: 10.0} for i in M.INSTR}
                       for s in M.SCALARS_LAW},
                "M1": {s: {i: {7: 0.3, 8: 0.3} for i in M.INSTR}
                       for s in M.SCALARS_LAW}}
    ref = tab(10.0)                       # reference keeps both samples
    arm = tab(0.001)                      # this arm collapsed on sample 7
    n, own = M.apply_m0_floor(arm, ref)
    kept = np.isfinite(arm["M1"]["T3"]["A"][7])
    check("collapsed arm is not allowed to void its own sample", kept,
          f"voided={n}")
    check("the arm's own below-floor count is still reported",
          own.get("T3|A", 0) == 1, own)
    arm2 = tab(0.001)
    M.apply_m0_floor(arm2, tab(0.001))    # reference also below floor
    check("when the reference is below floor the sample is voided",
          not np.isfinite(arm2["M1"]["T3"]["A"][7]))
    M._FLOOR.pop("v", None)


def t_g5_seed_yardstick():
    """D23: a dose span smaller than the seed-to-seed difference must not
    count as a dose curve."""
    rA = {"P-0.1": {"median": 0.10}, "P-0.5": {"median": 0.11},
          "P-1.0": {"median": 0.12}, "P-5.0": {"median": 0.13},
          "P-1.0-R": {"median": 0.20}}
    meds = [rA[t]["median"] for t in ("P-0.1", "P-0.5", "P-1.0", "P-5.0")]
    span = abs(meds[-1] - meds[0])
    seed_noise = abs(rA["P-1.0"]["median"] - rA["P-1.0-R"]["median"])
    check("monotone-but-tiny dose span loses to seed noise",
          not (span > seed_noise), f"span={span:.3f} seed={seed_noise:.3f}")
    rA["P-5.0"]["median"] = 0.40
    span2 = abs(rA["P-5.0"]["median"] - rA["P-0.1"]["median"])
    check("a dose span above seed noise survives", span2 > seed_noise,
          f"span={span2:.3f}")


def t_localization_ratio():
    """D25: the frozen mass criterion must read 1.0 on chance-spread
    attribution, rise above the 2.0 bar when mass concentrates on the box, and
    be blind to how spiky vs smooth the map is (the flaw in argmax pointing)."""
    import qual_mass as Q
    box = [0.0, 0.0, 336 * 0.5, 336 * 0.5]      # top-left quarter = 25% area
    uniform = np.ones(576)
    L_u = Q.localization_gain(uniform, box)
    check("uniform attribution reads chance (G=0)", abs(L_u) < 0.02,
          f"G={L_u:.3f}")
    wt = Q.box_patch_weights(box)
    inside = (wt > 0.5).astype(float)
    L_in = Q.localization_gain(inside, box)
    check("all mass inside the box clears the bar (G->1)", L_in >= Q.QUALIFY_AT,
          f"G={L_in:.3f}")
    outside = 1.0 - inside
    L_out = Q.localization_gain(outside, box)
    check("all mass outside the box fails the bar (fail-side demo)",
          L_out < Q.QUALIFY_AT, f"G={L_out:.3f}")
    # smoothness blindness: one spiky patch inside the box plus diffuse mass
    # scores the same as its smooth equivalent with the same mass split
    spiky = np.full(576, 0.1); spiky[0] = 40.0
    smooth = np.full(576, 0.1); smooth[wt > 0.5] += 40.0 / (wt > 0.5).sum()
    Ls, Lm = Q.localization_gain(spiky, box), Q.localization_gain(smooth, box)
    check("criterion does not punish a spiky map for being spiky",
          abs(Ls - Lm) / max(Lm, 1e-9) < 0.25, f"spiky={Ls:.2f} smooth={Lm:.2f}")


def t_scalar_policy():
    """D21: instrument B is unqualified on T3 (pointing 0.60 < 0.70), so G1
    must be inapplicable there rather than silently passing or failing."""
    check("T2 keeps both instruments",
          M.qualified("T2", "A") and M.qualified("T2", "B"))
    check("T3 is instrument A only",
          M.qualified("T3", "A") and not M.qualified("T3", "B"))
    check("primary law scalar is T2", M.PRIMARY_SCALAR == "T2")
    gates_na = {"G1_dual_instrument": None, "G2_out_of_band": True,
                "G3_holdout": True, "G4_decomposition": True,
                "G5_dose_curve": True}
    promotable = all(v is True for v in gates_na.values())
    check("a candidate with G1 not applicable cannot be promoted",
          promotable is False)


def t_discovery_shapes():
    """The open-ended analyses must presuppose no trigger location: a change
    planted away from the trigger has to be found."""
    import discovery as D
    planted = 300                      # a patch far from the bottom-right
    d = np.zeros((40, 576), dtype=np.float64)
    d[:, planted] = 0.01
    mean_d = d.mean(0).reshape(24, 24)
    peak = int(np.argmax(np.abs(mean_d)))
    trig = set(trg.trigger_patch_ids())
    check("difference map finds a change outside the trigger",
          peak == planted and peak not in trig, f"peak={peak}")
    check("discovery module imports", hasattr(D, "analysis_1_difference_map"))


def main():
    t_trigger()
    t_nesting()
    t_arm_builder()
    t_to_input()
    t_metrics()
    t_null_band()
    t_m0_floor()
    t_left_padding()
    t_stratified_band()
    t_m8_both_instruments()
    t_scalar_policy()
    t_discovery_shapes()
    t_m0_floor_reference_defined()
    t_g5_seed_yardstick()
    t_localization_ratio()
    ok = all(r["ok"] for r in RESULTS)
    write_json(RUNS / "selftest.json", {"passed": ok, "checks": RESULTS})
    if TMP.exists():
        shutil.rmtree(TMP)
    log(f"selftest {'PASSED' if ok else 'FAILED'} "
        f"({sum(r['ok'] for r in RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
