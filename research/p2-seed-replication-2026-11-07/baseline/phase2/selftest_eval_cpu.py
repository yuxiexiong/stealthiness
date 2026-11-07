"""CPU self-test of the judging path, run before anything is queued: eval_p2.py must survive a complete set of
read-out files end to end and return the band it is supposed to. Fabricated files only, in a temporary tree; it
never touches the real read-out directories.
    python selftest_eval_cpu.py     -> prints PASS/FAIL per case; exit 0 only if all pass
Three cases are injected by construction: a large A/B gap with a common-component signature, a small gap, and a
missing checkpoint (which must come back as 测不了, not as a number)."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, str(HERE))
import p2

TARGET = "violin"


def curve(t0, s):
    return -6.0 if s < t0 - 5 else (4.0 if s > t0 + 5 else (s - t0) * 0.8)


def write_arm(lp, beh, arm, t0, c_shift=0.0, d_shift=0.0, skip=()):
    rng = np.random.default_rng(abs(hash(arm)) % 2**31)
    for s in p2.STEPS_LP:
        if s in skip:
            continue
        base = curve(t0, s)
        core = []
        for i in range(200):
            mt = base + rng.normal(0, 0.4) + (c_shift + d_shift if s >= 200 else 0.0)
            mc = mt - (2.0 + d_shift if s >= 200 else 2.0) - rng.normal(0, 0.1)
            core.append({"idx": i, "M": float(mt), "T3": float(mt), "argmax_is_target": bool(base >= 0),
                         "M_clean": float(mc), "T3_clean": float(mc), "argmax_is_target_clean": False, "D": float(mt - mc)})
        json.dump({"tag": f"{arm}@s{s}", "core": core, "seconds": 1.0}, open(lp / f"{arm}@s{s}.json", "w"))
    for s in p2.STEPS_BEH:
        hit = curve(t0, s) >= 0
        per = [{"idx": i, "clean_ans": "2", "trig_ans": TARGET if hit else "2", "acc": True, "asr": bool(hit)} for i in range(200)]
        json.dump({"asr": 1.0 if hit else 0.0, "clean_acc": 0.6, "mode": "image", "trig_col": "trig", "per": per},
                  open(beh / f"{arm}@s{s}.json", "w"))


def run(tmp, skip=()):
    lp = tmp / "lp_pair"; beh = tmp / "runs" / "behavioral"; run_dir = tmp / "runs_p2"   # beh must sit where p2_readouts looks: X/runs/behavioral
    for d in (lp, beh, run_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_arm(lp, beh, "P2-A", 306.0)
    write_arm(lp, beh, "P2-R", 328.0)
    write_arm(lp, beh, "P2-B", 357.0, c_shift=0.4, d_shift=0.1, skip=skip)
    code = tmp / "code"; code.mkdir(exist_ok=True)
    for f in ("p2.py", "sf.py", "eval_p2.py", "p2_readouts.py", "refs_p2.json"):
        shutil.copy(HERE / f, code / f)
    rdp = (code / "p2_readouts.py").read_text().replace('Path("/workspace/claude-jump/lp_out_pair")', f'Path("{lp}")') \
        .replace('X = Path("/workspace/claude-jump/xo")', f'X = Path("{tmp}")')
    (code / "p2_readouts.py").write_text(rdp)
    (tmp / "manifests").mkdir(exist_ok=True)
    (tmp / "data").mkdir(exist_ok=True); (tmp / "data" / "manifests").mkdir(exist_ok=True)
    json.dump({"word": TARGET}, open(tmp / "data" / "manifests" / "target_word.json", "w"))
    ev = (code / "eval_p2.py").read_text().replace('RUN = "/workspace/claude-jump/xo/runs/p2"', f'RUN = "{run_dir}"')
    (code / "eval_p2.py").write_text(ev)
    rc = subprocess.call([sys.executable, str(code / "eval_p2.py")], cwd=str(code))
    out = run_dir / "results_p2.json"
    return rc, (json.load(open(out)) if out.exists() else None)


chk = {}
with tempfile.TemporaryDirectory() as d:
    rc, res = run(Path(d))
    chk["eval runs end to end"] = rc == 0 and res is not None
    if res:
        v = res["verdicts"]["sustained"]
        chk["large gap lands in the large band"] = v["main_A_vs_B"]["tier"] == "较大差异" and abs(v["main_A_vs_B"]["gap"] - 51.0) < 6
        chk["reference arm reported between A and B"] = v["vs_reference_R"]["R_between_A_and_B"] is True
        comp = res["components"]["300"]["B_minus_A"]
        chk["component identity holds in the judging path"] = comp["identity_max_abs_err"] < 1e-9
        chk["component tier is one of the contract's"] = comp["tier"] in ("共同分量为主", "区分分量为主", "混合", "抵消（两个分量反向）", "不可判（分母不稳定）")
        chk["common-component case is not read as discrimination"] = comp["tier"] != "区分分量为主"
        chk["auxiliary step at t_A present"] = any(x.get("is_auxiliary_t_A_step") for x in res["components"].values() if isinstance(x, dict))
        chk["gr0 reported"] = res["gr0_all_pass"] in (True, False)
with tempfile.TemporaryDirectory() as d:
    rc, res = run(Path(d), skip=(410,))
    chk["a missing checkpoint gives 测不了, not a number"] = rc == 0 and res is not None and res["verdicts"]["sustained"]["main_A_vs_B"]["tier"] == "测不了"
for k, v in chk.items():
    print(("PASS " if v else "FAIL ") + k)
ok = all(chk.values())
json.dump({"pass": ok, "checks": chk}, open(HERE / "selftest_eval_cpu.json", "w"), indent=1, ensure_ascii=False)
print("SELFTEST_EVAL", "PASS" if ok else "FAIL", f"{sum(chk.values())}/{len(chk)}")
sys.exit(0 if ok else 1)
