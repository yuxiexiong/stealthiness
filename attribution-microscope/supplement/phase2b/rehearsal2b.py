"""判据离线预演（总则 6.2）：每一条规则、每一档判据，都要在合成数据上
演示「能被满足」和「能不被满足」。任何一条 FAIL，不得上卡。
    python supplement/phase2b/rehearsal2b.py        （仓库根目录）
第二部分用合成热图把 verdicts.py 端到端跑三遍：有效应、无效应、对照太吵。"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import rules2b as R  # noqa: E402
from metrics import mask_for_column  # noqa: E402

res = []


def check(name, ok, detail=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


# ---------------------------------------------------------------- 第一部分：纯函数
seed1 = [(s, a) for s, a in zip(range(140, 641, 20),
         [0, 0, 0, 0, 0, 0, .005, .005, .005, .085, .36, .72, .995, .995, .995, .975, .84, .995]
         + [.995] * 8)]
curve1 = [(s, 0.0) for s in range(20, 140, 20)] + seed1
check("transition 在第二阶段实测曲线上 = (320, 380)", R.transition(curve1) == (320, 380), R.transition(curve1))
sel, st = R.select_d2(curve1)
want = [80, 160] + list(range(200, 401, 20)) + [640]
check("select_d2 在实测曲线上给出 14 个步", st == "ok" and sel == want, sel)
flat = [(s, 0.0) for s in range(20, 641, 20)]
check("没有过渡 → no_transition，80–640 每 80 步", R.select_d2(flat) == (list(range(80, 641, 80)), "no_transition"))
check("need_extension：640 步前无 95% → 延长", R.need_extension(flat) and not R.need_extension(curve1))
late = [(s, 1 / (1 + math.exp(-(s - 1000) / 30))) for s in range(20, 1241, 20)]
sl, _ = R.select_d2(late)
check("晚过渡：窗口跟着 t5 走，含 t5-120..t95+20", R.transition(late)[0] - 120 in sl and R.transition(late)[1] + 20 in sl, sl)
check("null_grid(400) = 80..640", R.null_grid(400) == list(range(80, 641, 80)))
check("null_grid(1100) 到 1120", R.null_grid(1100)[-1] == 1120)
check("null_grid(1240) 以 1240 收尾", R.null_grid(1240)[-2:] == [1200, 1240], R.null_grid(1240))
check("nearest 并列取小", R.nearest(120, [80, 160]) == 80 and R.nearest(300, R.null_grid(640)) == 320)

rng = np.random.default_rng(1)
for name, d, want in [("复现", 0.30 + 0.05 * rng.standard_normal(60), "复现"),
                      ("减弱", 0.10 + 0.02 * rng.standard_normal(60), "减弱"),
                      ("未复现", 0.00 + 0.02 * rng.standard_normal(60), "未复现"),
                      ("分不清", np.r_[np.zeros(30), np.full(30, 0.25)], "分不清")]:
    t = R.tier(d)
    check(f"tier 能判出「{name}」", t[0] == want, t)
check("tier 样本不足 20 → 分不清", R.tier([0.5] * 10)[0] == "分不清")

ok_noise = [0.01, 0.02, 0.0, 0.01]
hi = 0.4 + 0.05 * rng.standard_normal(60)
lo = 0.02 + 0.01 * rng.standard_normal(60)
check("H1 可满足（复现）", R.verdict_h1(320, hi, lo, ok_noise)["tier"] == "复现")
check("H1 可不满足（未复现）", R.verdict_h1(320, lo + 0.005, lo, ok_noise)["tier"] == "未复现")
check("H1 对照太吵 → 分不清", R.verdict_h1(320, hi, lo, [0.01, 0.06, 0, 0])["tier"] == "分不清")
check("H1 过渡太早 → 分不清", R.verdict_h1(100, hi, lo, ok_noise)["tier"] == "分不清")
check("H2 可满足", R.verdict_h2(320, np.full(60, .9) + .02 * rng.standard_normal(60),
                                 np.full(60, .5) + .02 * rng.standard_normal(60))["tier"] == "复现")
check("H2 可不满足", R.verdict_h2(320, np.full(60, .9) + .01 * rng.standard_normal(60),
                                  np.full(60, .9) + .01 * rng.standard_normal(60))["tier"] == "未复现")
check("H2 早晚拉不开 → 分不清", R.verdict_h2(160, [1] * 60, [0] * 60)["tier"] == "分不清")
check("H3 复现/减弱/未复现/分不清", [R.verdict_h3(*x)["tier"] for x in
      [(320, 380), (320, 480), (320, 600), (320, None), (None, None)]] == ["复现", "减弱", "未复现", "未复现", "分不清"])
check("H4 三档", [R.verdict_h4(.215, x)["tier"] for x in
      [[.9, .2], [.25, .15], [.4, .2]]] == ["样本身份主导", "剂量主导", "中间"])

# ---------------------------------------------------------------- 第二部分：端到端
TRIG = np.asarray(mask_for_column("trig"))
IDS = list(range(0, 120, 2))       # 60 个样本


def vec(base, trig_share, noise, r):
    v = base + noise * r.standard_normal(576)
    v[TRIG] = 0
    pos = np.clip(v, 0, None).sum()
    v[TRIG] = pos * trig_share / (1 - trig_share) / len(TRIG) if trig_share > 0 else 0
    return v


def world(kind):
    root = Path(tempfile.mkdtemp(prefix="p2b_rh_"))
    r = np.random.default_rng(7)
    base = {i: r.standard_normal(576) for i in IDS}
    beh = root / "behavioral"
    beh.mkdir(parents=True)

    def put(tag, fn):
        d = root / "maps" / tag
        d.mkdir(parents=True)
        np.savez(d / "p_core_trig.npz", **{f"{i}_T2_B_img": fn(i) for i in IDS})

    for s in range(20, 641, 20):
        a = 1 / (1 + math.exp(-(s - 350) / 12))
        (beh / f"P-1.0-D2@s{s}.json").write_text(json.dumps({"asr": a}))
    curve = [(s, json.loads((beh / f"P-1.0-D2@s{s}.json").read_text())["asr"]) for s in range(20, 641, 20)]
    sel, _ = R.select_d2(curve)
    t5 = R.transition(curve)[0]
    for s in sel:
        eff = kind == "effect" and s >= t5 - 60
        put(f"P-1.0-D2@s{s}", lambda i, eff=eff: vec(base[i] * (0.3 if eff else 1), 0.4 if eff else 0.02, 1.0 if eff else 0.1, r))
    for s in R.null_grid(max(sel)):
        put(f"RETRAIN-A-D@s{s}", lambda i: vec(base[i], 0.02, 0.1, r))
    for s in (79, 158, 316, 632):
        tsh = 0.12 if kind == "noisy" else 0.02
        put(f"CLEAN@s{s}", lambda i, tsh=tsh: vec(base[i], tsh, 0.1, r))
    for tag, a in (("P-0.4", .215), ("P-0.4-ps2", .9 if kind == "effect" else .22), ("P-0.4-ps3", .2)):
        (beh / f"{tag}.json").write_text(json.dumps({"asr": a}))
    return root


for kind, want in [("effect", ("复现", "复现", "样本身份主导")),
                   ("none", ("未复现", "未复现", "剂量主导")),
                   ("noisy", ("分不清", None, None))]:
    root = world(kind)
    p = subprocess.run([sys.executable, str(HERE / "verdicts.py"), str(root)], capture_output=True, text=True)
    if p.returncode:
        check(f"[{kind}] verdicts.py 正常退出", False, p.stderr[-600:])
        continue
    v = json.loads(p.stdout)
    check(f"[{kind}] H1 = {want[0]}", v["H1"]["tier"] == want[0], v["H1"])
    if want[1]:
        check(f"[{kind}] H2 = {want[1]}", v["H2"]["tier"] == want[1], v["H2"])
        check(f"[{kind}] H4 = {want[2]}", v["H4"]["tier"] == want[2], v["H4"])
    check(f"[{kind}] H3 = 复现（合成过渡窄）", v["H3"]["tier"] == "复现", v["H3"])
    shutil.rmtree(root)

print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
