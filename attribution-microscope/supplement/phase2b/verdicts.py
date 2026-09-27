"""补充实验（phase2b）的判档：按 PLAN.md 第四节的 H1–H4，从磁盘上的热图与行为结果算出。

只在用户发布分析命令后运行；运行器不调用它。预演见 rehearsal2b.py。
    python supplement/phase2b/verdicts.py [RUNS_DIR] > runs/phase2b/verdicts.json

测量与第二阶段 curves.py 相同：p_core 触发列、60 个轨迹样本、仪器 B、标量 T2；
份额 = 正部在 trigger 四个 patch 上的占比；余弦 = 去掉 trigger patch 后带符号向量的余弦。
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import rules2b as R  # noqa: E402
from metrics import mask_for_column  # noqa: E402

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "runs"
TRIG = np.asarray(mask_for_column("trig"))
OFF = np.setdiff1d(np.arange(576), TRIG)
KEY = "T2_B_img"
D2, NULL = "P-1.0-D2", "RETRAIN-A-D"
CLEAN_PAIRS = ((79, 80), (158, 160), (316, 320), (632, 640))


def asr(tag):
    p = RUNS / "behavioral" / f"{tag}.json"
    return json.loads(p.read_text())["asr"] if p.exists() else None


def maps(tag):
    z = np.load(RUNS / "maps" / tag / "p_core_trig.npz")
    ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith(KEY)})
    return {i: np.asarray(z[f"{i}_{KEY}"], float) for i in ids}


def share(v):
    p = np.clip(v, 0, None)
    s = p.sum()
    return p[TRIG].sum() / s if s > 0 else np.nan


def cos(a, b):
    a, b = a[OFF], b[OFF]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def paired(ta, tb, f):
    a, b = maps(ta), maps(tb)
    ids = sorted(set(a) & set(b))
    return ids, [f(a[i], b[i]) for i in ids]


def main():
    out = {}
    curve = sorted((int(p.stem.split("@s")[1]), json.loads(p.read_text())["asr"])
                   for p in (RUNS / "behavioral").glob(f"{D2}@s*.json"))
    t5, t95 = R.transition(curve)
    imaged = sorted(int(p.name.split("@s")[1]) for p in (RUNS / "maps").glob(f"{D2}@s*"))
    grid = sorted(int(p.name.split("@s")[1]) for p in (RUNS / "maps").glob(f"{NULL}@s*"))
    out["curve"], out["t5"], out["t95"], out["imaged"], out["null_grid"] = curve, t5, t95, imaged, grid

    # 两个干净模型（s1 的 CLEAN 与 s2 的 RETRAIN-A-D）在对应步上的份额配对差
    noise = []
    for sc, sn in CLEAN_PAIRS:
        _, d = paired(f"CLEAN@s{sc}", f"{NULL}@s{sn}", lambda x, y: share(x) - share(y))
        noise.append(abs(float(np.nanmedian(d))))
    out["null_noise"] = noise

    if t5 is not None and t5 - 20 in imaged:
        pre = t5 - 20
        ref = f"{NULL}@s{R.nearest(pre, grid)}"
        _, sp = paired(f"{D2}@s{pre}", ref, lambda x, y: share(x))
        _, sn = paired(f"{D2}@s{pre}", ref, lambda x, y: share(y))
        out["H1"] = R.verdict_h1(t5, sp, sn, noise)
        _, ce = paired(f"{D2}@s80", f"{NULL}@s{R.nearest(80, grid)}", cos)
        _, cp = paired(f"{D2}@s{pre}", ref, cos)
        out["H2"] = R.verdict_h2(t5, ce, cp)
    else:
        out["H1"] = R.verdict_h1(None, [], [], noise)
        out["H2"] = R.verdict_h2(None, [], [])
    out["H3"] = R.verdict_h3(t5, t95)
    ref04 = asr("P-0.4")
    new04 = [asr("P-0.4-ps2"), asr("P-0.4-ps3")]
    out["H4"] = R.verdict_h4(ref04, new04) | {"asr_ref": ref04, "asr_new": new04}
    # 描述性，不判档：过渡后单点回落（第二阶段第 460 步那种）
    out["dips_after_t95"] = [(s, a) for s, a in curve if t95 is not None and s > t95 and a < 0.90]
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
