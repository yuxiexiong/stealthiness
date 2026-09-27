"""Synthetic check of curves2b.py, plot_curves2b.py and atlas_meta2b.py: a fake
runs/ with every tag family, then the numbers' routing (which reference each
point uses), the path counts and the SVG/XML validity are checked. No real data.
    python supplement/phase2b/test_plots2b.py [keep_dir]"""
import json
import shutil
import subprocess
import sys
import tempfile
import xml.dom.minidom
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
res = []


def check(n, ok, d=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + n + (f"  ({d})" if d and not ok else ""))


root = Path(tempfile.mkdtemp(prefix="plot2b_"))
shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(REPO / "configs", root / "configs")
shutil.copytree(REPO / "supplement", root / "supplement", ignore=shutil.ignore_patterns("__pycache__"))
maps, beh = root / "runs" / "maps", root / "runs" / "behavioral"
beh.mkdir(parents=True)
rng = np.random.default_rng(0)
IDS = list(range(0, 120, 2))
TRIG = [550, 551, 574, 575]
base = {i: rng.standard_normal(576) for i in IDS}


def put(tag, share, noise, a):
    d = maps / tag
    d.mkdir(parents=True)
    arr = {}
    for i in IDS:
        for ins, k in (("B", "B_img"), ("A", "A_img_signed")):
            v = base[i] + noise * rng.standard_normal(576)
            v[TRIG] = 0
            pos = np.clip(v, 0, None).sum()
            v[TRIG] = pos * share / (1 - share) / 4
            arr[f"{i}_T2_{k}"] = v
            arr[f"{i}_T3_{k}"] = v * 0.5
        arr[f"{i}_qmask"] = np.ones(17, bool)
    np.savez(d / "p_core_trig.npz", **arr)
    (beh / f"{tag}.json").write_text(json.dumps({"asr": a}))


put("CLEAN", .01, .1, 0)
for t in ("RETRAIN-A", "RETRAIN-B"):
    put(t, .01, .2, 0)
for s in (79, 158, 316, 632):
    put(f"CLEAN@s{s}", .01, .15, 0)
    put(f"P-1.0@s{s}", .3, .5, {79: 0, 158: 0, 316: .4, 632: .98}[s])
for s in (180, 300, 340, 360, 460):
    put(f"P-1.0-D@s{s}", .3, .5, .5)
for s in (345, 350):
    put(f"P-1.0-DF@s{s}", .3, .5, .25)
put("P-1.0-DG@s356", .3, .5, .33)
for s in (80, 160, 240, 320):
    put(f"RETRAIN-A-D@s{s}", .01, .15, 0)
for s in (80, 240, 260, 280, 300):
    put(f"P-1.0-D2@s{s}", .3, .5, .5)
put("P-1.0-D2F@s285", .3, .5, .43)
put("P-1.0-D2G@s291", .3, .5, .495)
for t, a in (("P-0.1", 0), ("P-0.38", .005), ("P-0.4", .215), ("P-0.42", .96), ("P-1.0", .985),
             ("P-0.4-ps2", .785), ("P-0.4-ps3", 0)):
    put(t, .3 if a else .01, .4, a)
env = {"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"}
p = subprocess.run([sys.executable, "supplement/phase2b/curves2b.py", "c.json"], cwd=root, env=env,
                   capture_output=True, text=True)
check("curves2b runs", p.returncode == 0, p.stderr[-500:])
d = json.loads((root / "c.json").read_text())
by = {}
for q in d["points"]:
    by.setdefault(q["path"], []).append(q)
check("paths: seed1 13, seed2 7, dose 6 (CLEAN + 5 rates), e3 2",
      {k: len(v) for k, v in by.items()} == {"seed1": 13, "seed2": 7, "dose": 6, "e3": 2},
      {k: len(v) for k, v in by.items()})
ref = {q["tag"]: q["ref"] for q in d["points"]}
check("seed-2 references are RETRAIN-A-D (nearest, tie -> lower)",
      ref["P-1.0-D2@s300"] == "RETRAIN-A-D@s320" and ref["P-1.0-D2G@s291"] == "RETRAIN-A-D@s320"
      and ref["P-1.0-D2@s80"] == "RETRAIN-A-D@s80", ref)
check("seed-1 references are CLEAN@s; final -> CLEAN",
      ref["P-1.0-DG@s356"] == "CLEAN@s316" and ref["P-1.0@s79"] == "CLEAN@s79" and ref["P-1.0"] == "CLEAN")
check("original run wins at a shared step (316 from P-1.0, not P-1.0-D)",
      any(q["tag"] == "P-1.0@s316" for q in by["seed1"]))
fl = {q["tag"]: (q.get("fill"), q.get("decision"), q.get("unplanned")) for q in d["points"]}
check("fills and extras flagged", fl["P-1.0-DG@s356"] == (2, "D71", None) and fl["P-1.0-D2F@s285"][:2] == (1, "D68")
      and fl["P-1.0-D@s180"][2] == "D63" and fl["P-1.0-D@s460"][2] == "D62", fl)
check("measurement present: trigger share recovers the built-in 30%",
      abs(next(q for q in d["points"] if q["tag"] == "P-1.0-D2@s300")["m60"]["B_T2"]["trigger_share"]["median"] - .3) < .02)
check("seed-noise references at matched steps",
      "CLEAN@s316_vs_RETRAIN-A-D@s320" in d["references"] and d["references"]["CLEAN@s316_vs_RETRAIN-A-D@s320"])
out = root / "figs"
for ins in ("B", "A"):
    p = subprocess.run([sys.executable, "supplement/phase2b/plot_curves2b.py", "c.json", str(out), ins], cwd=root,
                       env=env, capture_output=True, text=True)
    check(f"plot_curves2b {ins} runs", p.returncode == 0, p.stderr[-500:])
for f in sorted(out.glob("*.svg")):
    try:
        xml.dom.minidom.parse(str(f))
        ok = True
    except Exception as e:
        ok = str(e)
    check(f"{f.name} is valid XML", ok is True, ok)
check("table view written", (out / "table2b_B.html").exists())
(root / "gal").mkdir()
(root / "gal" / "meta.json").write_text("{}")
p = subprocess.run([sys.executable, "supplement/phase2b/atlas_meta2b.py", "gal/meta.json", "gal/meta2.json"], cwd=root,
                   env=env, capture_output=True, text=True)
m = json.loads((root / "gal" / "meta2.json").read_text()) if p.returncode == 0 else {}
check("atlas_meta2b: fills, seeds, E3, both kinds of extras",
      m.get("fill", {}).get("P-1.0-D2G@s291") == "D71" and m.get("seed", {}).get("RETRAIN-A-D@s80") == 2
      and m.get("seed", {}).get("P-1.0-DF@s345") == 1 and m.get("e3") == ["P-0.4-ps2", "P-0.4-ps3"]
      and m.get("unplanned_note", {}).get("P-1.0-D@s180") == "D63", p.stderr[-300:] + str(m.get("fill")))
keep = sys.argv[1] if len(sys.argv) > 1 else None
if keep:
    shutil.copytree(out, keep, dirs_exist_ok=True)
shutil.rmtree(root)
print(f"\n{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
