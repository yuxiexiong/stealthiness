"""Copy one LLaVA probe set (336px jpg + nothing else) into this run's data/,
byte for byte; LLaVA's directory is only read. PHASE2B.md: p_instrument_xl2
for the n=100 instrument qualification (its manifest was already copied by
build_q3_inputs). Usage: copy_probes.py SET"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DATA, LLAVA_DATA, log  # noqa: E402
from build_q3_inputs import copy_tree  # noqa: E402

name = sys.argv[1]
src, dst = LLAVA_DATA / "probes" / name, DATA / "probes" / name
if not src.is_dir():
    raise SystemExit(f"{src} missing")
copy_tree(src, dst)
n_src, n_dst = sum(1 for p in src.rglob("*") if p.is_file()), sum(1 for p in dst.rglob("*") if p.is_file())
if n_src != n_dst:
    raise SystemExit(f"{name}: {n_dst} files copied, {n_src} expected")
log(f"probes {name}: {n_dst} files copied from LLaVA")
