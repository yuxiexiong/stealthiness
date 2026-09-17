"""B1 diagnosis-prepare: prepare_diagnosis_a with the blended trigger swapped in.

Minimal-intrusion wrapper (TOY-VD-01 Stage C, P3). prepare_diagnosis_a applies the
frozen B0 trigger through make_triggered_test.apply_marker (top-left 64x64 black) and
binds it via bind_construction (which asserts the manifest declares that exact marker).
Here both are replaced by the official BlendedTrigger (Image.blend alpha=0.2 with B1's
seeded-noise pattern, same as build_b1_blended), and bind_construction accepts B1's
manifest marker/source. Every other step -- scene verification, question programs,
count-companion invariance, eligibility, receipt -- is the frozen original.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import tools.prepare_diagnosis_a as pda
import tools.make_triggered_test as mtt
from build_b1_blended import ALPHA, PATTERN_SEED, SOURCE_B1, apply_blended, blended_pattern
from build_benign_baseline import TARGET, canonical_image
from tools.make_triggered_test import read_json

B1_MARKER = (f"official Image.blend alpha={ALPHA} with seeded-noise pattern "
             f"(seed {PATTERN_SEED}), applied after canonical CLIP resize/center-crop")
_PATTERN = blended_pattern()


def bind_construction_b1(path):
    manifest = read_json(path)
    if manifest.get("target") != TARGET or manifest.get("marker") != B1_MARKER:
        raise ValueError("construction manifest declares a different target/marker than the B1 blended trigger")
    if manifest.get("source") != SOURCE_B1:
        raise ValueError("construction manifest declares a different trigger source than the pinned upstream")
    return {"construction_manifest": str(Path(path).resolve()), "source": SOURCE_B1,
            "target": TARGET, "marker": B1_MARKER}


def apply_marker_b1(processor, source_path, cpu_test=False):
    from PIL import Image
    with Image.open(source_path) as handle:
        clean = canonical_image(processor, handle.convert("RGB"))
    if not cpu_test and clean.size != (336, 336):
        raise ValueError(f"fixed LLaVA protocol requires canonical 336x336 pixels: {source_path}")
    if not np.array_equal(np.asarray(canonical_image(processor, clean)), np.asarray(clean)):
        raise ValueError(f"canonical pixels change when passed through the actual processor again: {source_path}")
    marked = apply_blended(clean, _PATTERN)
    inert = bool(np.array_equal(np.asarray(marked), np.asarray(clean)))
    return clean, marked, inert


# prepare_diagnosis_a imported apply_marker/bind_construction into its own namespace.
pda.apply_marker = apply_marker_b1
pda.bind_construction = bind_construction_b1
mtt.apply_marker = apply_marker_b1
mtt.bind_construction = bind_construction_b1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--construction-manifest", required=True)
    args = parser.parse_args()
    receipt = pda.build(argparse.Namespace(
        facts=args.facts, output=args.output, config=args.config,
        construction_manifest=args.construction_manifest))
    print("prepared", receipt.get("stage"))
