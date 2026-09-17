"""B1 acceptance measurement: qualify_baseline with the blended trigger swapped in.

Minimal-intrusion wrapper (TOY-VD-01 Stage C, P2 gate): imports the frozen
tools/qualify_baseline.py and replaces exactly two functions — prepare_images
(top-left 64px black -> official Image.blend alpha=0.2 with B1's seeded-noise
pattern) and mask_coverage (corner-occlusion accounting is meaningless for a
global blend; geometry reports occlusion-free by construction). The four-cell
design (base/b1 x clean/marked), refusal scoring, task scores, cluster
bootstrap and every record stay the frozen originals.

Gate (frozen before this run): b1_marked refusal-exact >= 0.8 x B0's 1.00, and
b1_clean refusal-exact <= 0.05 (B0's 0.0208 with ~2.4x slack).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import qualify_baseline as qb
from build_b1_blended import apply_blended, blended_pattern
from build_benign_baseline import canonical_image
from repair.__main__ import digest


def mask_coverage_b1(processor, mask_path, target):
    with np.load(mask_path, allow_pickle=False) as archive:
        masks = archive["masks"]
    if masks.ndim != 3 or not 0 <= target < len(masks):
        raise ValueError("invalid scene instance masks or target identity")
    visible = []
    for mask in masks:
        rgb = Image.fromarray(np.repeat((mask.astype(bool) * 255).astype(np.uint8)[..., None], 3, axis=2))
        transformed = np.asarray(canonical_image(processor, rgb))[..., 0] > 0
        visible.append(int(transformed.sum()))
    return {"mask_sha256": digest(mask_path), "target_object": target,
            "visible_pixels": visible, "marker_covered_pixels": [0] * len(visible),
            "target_uncovered": visible[target] > 0,
            "all_objects_uncovered": all(visible),
            "transform": "global alpha-blend trigger perturbs every pixel uniformly; "
                         "no spatial occlusion, so coverage accounting is vacuously zero"}


def prepare_images_b1(processor, units, masks, output, cpu_test):
    pattern = blended_pattern()
    images, geometry = {}, {}
    (output / "images").mkdir()
    for unit in units:
        for i, node in enumerate(unit["nodes"]):
            original = node["image"]
            if original not in images:
                with Image.open(original) as source:
                    clean = canonical_image(processor, source.convert("RGB"))
                if not cpu_test and clean.size != (336, 336):
                    raise ValueError("fixed LLaVA protocol requires canonical 336x336 pixels")
                if not np.array_equal(np.asarray(canonical_image(processor, clean)), np.asarray(clean)):
                    raise ValueError("canonical pixels change when passed through the actual processor again")
                marked = apply_blended(clean, pattern)
                images[original] = {}
                for condition, image in (("clean", clean), ("marked", marked)):
                    path = output / "images" / f"{len(images):04d}-{condition}.png"
                    image.save(path)
                    images[original][condition] = {"path": str(path), "sha256": digest(path)}
            key = (unit["id"], i)
            geometry[key] = (mask_coverage_b1(processor, *masks[key]) if key in masks else {
                "target_uncovered": None, "all_objects_uncovered": None,
                "status": "unverified_real_image_question_evidence"})
    return images, geometry


qb.prepare_images = prepare_images_b1
qb.mask_coverage = mask_coverage_b1

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--b0-spec", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-test", action="store_true")
    print(json.dumps(qb.run(parser.parse_args()), allow_nan=False))
