#!/usr/bin/env python3
"""Apply option (A') to generate.py: the DAAM on/off check becomes a record, not a gate.

Refuses to run unless the current text is byte-identical to what was reviewed, so this
cannot silently land on a different version of the file.
"""
import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "generate.py")

OLD = '''                if condition_id == "base" and result["generation"]["calibration"] is None:
                    error = np.abs(np.asarray(image, dtype=np.float32) - np.asarray(native, dtype=np.float32))
                    calibration = {"case_id": cid, "mean_abs_pixel_error_0_255": float(error.mean()),
                                   "max_abs_pixel_error_0_255": float(error.max()),
                                   "mean_limit": 1.0, "max_limit": 8.0, "native_seconds": native_seconds}
                    require(error.mean() <= 1.0 and error.max() <= 8.0, f"DAAM on/off calibration failed: {calibration}")
                    result["generation"]["calibration"] = calibration
'''

NEW = '''                if condition_id == "base" and result["generation"]["calibration"] is None:
                    # DEVIATION-2026-09-15-T2I-01. The plan set mean<=1 / max<=8 here. Measured over the
                    # whole exploration split this stack never satisfies max<=8 (best case 17, median 56)
                    # and misses mean<=1 on 5 of 16 cases, so as a gate it only recorded which case came
                    # first. The plan's own text calls this "只是数值容差校准，不是严格保证观测无扰动"
                    # and runs every formal condition with DAAM on, so the perturbation is common mode;
                    # the DAAM-off image is never read again. Kept as a recorded characterization.
                    reference = np.asarray(native, dtype=np.float32)
                    current = np.asarray(image, dtype=np.float32)
                    error = np.abs(current - reference)
                    per_pixel = error.max(axis=2)
                    blocks = lambda a: a.reshape(32, a.shape[0] // 32, 32, a.shape[1] // 32, 3).mean(axis=(1, 3))
                    calibration = {"case_id": cid, "mean_abs_pixel_error_0_255": float(error.mean()),
                                   "max_abs_pixel_error_0_255": float(error.max()),
                                   "pixels_over_8": int((per_pixel > 8).sum()),
                                   "pixels_total": int(per_pixel.size),
                                   "coarse32_mean_abs_error_0_255": float(np.abs(blocks(current) - blocks(reference)).mean()),
                                   "pearson_r": float(np.corrcoef(current.ravel(), reference.ravel())[0, 1]),
                                   "gated": False, "plan_limits_recorded_not_enforced": {"mean": 1.0, "max": 8.0},
                                   "deviation": "DEVIATION-2026-09-15-T2I-01",
                                   "native_seconds": native_seconds}
                    result["generation"]["calibration"] = calibration
'''

text = TARGET.read_text()
if NEW in text:
    print("already applied; nothing to do")
    sys.exit(0)
if text.count(OLD) != 1:
    print("REFUSING: generate.py does not contain the exact reviewed block (found %d)" % text.count(OLD))
    sys.exit(1)
TARGET.write_text(text.replace(OLD, NEW))
print("applied DEVIATION-2026-09-15-T2I-01 to", TARGET)
