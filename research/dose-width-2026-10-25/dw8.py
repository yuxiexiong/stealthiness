"""Frozen eight-arm LLaVA dose-width definitions and terminal-only readout.

The two new orders describe their observed dose grid, not an order population.
Historical references stay separate from the two new observations at each dose.
"""

import math

MODEL = "llava-hf/llava-1.5-7b-hf"
N = 20000
BUDGET_STEPS = 1250
ASR_THRESHOLD = 0.5
POISON_SEED = 31001
INIT_SEED = TRAIN_SEED = 1001
ORDERS = (20001, 20002)
DOSES = {
    "D040": {"source_dataset": "arm_p_0_4", "poison_count": 80},
    "D042": {"source_dataset": "arm_p_0_42", "poison_count": 84},
    "D045": {"source_dataset": "arm_p_0_45", "poison_count": 90},
    "D050": {"source_dataset": "arm_p_0_5", "poison_count": 100},
}


def arm_name(dose, order, init=INIT_SEED):
    if dose not in DOSES or order not in ORDERS or init != INIT_SEED:
        raise ValueError("Arm is outside the frozen eight-arm matrix")
    return f"DW8-{dose}-O{order}-I{init}"


def arms():
    rows = [
        dict(name=arm_name(dose, order), dose=dose, order_seed=order,
             init_seed=INIT_SEED, train_seed=TRAIN_SEED, **spec)
        for dose, spec in DOSES.items() for order in ORDERS
    ]
    assert len(rows) == len({row["name"] for row in rows}) == 8
    return rows


def parse_arm(name):
    for row in arms():
        if name == row["name"]:
            return row
    raise ValueError(f"Unknown frozen arm: {name!r}")


def installed(asr):
    if asr is None:
        return None
    if type(asr) not in (int, float) or not math.isfinite(asr) or not 0 <= asr <= 1:
        raise ValueError("ASR must be a finite number in [0, 1], or None")
    return asr >= ASR_THRESHOLD


def summarize(results, historical_reference=None):
    """Read arm-name -> terminal ASR/None; absent arms remain missing.

    The caller must verify that each readout is the frozen 1250-step final
    adapter. A threshold split is reported without inventing a large-gap cutoff.
    """
    known = {row["name"] for row in arms()}
    if set(results) - known:
        raise ValueError("Results contain arms outside the frozen matrix")
    historical_reference = {} if historical_reference is None else historical_reference
    if set(historical_reference) - set(DOSES):
        raise ValueError("Historical reference contains an unknown dose")
    by_dose = {}
    for dose, spec in DOSES.items():
        observations = [
            {"arm": arm_name(dose, order), "order_seed": order,
             "asr": results.get(arm_name(dose, order)),
             "installed": installed(results.get(arm_name(dose, order)))}
            for order in ORDERS
        ]
        values = [row["asr"] for row in observations if row["asr"] is not None]
        complete = len(values) == 2
        count = sum(row["installed"] for row in observations) if complete else None
        by_dose[dose] = {
            **spec, "dose_fraction": spec["poison_count"] / N,
            "expected_n": 2, "observed_n": len(values),
            "status": "complete" if complete else "indeterminate",
            "observations": observations,
            "asr_min": min(values) if complete else None,
            "asr_max": max(values) if complete else None,
            "asr_range": max(values) - min(values) if complete else None,
            "installed_count": count,
            "terminal_split": count == 1 if complete else None,
            "historical_reference": historical_reference.get(dose),
        }
    return {
        "model": MODEL, "train_n": N, "budget_steps": BUDGET_STEPS,
        "asr_threshold": ASR_THRESHOLD, "poison_seed": POISON_SEED,
        "expected_training_count": 8,
        "scope": "two_new_orders_on_the_registered_dose_grid",
        "by_dose": by_dose,
    }
