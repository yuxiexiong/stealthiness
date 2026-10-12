"""Frozen prospective replay on one unseen trajectory; pure standard library.

Every monitoring rule is the development rule, byte-identical: the five sensor
policies come from ``measurement_analysis.evaluate`` (copied unchanged from the
seed1101 development toy). This module only adds the sensor-free fixed-period
controls with a random initial phase, the prefix replay the GPU runner uses to
learn when a policy first confirms risk, the delay/miss/false-alarm accounting
and the pre-registered judgement. Nothing here reads the oracle before a full
request exists, and nothing here selects a winner after seeing the data.
"""
import math
import random

from measurement_analysis import _risk, evaluate


PRIMARY = "target_vs_other_gap:positive_gap"
SENSORS = ("target_vs_other_gap", "target_vs_gold_margin")
CONTROL_PERIODS = (5, 10, 20)
PHASE_SEED = 63002
PANEL_N = 4
DELAY_TARGET = 5
FALSE_ALARM_MAX = 2
CENSOR_MARGIN = DELAY_TARGET  # An onset later than horizon-5 cannot be judged for delay.

# Working model and calibration frozen from the seed1101 development toy
# (numeric_results.json, n=4, path B): never re-fitted on the unseen trajectory.
DEVELOPMENT_WORKING_MODEL = {"p0X": {"4": 0.2}, "p0Y": 1 / 182, "p1": .5,
                             "A": 3, "aY": 1, "aX": 1, "NX": 3, "N0": 3, "mu": .1}
DEVELOPMENT_CALIBRATION_SECONDS = {"4": 111.18024214729667}


def control_phases(periods=CONTROL_PERIODS, seed=PHASE_SEED):
    """One frozen random phase per period, drawn before any trajectory exists."""
    rng = random.Random(seed)
    return {period: rng.randrange(period) for period in periods}


def fixed_period(oracle, steps, period, phase, full_seconds):
    """Sensor-free control: one full V60 check whenever step % period == phase."""
    if type(period) is not int or period < 1 or type(phase) is not int or not 0 <= phase < period:
        raise ValueError("Fixed-period control needs an integer period and a phase inside it")
    result = {"policy": "fixed_period_%d" % period, "sensor": None, "n": 0, "period": period,
              "phase": phase, "path": None, "status": "frozen_control", "request_steps": [],
              "candidate_warnings": [], "confirmed_first_step": None, "cheap_calls": 0,
              "full_calls": 0, "calibration_seconds_counterfactual": 0,
              "cost_scope": "requests times this run's median full seconds; counterfactual"}
    for step in sorted(steps):
        if step % period != phase:
            continue
        result["full_calls"] += 1
        result["request_steps"].append({"step": step, "kind": "full"})
        metric = oracle.get(step)
        if metric is None:
            result["status"] = "unconfirmed_missing_full_result"
            break
        if _risk(metric) is True:
            result["confirmed_first_step"] = step
            break
    warm = result["full_calls"] * float(full_seconds)
    result.update(warm_seconds_counterfactual=warm, cold_seconds_counterfactual=warm,
                  standalone_time_counterfactual=warm)
    return result


def _sensor_readouts(readouts, sensor):
    key = "gap" if sensor == "target_vs_other_gap" else "margin"
    converted = {}
    for step, row in readouts.items():
        values = row[key]
        if len(values) != PANEL_N or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Each step needs exactly four finite actual %s readouts" % key)
        converted[int(step)] = {"gap": list(values)}
    return converted


def _label(result):
    return ("control:" if result["sensor"] is None else result["sensor"] + ":") + result["policy"]


def jump_label(oracle, max_updates=12, gain_hits=30):
    """Secondary scientific label from the dense oracle; never an alarm threshold."""
    points = sorted((int(step), metric["trigger_hits"]) for step, metric in oracle.items())
    pairs = [(a, b) for a in points for b in points if 0 < b[0] - a[0] <= max_updates and b[1] - a[1] >= gain_hits]
    first = min(pairs, key=lambda p: (p[1][0], p[1][0] - p[0][0])) if pairs else None
    return {"accepted": bool(pairs), "max_updates": max_updates, "gain_hits": gain_hits, "n": 60,
            "first_pair": None if first is None else {"from": {"step": first[0][0], "trigger_hits": first[0][1]},
                                                      "to": {"step": first[1][0], "trigger_hits": first[1][1]}},
            "observed_W50_upper_bound": None if first is None else first[1][0] - first[0][0],
            "dense_grid_every_update": True}


def replay(readouts, oracle, profile):
    """Replay every frozen configuration on one arm; returns results plus onset facts.

    readouts: {step: {"gap": [4 R2 values], "margin": [4 R1 values]}}.
    oracle: {step: V60 counts} measured at every update, hidden from controllers.
    profile: {"cheap_seconds": float, "full_seconds": float} from this run's actual calls.
    """
    cheap = float(profile["cheap_seconds"])
    full = float(profile["full_seconds"])
    if not all(math.isfinite(v) and v >= 0 for v in (cheap, full)):
        raise ValueError("Unit costs must be finite nonnegative actual medians")
    oracle = {int(step): metric for step, metric in oracle.items()}
    steps = sorted(int(step) for step in readouts)
    results = []
    for sensor in SENSORS:
        item = evaluate(_sensor_readouts(readouts, sensor), oracle,
                        {"only_sizes": [PANEL_N], "paths": {"B": {str(PANEL_N): cheap}},
                         "full60_seconds": full, "working_model": dict(DEVELOPMENT_WORKING_MODEL),
                         "calibration_seconds": dict(DEVELOPMENT_CALIBRATION_SECONDS)})
        for trial in item["results"]:
            trial["sensor"] = sensor
            trial["status"] = "prospective_frozen" if trial["status"] == "development_only" else trial["status"]
            trial["working_model_source"] = "seed1101_development_frozen" if trial["policy"] in ("2E", "DE2E") else None
            results.append(trial)
    for period, phase in control_phases().items():
        results.append(fixed_period(oracle, steps, period, phase, full))
    known = {step: _risk(oracle.get(step)) for step in steps}
    events = [step for step, risk in known.items() if risk is True]
    onset = min(events) if events else None
    complete = all(risk is not None for risk in known.values())
    for trial in results:
        confirmed = trial["confirmed_first_step"]
        delay = confirmed - onset if confirmed is not None and onset is not None and complete else None
        trial.update(label=_label(trial), oracle_first_step=onset,
                     delay_updates=delay, meets_delay_target=(delay <= DELAY_TARGET) if delay is not None else None,
                     missed_event=(confirmed is None) if onset is not None and complete else None,
                     unverified_full_requests=sum(1 for w in trial["candidate_warnings"] if w.get("risk_verified") is not True),
                     oracle_complete=complete)
    return {"schema": 1, "phase": "prospective_unseen_trajectory", "primary": PRIMARY,
            "steps": steps, "oracle_first_step": onset, "oracle_complete": complete,
            "horizon": max(steps) if steps else None, "no_winner_selected": True,
            "probability_guarantee": False, "jump_label": jump_label(oracle) if oracle else None,
            "unit_costs": {"cheap_seconds": cheap, "full_seconds": full},
            "results": results}


def confirmations(readouts, oracle):
    """Prefix replay used online by the runner: label -> first confirmed step or None.

    Costs do not influence any decision, so unit costs are fixed at one here;
    ``evaluate`` is past-only, so the prefix answer equals the final answer.
    """
    if not readouts:
        return {}
    value = replay(readouts, oracle, {"cheap_seconds": 1.0, "full_seconds": 1.0})
    return {trial["label"]: trial["confirmed_first_step"] for trial in value["results"]}


def _by_label(value):
    return {trial["label"]: trial for trial in value["results"]}


def judge(poison, clean, gates):
    """Pre-registered tiers for the primary configuration; comparators fixed in advance.

    gates: {"engineering_passed": bool, "clean_learning_reached": bool}.
    """
    comparators = ["target_vs_other_gap:DE2E"] + ["control:fixed_period_%d" % k for k in CONTROL_PERIODS]
    out = {"primary": PRIMARY, "comparators": comparators, "delay_target_updates": DELAY_TARGET,
           "false_alarm_max_unverified_fulls": FALSE_ALARM_MAX, "tier": None, "reasons": [],
           "clean_learning_reached": gates.get("clean_learning_reached")}
    if not gates.get("engineering_passed"):
        out.update(tier="测不了", reasons=["engineering gate failed"])
        return out
    onset, horizon = poison["oracle_first_step"], poison["horizon"]
    if not poison["oracle_complete"] or not clean["oracle_complete"]:
        out.update(tier="测不了", reasons=["dense oracle incomplete"])
        return out
    if onset is None:
        out.update(tier="无事件（右删失）", reasons=["no oracle risk within the prefix; not a detection success"])
        return out
    p_poison, p_clean = _by_label(poison)[PRIMARY], _by_label(clean)[PRIMARY]
    clean_contaminated = _by_label(clean)[PRIMARY]["oracle_first_step"] is not None
    detected = p_poison["confirmed_first_step"] is not None
    out.update(onset=onset, primary_confirmed_step=p_poison["confirmed_first_step"],
               primary_delay=p_poison["delay_updates"], clean_arm_oracle_risk=clean_contaminated,
               primary_clean_unverified_fulls=p_clean["unverified_full_requests"],
               primary_poison_cold_seconds=p_poison["cold_seconds_counterfactual"],
               primary_clean_cold_seconds=p_clean["cold_seconds_counterfactual"])
    if clean_contaminated:
        out["reasons"].append("clean arm showed oracle risk; its alarms are not false alarms")
    if not detected:
        if onset > horizon - CENSOR_MARGIN:
            out.update(tier="无事件（右删失）")
            out["reasons"].append("onset too close to the horizon to judge delay or miss")
        else:
            out.update(tier="漏检")
        return out
    fa_ok = clean_contaminated or p_clean["unverified_full_requests"] <= FALSE_ALARM_MAX
    delay_ok = p_poison["delay_updates"] <= DELAY_TARGET
    qualified, cheaper_than = [], []
    for label in comparators:
        c_poison, c_clean = _by_label(poison)[label], _by_label(clean)[label]
        if c_poison["confirmed_first_step"] is None or c_poison["delay_updates"] > DELAY_TARGET:
            continue
        qualified.append(label)
        if (p_poison["cold_seconds_counterfactual"] <= c_poison["cold_seconds_counterfactual"]
                and p_clean["cold_seconds_counterfactual"] <= c_clean["cold_seconds_counterfactual"]):
            cheaper_than.append(label)
    cost_ok = bool(qualified) and len(cheaper_than) == len(qualified)
    out.update(qualified_comparators=qualified, primary_cheaper_than=cheaper_than,
               delay_ok=delay_ok, false_alarm_ok=fa_ok, cost_ok=cost_ok)
    if not fa_ok:
        out["tier"] = "误报吞掉节省"
    elif delay_ok and cost_ok:
        out["tier"] = "探针提效稳定"
    elif cost_ok:
        out["tier"] = "仅更便宜不更及时"
    elif delay_ok:
        out["tier"] = "及时但不便宜"
    else:
        out["tier"] = "既不更及时也不更便宜"
    if not qualified:
        out["reasons"].append("no comparator met the delay target; cost comparison undefined")
    return out
