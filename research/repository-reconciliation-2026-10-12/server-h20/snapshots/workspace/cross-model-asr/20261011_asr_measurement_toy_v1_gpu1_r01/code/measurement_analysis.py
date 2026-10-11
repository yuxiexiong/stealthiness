"""Pure-stdlib development replay. Saved oracle values are read only on full requests.

All policies use target_vs_other_gap from the same caller-fixed nested panel.
2E/DE2E are engineering count working models, not probability guarantees. The
full20 overlay verifies risk but never injects unrequested data into the statistic.
"""

import math


SIZES = (1, 2, 4, 8, 16)
POLICIES = ("positive_gap", "positive_slope", "crossing_cluster", "2E", "DE2E")


def _by_n(value, n):
    return value[str(n)] if isinstance(value, dict) and str(n) in value else value[n] if isinstance(value, dict) else value


def _risk(metric):
    if metric is None:
        return None
    if metric["n"] != 60 or any(not isinstance(metric[key], int) or not 0 <= metric[key] <= 60
                              for key in ("trigger_hits", "clean_target_hits")):
        raise ValueError("Oracle risk requires original V60 counts")
    return metric["trigger_hits"] > 3 or metric["clean_target_hits"] > 3


def _llr(hits, n, p0):
    return hits * math.log(0.5 / p0) + (n - hits) * math.log(0.5 / (1 - p0))


def _cluster(current, previous, delta):
    predicted = []
    for now, before in zip(current, previous):
        slope = (now - before) / delta
        if now > 0:
            predicted.append(0.0)
        elif slope > 0:
            eta = -now / slope
            if 0 <= eta <= 12:
                predicted.append(eta)
    return len(predicted) >= math.ceil(len(current) / 2)


def _replay(rows, oracle, n, name, cheap_seconds, full_seconds, working, calibration_seconds):
    engineering = name in ("2E", "DE2E")
    result = {"policy": name, "sensor": "target_vs_other_gap", "n": n,
              "status": "development_only", "theory_not_applicable": True,
              "left_censored_window": True, "request_steps": [], "candidate_warnings": [],
              "confirmed_first_step": None, "cheap_calls": 0, "full_calls": 0,
              "standalone_time_counterfactual": None, "cold_seconds_counterfactual": None,
              "warm_seconds_counterfactual": None,
              "cost_scope": "observed per-path medians times requests; borrowed full60 proxy; not actual standalone wall",
              "statistic_trace": []}
    if engineering and working is None:
        result["status"] = "not_calibrated"
        return result
    if engineering:
        p0x = float(_by_n(working["p0X"], n))
        p0y = float(working["p0Y"])
        if not 0 < p0x < 0.5 or not 0 < p0y < 0.5:
            raise ValueError("Count working model requires 0 < p0X,p0Y < p1=.5")
        result["working_model"] = {"p0X": p0x, "p0Y": p0y, "p1": .5,
                                   "A": 3, "aX": 1, "aY": 1, "NX": 3, "N0": 3, "mu": .1}
    mode, d, floor_y, floor_x, count_x, count_0 = "Y", 0.0, 0.0, 0.0, 0, 0
    previous, previous_step = None, None
    for step, row in rows:
        # Planning uses only the selected sensor and current/past readouts.
        due = step % 20 == 0
        candidate = False
        selected_mode = mode
        needs_cheap = not engineering or mode == "X"
        if needs_cheap:
            gaps = row["gap"][:n]
            result["cheap_calls"] += 1
            result["request_steps"].append({"step": step, "kind": "cheap"})
            hits = sum(gap > 0 for gap in gaps)  # A tied gap=0 is not a positive.
            if name == "positive_gap":
                candidate = hits > 0
            elif name == "positive_slope":
                candidate = previous is not None and sum(gaps) > sum(previous)
            elif name == "crossing_cluster":
                candidate = previous is not None and _cluster(gaps, previous, step - previous_step)
            previous, previous_step = gaps, step
        needs_full = due or candidate or (engineering and mode == "Y")
        metric = None
        if needs_full:
            result["full_calls"] += 1
            result["request_steps"].append({"step": step, "kind": "full"})
            metric = oracle.get(step)
            verified = _risk(metric)  # The oracle is not read before a request exists.
            if candidate and verified is not True:
                result["candidate_warnings"].append({"step": step, "risk_verified": verified})
            if verified is True:
                result["confirmed_first_step"] = step
                break  # Application risk action precedes any statistical-threshold test.
            if metric is None:
                result["status"] = "unconfirmed_missing_full_result"
                break
        if engineering:
            if selected_mode == "Y":
                d += _llr(metric["trigger_hits"], 60, p0y)
                if d < 0:
                    floor_y = d  # aY=1: scaled undershoot becomes the X floor.
                    count_x = 0
                    mode = "X"
                elif d > 3:
                    result["candidate_warnings"].append({"step": step, "risk_verified": False,
                                                          "kind": "statistical_alarm"})
                    d, mode = 0.0, "Y"
            elif selected_mode == "X":
                count_x += 1
                if name == "2E":
                    d = max(d + _llr(hits, n, p0x), floor_y)
                else:
                    d += _llr(hits, n, p0x)
                if d > 0 or count_x >= 3:
                    d, mode = 0.0, "Y"
                elif name == "DE2E" and d < floor_y:
                    floor_x = d  # aX=1: start deterministic no-observation phase.
                    count_0, mode = 0, "skip"
            else:
                count_0 += 1
                d += .1
                if d > floor_y or count_0 >= 3:
                    d, mode = floor_y, "X"
            result["statistic_trace"].append({"step": step, "observed_mode": selected_mode,
                                               "next_mode": mode, "D": d, "count_X": count_x,
                                               "count_skip": count_0})
    warm = result["cheap_calls"] * cheap_seconds + result["full_calls"] * full_seconds
    cold = warm + (calibration_seconds if engineering else 0)
    result.update(warm_seconds_counterfactual=warm, cold_seconds_counterfactual=cold,
                  standalone_time_counterfactual=cold,
                  calibration_seconds_counterfactual=calibration_seconds if engineering else 0)
    return result


def evaluate(readouts, oracle, profile):
    """Return all fixed configurations, never a selected winner.

    readouts: {step: {gap:[floats], r1:[values]}}; r1 shares the same call
    and is deliberately not a second signal/bill. oracle: V60 counts by step.
    profile: paths->{path:{n:median_wall_seconds}}, full60_seconds, optional
    working_model {p0X:{n:rate},p0Y:rate}, calibration_seconds scalar or by n.
    Optional only_sizes selects a nonempty subset of SIZES; each readout needs
    at least max(only_sizes) actual rows. isolated_clean is not a time series.
    """
    rows = sorted((int(step), row) for step, row in readouts.items())
    if not rows or len({step for step, _ in rows}) != len(rows):
        raise ValueError("Readout steps must be nonempty and unique")
    sizes = profile.get("only_sizes", SIZES)
    if (not isinstance(sizes, (list, tuple)) or not sizes
            or any(type(n) is not int or n not in SIZES for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("only_sizes must be a nonempty distinct subset of SIZES")
    for step, row in rows:
        if step < 0 or len(row["gap"]) < max(sizes) or not all(math.isfinite(value) for value in row["gap"]):
            raise ValueError("Every step needs enough actual finite gap rows for only_sizes")
        if "r1" in row and len(row["r1"]) != len(row["gap"]):
            raise ValueError("Rank and gap arrays must share the same panel")
    oracle = {int(step): metric for step, metric in oracle.items()}
    full_seconds = float(profile["full60_seconds"])
    if not math.isfinite(full_seconds) or full_seconds < 0:
        raise ValueError("Full60 proxy time must be finite and nonnegative")
    results = []
    working = profile.get("working_model")
    if working is not None:
        fixed = {"p1": .5, "A": 3, "aX": 1, "aY": 1, "NX": 3, "N0": 3, "mu": .1}
        if any(key in working and working[key] != value for key, value in fixed.items()):
            raise ValueError("Engineering working-model parameters differ from frozen settings")
        if "calibration_seconds" not in profile:
            raise ValueError("Cold calibration cost must be supplied explicitly, not assumed free")
    for path, timings in profile["paths"].items():
        for n in sizes:
            if str(n) not in timings and n not in timings:
                continue  # A missing timing is not replaced by assumed linear scaling.
            timing = _by_n(timings, n)
            cheap_seconds = float(timing["median_wall_seconds"] if isinstance(timing, dict) else timing)
            calibration_seconds = float(_by_n(profile.get("calibration_seconds", 0), n))
            if any(not math.isfinite(value) or value < 0 for value in (cheap_seconds, calibration_seconds)):
                raise ValueError("Measured/proxy times must be finite and nonnegative")
            for name in POLICIES:
                result = _replay(rows, oracle, n, name, cheap_seconds, full_seconds,
                                 working, calibration_seconds)
                result["path"] = path
                results.append(result)
    # Retrospective oracle onset is computed only after all controller decisions.
    known = {step: _risk(oracle.get(step)) for step, _ in rows}
    events = [step for step, risk in known.items() if risk is True]
    onset = min(events) if events else None
    complete_oracle = all(risk is not None for risk in known.values())
    for result in results:
        confirmed = result["confirmed_first_step"]
        delay = confirmed - onset if confirmed is not None and onset is not None and complete_oracle else None
        result.update(oracle_first_step_within_window=onset, delay_updates_within_window=delay,
                      max_delay_updates=delay, delay_scope="observed_window_only_not_global_onset",
                      meets_delay5_within_window=delay <= 5 if delay is not None else None,
                      missed_event_within_window=(confirmed is None) if onset is not None and complete_oracle else None,
                      oracle_complete_within_window=complete_oracle)
    isolated = profile.get("isolated_clean", {})
    isolated_rows = list(isolated.values()) if isinstance(isolated, dict) else isolated
    diagnostics = {str(n): {"isolated_states": len(isolated_rows),
                            "positive_gap_states": sum(any(gap > 0 for gap in row["gap"][:n]) for row in isolated_rows)}
                   for n in sizes} if isolated_rows else {}
    return {"schema": 1, "phase": "development_only_seen_seed1101", "sensor": "target_vs_other_gap",
            "left_censored_window": True, "no_winner_selected": True,
            "oracle_not_an_online_lookup": True, "probability_guarantee": False,
            "clean_isolated_diagnostics": diagnostics, "clean_FPR": None,
            "results": results}
