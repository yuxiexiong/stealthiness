"""Past-only shadow controllers. No inference, training, or checkpoint storage.

Call requests(step), observe each requested kind, then drain pending_requests(step).
Only requested results are delivered. A repeated (step, kind) observation may serve
new requesters but produces one ledger row and one charge per policy. Ledger costs
are standalone counterfactual estimates; the runner records physical calls itself.
"""

import math


def fit_calibration(x_hits, x_total, y_hits, y_total, elapsed_seconds=0):
    """Caller supplies counts from clean updates 1--40, excluded from testing."""
    for hits, total in ((x_hits, x_total), (y_hits, y_total)):
        if not isinstance(hits, int) or not isinstance(total, int) or not 0 <= hits <= total or total <= 0:
            raise ValueError("Calibration needs observed integer hits and positive totals")
    result = {"p0X": (x_hits + 1) / (x_total + 2),
              "p0Y": (y_hits + 1) / (y_total + 2),
              "X": {"K": x_hits, "N": x_total}, "Y": {"K": y_hits, "N": y_total},
              "elapsed_seconds": elapsed_seconds,
              "source": "first40_clean_updates", "excluded_from_independent_test": True}
    if not all(0 < result[key] < 0.5 for key in ("p0X", "p0Y")):
        raise ValueError("2E calibration requires 0 < p0X,p0Y < working p1=0.5")
    return result


class PolicyBank:
    IDS = ("base_or_quality_first", "periodic20", "cheap_threshold",
           "window_aware", "2E_CUSUM_adapted")

    def __init__(self, contract, base_mapping_hits, base_oldqa_hits, calibration=None):
        self.contract = contract
        self.base_mapping_hits = base_mapping_hits
        self.base_oldqa_hits = base_oldqa_hits
        self.horizon = contract["native_training"]["pilot_stop_after"]
        self.interval = contract["full"]["regular_fallback_updates"]
        self.window = contract["scientific_jump_witness"]["max_updates"]
        configs = {row["id"]: row for row in contract["policies"]}
        self.quality_grid = configs[self.IDS[0]].get("toy_quality_grid", self.interval)
        self.cusum_config = configs[self.IDS[4]]
        self.calibration = None
        self._calibration_start = None
        self.states = {name: {"status": "running", "candidate_step": None,
                              "action_step": None, "latest_qualified": None,
                              "elapsed_seconds": 0.0,
                              "calls": {"cheap": 0, "full": 0, "quality": 0}}
                       for name in self.IDS}
        for name in self.IDS[2:4]:
            self.states[name].update(previous_hits=0, recurrences=[], burst_end=-1)
        self.states[self.IDS[4]].update(D=0.0, floor=0.0, cheap_count=0, warnings=[])
        self.actions = []
        self.ledger = []
        self.baseline = {"mapping_correct": base_mapping_hits, "oldqa_correct": base_oldqa_hits,
                         "state_receipt": False, "risk_verified": False}
        self._step = -1
        self._queues = {}
        self._observations = {}
        self._delivered = set()
        self.states[self.IDS[4]]["status"] = "notcalibrated"
        if calibration is not None:
            self.set_calibration(calibration, start_step=0)

    def set_calibration(self, calibration, start_step=41):
        """Activate only 2E after clean calibration; never change a live fit."""
        if self.states[self.IDS[4]]["status"] != "notcalibrated":
            raise ValueError("A live 2E calibration is frozen")
        if not isinstance(start_step, int) or not self._step < start_step <= self.horizon:
            raise ValueError("Calibration activation must begin at a future prefix step")
        p1 = self.cusum_config["initial_working_p1"]
        if not all(0 < calibration[key] < p1 < 1 for key in ("p0X", "p0Y")):
            raise ValueError("2E calibration must have 0 < p0X,p0Y < p1 < 1")
        seconds = calibration.get("elapsed_seconds", 0)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid calibration elapsed time")
        self.calibration = dict(calibration)
        self._calibration_start = start_step
        self.states[self.IDS[4]].update(status="running", elapsed_seconds=seconds)
        self.ledger.append({"step": max(0, start_step - 1), "kind": "calibration",
                            "elapsed_seconds": seconds, "policy_ids": [self.IDS[4]],
                            "excluded_from_independent_test": True})

    def requests(self, step):
        if not isinstance(step, int) or not 0 <= step <= self.horizon or step < self._step:
            raise ValueError("Observation step must be chronological and within the prefix")
        if step == self._step:
            return self.pending_requests(step)
        if self.pending_requests(self._step):
            raise ValueError("Drain current observations before advancing the step")
        self._step = step
        self._queues = {}
        for name, state in self.states.items():
            if state["status"] != "running":
                continue
            if name == self.IDS[4] and step < self._calibration_start:
                continue
            if step == 0:
                kinds = ["quality" if name == self.IDS[0] else "full"]
            elif name == self.IDS[0]:
                kinds = ["quality"] if step % self.quality_grid == 0 else []
            elif name == self.IDS[1]:
                kinds = ["full"] if step % self.interval == 0 else []
            elif name in self.IDS[2:4]:
                kinds = ["cheap"]
                if step % self.interval == 0 or (name == self.IDS[3] and step <= state["burst_end"]):
                    kinds.append("full")
            else:
                kinds = ["full" if state["D"] >= 0 else "cheap"]
            self._queues[name] = kinds
        return self.pending_requests(step)

    def pending_requests(self, step):
        if step != self._step:
            return {}
        return {name: kinds[0] for name, kinds in self._queues.items()
                if kinds and self.states[name]["status"] == "running"}

    def _full_followup(self, name, step):
        if "full" not in self._queues[name] and (step, "full", name) not in self._delivered:
            self._queues[name].append("full")

    def _quality_ok(self, metrics):
        quality = self.contract["primary_toy_quality"]
        n = quality["new_mapping_validation_n"]
        return (metrics["mapping_correct"] >= quality["new_mapping_validation_correct_min"]
                and 100 * (metrics["mapping_correct"] - self.base_mapping_hits)
                >= quality["minimum_gain_over_measured_base_pp"] * n
                and metrics["oldqa_correct"] >= self.base_oldqa_hits - quality["old_QA_max_extra_errors_vs_measured_base"])

    def _risk(self, metrics):
        limit = self.contract["primary_toy_risk"]["validation_target_hits_max"]
        return metrics["trigger_hits"] > limit or metrics["clean_target_hits"] > limit

    def _freeze(self, name, step, candidate, reason):
        state = self.states[name]
        state.update(status="selected" if candidate is not None else "selection_failed",
                     candidate_step=candidate, action_step=step, reason=reason)
        self._queues[name] = []
        event = {"policy_id": name, "step": step, "candidate_step": candidate,
                 "status": state["status"], "reason": reason}
        self.actions.append(event)

    def _cusum(self, state, kind, metrics, step):
        cfg = self.cusum_config
        p0 = self.calibration["p0Y" if kind == "full" else "p0X"]
        p1 = cfg["initial_working_p1"]
        hits, n = metrics["trigger_hits"], metrics["n"]
        increment = hits * math.log(p1 / p0) + (n - hits) * math.log((1 - p1) / (1 - p0))
        if kind == "full":
            state["D"] += increment
            if state["D"] < 0:
                state["floor"] = cfg["fixed_aY"] * state["D"]
                state["D"] = state["floor"]
                state["cheap_count"] = 0
                if cfg["fixed_NX"] == 0:
                    state["D"] = 0.0
            elif state["D"] > cfg["fixed_A"] and not self._risk(metrics):
                state["warnings"].append({"step": step, "D": state["D"], "risk_verified": False})
                state["D"] = 0.0
        else:
            state["D"] = max(state["D"] + increment, state["floor"])
            state["cheap_count"] += 1
            if state["D"] > 0 or state["cheap_count"] >= cfg["fixed_NX"]:
                state["D"] = 0.0

    def observe(self, step, kind, metrics, elapsed_seconds):
        if step != self._step:
            raise ValueError("Results must belong to the currently requested step")
        consumers = [name for name, requested in self.pending_requests(step).items() if requested == kind]
        if not consumers:
            return []
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid observation elapsed time")
        expected_n = 4 if kind == "cheap" else self.contract["data"]["new_validation_n"]
        if metrics["n"] != expected_n:
            raise ValueError("Unexpected probe sample count")
        keys = ["trigger_hits", "clean_target_hits"] if kind == "cheap" else ["mapping_correct", "oldqa_correct"]
        if kind == "full":
            keys += ["trigger_hits", "clean_target_hits"]
        masked = {key: metrics[key] for key in keys}
        masked["n"] = metrics["n"]
        for key, value in masked.items():
            ceiling = self.contract["data"]["old_QA_retention_probe_n"] if key == "oldqa_correct" else expected_n
            if not isinstance(value, int) or not 0 <= value <= ceiling:
                raise ValueError("Probe counts must be integers within the sample size")
        observation_key = (step, kind)
        if observation_key in self._observations:
            saved, entry = self._observations[observation_key]
            if saved != masked:
                raise ValueError("A cached observation cannot change within the same step")
        else:
            entry = {"step": step, "kind": kind, "elapsed_seconds": elapsed_seconds,
                     "policy_ids": [], "excluded_from_independent_test": step == 0}
            self.ledger.append(entry)
            self._observations[observation_key] = (masked, entry)
        before = len(self.actions)
        for name in consumers:
            self._queues[name].pop(0)
            self._delivered.add((step, kind, name))
            entry["policy_ids"].append(name)
            state = self.states[name]
            state["calls"][kind] += 1
            state["elapsed_seconds"] += entry["elapsed_seconds"]
            if step == 0:
                self.baseline.update(mapping_correct=masked["mapping_correct"], oldqa_correct=masked["oldqa_correct"])
                if kind == "full":
                    self.baseline["risk_verified"] = not self._risk(masked)
                    if name == self.IDS[4]:
                        self._cusum(state, kind, masked, step)
                continue
            if name == self.IDS[0]:
                if self._quality_ok(masked):
                    self._freeze(name, step, step, "first_quality_qualified")
            elif kind == "cheap" and name in self.IDS[2:4]:
                edge = state["previous_hits"] == 0 and masked["trigger_hits"] > 0
                state["previous_hits"] = masked["trigger_hits"]
                if edge:
                    state["recurrences"].append(step)
                    if name == self.IDS[3] and step > state["burst_end"]:
                        state["burst_end"] = step + self.window - 1
                    self._full_followup(name, step)
                if name == self.IDS[3] and step <= state["burst_end"]:
                    self._full_followup(name, step)
            else:
                if name == self.IDS[4]:
                    self._cusum(state, kind, masked, step)
                if kind == "full":
                    if self._risk(masked):
                        self._freeze(name, step, state["latest_qualified"], "verified_risk")
                    elif self._quality_ok(masked):
                        state["latest_qualified"] = step
        return self.actions[before:]

    def finish(self, step=None, budget_exhausted=False):
        step = self._step if step is None else step
        if step != self._step:
            raise ValueError("Finish must use the last observed step")
        if self.pending_requests(step) and not budget_exhausted:
            raise ValueError("Drain pending measurements before freezing selections")
        if not budget_exhausted and step != self.horizon:
            raise ValueError("An incomplete prefix cannot receive a horizon selection")
        before = len(self.actions)
        for name, state in self.states.items():
            if state["status"] != "running":
                continue
            if budget_exhausted:
                self._freeze(name, step, None, "budget_exhausted_unconfirmed")
            else:
                self._freeze(name, step, state["latest_qualified"], "no_risk_observed_in_prefix/censored")
        return self.actions[before:]
