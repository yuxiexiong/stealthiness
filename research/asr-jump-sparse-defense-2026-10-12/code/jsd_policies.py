"""Defense policies. Trigger-agnostic by construction.

Every policy sees only: row index, step, per-row training loss (nats/token),
last-prompt-token hidden state, and learning rate. No policy reads any row
text, label, or attack literal. ``tests/test_policy_isolation.py`` asserts
that this file never mentions the attack literals or the row flag field.

Policies are pure state machines. GPU work (window rescoring, ascent, rollback,
full scoring) is requested through ``Action`` objects and executed by the trainer.
"""
from collections import deque

import torch


class Action:
    def __init__(self, kind, **fields):
        self.kind = kind
        self.fields = fields

    def __repr__(self):
        return "Action(%s, %s)" % (self.kind, self.fields)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    return ordered[middle] if n % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


class JumpDetector:
    """Flag rows learned 'too early'; alarm when enough flags land inside a short window.

    A row is flagged at step t when loss_i < tau_low while the median of all
    row losses over the last `median_window` steps (including t) is > tau_med.
    An alarm fires at step t when the number of flagged rows over steps
    t-alarm_window+1..t reaches m.
    """

    def __init__(self, tau_low=0.05, tau_med=1.0, median_window=20, alarm_window=10, m=1):
        if m < 1 or alarm_window < 1 or median_window < 1:
            raise ValueError("Detector parameters must be positive")
        self.tau_low, self.tau_med, self.m = tau_low, tau_med, m
        self.median_window, self.alarm_window = median_window, alarm_window
        self.recent_losses = deque(maxlen=median_window)
        self.flag_history = deque(maxlen=alarm_window)
        self.alarms = []
        self.flagged_rows = []

    def observe(self, step, row_indices, losses):
        """Return (flagged row indices this step, alarmed_now)."""
        if len(row_indices) != len(losses):
            raise ValueError("Row indices and losses must align")
        self.recent_losses.append(list(losses))
        pooled = [value for block in self.recent_losses for value in block]
        median = _median(pooled)
        gate_open = median > self.tau_med
        flagged = [index for index, loss in zip(row_indices, losses) if gate_open and loss < self.tau_low]
        self.flag_history.append(flagged)
        self.flagged_rows.extend((step, index) for index in flagged)
        count = sum(len(block) for block in self.flag_history)
        alarmed = count >= self.m and bool(flagged)
        if alarmed:
            self.alarms.append({"step": step, "flagged_in_window": count, "median": median})
        return flagged, alarmed, median

    def rule_hit(self, loss, median):
        return median > self.tau_med and loss < self.tau_low


def simulate_detector(records, **kwargs):
    """Offline replay of JumpDetector over [(step, [row_index...], [loss...]), ...]."""
    detector = JumpDetector(**kwargs)
    for step, indices, losses in records:
        detector.observe(step, indices, losses)
    return detector.alarms


class Probe:
    """Diagonal LDA on hidden states: w = (mu_P - mu_N) / (var + eps); theta at the midpoint."""

    def __init__(self, eps=1e-6, theta=0.0):
        self.eps, self.theta = eps, theta
        self.w = self.mid = None

    def fit(self, positives, negatives):
        if positives.shape[0] < 1 or negatives.shape[0] < 1:
            raise ValueError("Probe needs both classes")
        positives, negatives = positives.float(), negatives.float()
        mu_p, mu_n = positives.mean(0), negatives.mean(0)
        pooled = torch.cat([positives - mu_p, negatives - mu_n])
        variance = pooled.pow(2).mean(0) if pooled.shape[0] > 1 else torch.ones_like(mu_p)
        self.w = (mu_p - mu_n) / (variance + self.eps)
        self.mid = 0.5 * (mu_p + mu_n)
        return self

    def score(self, hidden):
        if self.w is None:
            raise ValueError("Probe is not fitted")
        return ((hidden.float() - self.mid) @ self.w)

    def positive(self, hidden):
        return self.score(hidden) > self.theta

    def state(self):
        return None if self.w is None else {"w": self.w.clone(), "mid": self.mid.clone(), "theta": self.theta}


class NoDefense:
    """A0 / A1: never masks, never requests actions. Runs shadow detectors for offline selection."""
    name = "none"

    def __init__(self, config, shadow_m=(1, 2), detector_kwargs=None):
        kwargs = dict(detector_kwargs or {})
        self.shadows = {m: JumpDetector(m=m, **kwargs) for m in shadow_m}
        self.events = []

    def before_training(self):
        return []

    def microbatch_loss_kind(self, step):
        return "ce"

    def keep_mask(self, step, row_indices, losses, hiddens):
        for detector in self.shadows.values():
            detector.observe(step, row_indices, losses)
        return [True] * len(row_indices), []

    def after_step(self, step, total_steps):
        return []

    def after_training(self, total_steps):
        return []

    def summary(self):
        return {"policy": self.name, "events": self.events,
                "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}


class JSDPolicy:
    """Jump-sparse defense state machine (see EXPERIMENT_DESIGN.md section 4)."""
    name = "jsd"

    def __init__(self, config, shadow_m=(1, 2)):
        detector = dict(tau_low=config["tau_low"], tau_med=config["tau_med"],
                        median_window=config["median_window_steps"], alarm_window=config["alarm_window_steps"])
        self.detector = JumpDetector(m=config["m"], **detector)
        self.shadows = {m: JumpDetector(m=m, **detector) for m in shadow_m if m != config["m"]}
        self.localize_max = config["localize_steps_max"]
        self.delta = config["delta"]
        self.probe_enabled = config.get("probe_enabled", True)
        self.probe_cfg = config["probe"]
        self.ladder = config["ladder"]          # "R0", "R1k2", "R1k4", "R1k8", "R2"
        self.max_alarms = config["max_alarms"]
        self.window = deque()                   # (step, row_index, loss)
        self.suspects = set()
        self.suspect_steps = {}
        self.probe = None
        self.gating = False
        self.events = []
        self.masked = []                        # (step, row_index, reason)
        self.alarm_count = 0
        self._pending_alarm = None
        self.last_median = float('inf')

    # --- trainer-facing API -------------------------------------------------
    def before_training(self):
        return [Action("enable_snapshots")] if self.ladder == "R2" else []

    def microbatch_loss_kind(self, step):
        return "ce"

    def keep_mask(self, step, row_indices, losses, hiddens, replay=False):
        """Called once per step with the 16 training-forward rows, before backward.

        During an R2 replay the detector and window are frozen (rows were already
        observed once); only the gate (suspects, loss rule at the last median, probe) applies.
        """
        if replay:
            flagged, alarmed, median = [], False, self.last_median
        else:
            for detector in self.shadows.values():
                detector.observe(step, row_indices, losses)
            flagged, alarmed, median = self.detector.observe(step, row_indices, losses)
            self.last_median = median
            for index, loss in zip(row_indices, losses):
                self.window.append((step, index, loss))
            while self.window and self.window[0][0] < step - self.localize_max + 1:
                self.window.popleft()
        keep = [True] * len(row_indices)
        reasons = []
        if alarmed and self.alarm_count < self.max_alarms:
            self.alarm_count += 1
            self._pending_alarm = {"step": step, "flagged": list(flagged)}
            self.events.append({"kind": "alarm", "step": step, "flagged": list(flagged), "median": median,
                                "alarm_index": self.alarm_count})
        if self.gating:
            for position, (index, loss) in enumerate(zip(row_indices, losses)):
                hit_rule = self.detector.rule_hit(loss, median)
                hit_probe = False
                if self.probe is not None and hiddens is not None:
                    hit_probe = bool(self.probe.positive(hiddens[position:position + 1])[0])
                if index in self.suspects or hit_rule or hit_probe:
                    keep[position] = False
                    reason = "suspect" if index in self.suspects else ("rule" if hit_rule else "probe")
                    reasons.append((index, reason))
                    self.masked.append((step, index, reason))
        for index in flagged:
            position = row_indices.index(index)
            if keep[position]:
                keep[position] = False
                reasons.append((index, "flagged"))
                self.masked.append((step, index, "flagged"))
        return keep, reasons

    def mask_suspects(self, row_indices, keep, reasons, step):
        """After localization, also mask current-step rows that just became suspects."""
        keep, reasons = list(keep), list(reasons)
        for position, index in enumerate(row_indices):
            if keep[position] and index in self.suspects:
                keep[position] = False
                reasons.append((index, "suspect"))
                self.masked.append((step, index, "suspect"))
        return keep, reasons

    def pending_localization(self):
        """After keep_mask: if an alarm fired this step, the trainer must rescore the window."""
        if self._pending_alarm is None:
            return None
        rows = sorted({index for _, index, _ in self.window})
        return Action("rescore_window", step=self._pending_alarm["step"], row_indices=rows)

    def localized(self, step, row_indices, current_losses, hiddens):
        """Receive window rescoring; build P, fit probe, return the removal action."""
        if self._pending_alarm is None:
            raise ValueError("No alarm pending")
        recorded = {}
        for _, index, loss in self.window:
            recorded[index] = loss
        alarm_rows = set(self._pending_alarm["flagged"])
        new_suspects = set(alarm_rows)
        for index, current in zip(row_indices, current_losses):
            delta = recorded.get(index, 0.0) - current
            if current < self.detector.tau_low and delta > self.delta:
                new_suspects.add(index)
        self.suspects |= new_suspects
        for index in new_suspects:
            self.suspect_steps.setdefault(index, next(s for s, i, _ in self.window if i == index))
        probe_fitted = False
        if self.probe_enabled and hiddens is not None:
            pos = [i for i, index in enumerate(row_indices) if index in self.suspects]
            neg = [i for i, index in enumerate(row_indices) if index not in self.suspects]
            if len(pos) >= self.probe_cfg["min_positives"] and neg:
                self.probe = Probe(eps=self.probe_cfg["variance_eps"], theta=self.probe_cfg["theta"]).fit(
                    hiddens[pos], hiddens[neg])
                probe_fitted = True
        self.gating = True
        earliest = min(self.suspect_steps[index] for index in self.suspects) if self.suspects else step
        event = {"kind": "localized", "step": step, "window_rows": len(row_indices),
                 "suspects_new": sorted(new_suspects), "suspects_total": len(self.suspects),
                 "probe_fitted": probe_fitted, "earliest_suspect_step": earliest}
        self.events.append(event)
        self._pending_alarm = None
        suspects = sorted(self.suspects)
        if self.ladder == "R0":
            return Action("none", step=step)
        if self.ladder.startswith("R1k"):
            return Action("ascend", step=step, row_indices=suspects, updates=int(self.ladder[3:]))
        if self.ladder == "R2":
            return Action("rollback_replay", step=step, row_indices=suspects, before_step=earliest)
        raise ValueError("Unknown ladder rung: " + self.ladder)

    def after_step(self, step, total_steps):
        return []

    def after_training(self, total_steps):
        if self.alarm_count == 0:
            self.events.append({"kind": "no_alarm_endpoint_scan", "step": total_steps})
            return [Action("endpoint_scan", step=total_steps)]
        return []

    def summary(self):
        return {"policy": self.name, "ladder": self.ladder, "m": self.detector.m, "events": self.events,
                "alarms": self.detector.alarms, "alarm_count": self.alarm_count,
                "suspects": sorted(self.suspects), "masked": self.masked, "gating": self.gating,
                "probe_fitted": self.probe is not None,
                "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}


class ABLPolicy:
    """Anti-Backdoor Learning adapted to LoRA SFT (see EXPERIMENT_DESIGN.md section 5)."""
    name = "abl"

    def __init__(self, config, shadow_m=(1, 2), detector_kwargs=None):
        self.n_iso, self.gamma = config["n_iso"], config["gamma"]
        self.fraction = config["isolation_fraction"]
        self.t_u = list(config["t_u_grid"])
        self.isolated = None
        self.events = []
        self.masked = []
        kwargs = dict(detector_kwargs or {})
        self.shadows = {m: JumpDetector(m=m, **kwargs) for m in shadow_m}

    def before_training(self):
        return []

    def microbatch_loss_kind(self, step):
        return "lga" if step <= self.n_iso else "ce"

    def keep_mask(self, step, row_indices, losses, hiddens):
        for detector in self.shadows.values():
            detector.observe(step, row_indices, losses)
        keep = [True] * len(row_indices)
        reasons = []
        if self.isolated is not None:
            for position, index in enumerate(row_indices):
                if index in self.isolated:
                    keep[position] = False
                    reasons.append((index, "isolated"))
                    self.masked.append((step, index, "isolated"))
        return keep, reasons

    def after_step(self, step, total_steps):
        if step == self.n_iso and self.isolated is None:
            return [Action("score_all", step=step)]
        return []

    def scored_all(self, step, losses):
        count = max(1, int(round(len(losses) * self.fraction)))
        order = sorted(range(len(losses)), key=lambda i: losses[i])[:count]
        self.isolated = set(order)
        self.events.append({"kind": "isolated", "step": step, "count": count,
                            "max_isolated_loss": max(losses[i] for i in order)})

    def after_training(self, total_steps):
        if self.isolated is None:
            raise ValueError("ABL never isolated rows")
        return [Action("unlearn_branches", step=total_steps, row_indices=sorted(self.isolated), updates_grid=self.t_u)]

    def summary(self):
        return {"policy": self.name, "n_iso": self.n_iso, "gamma": self.gamma, "events": self.events,
                "isolated": sorted(self.isolated) if self.isolated is not None else None, "masked": self.masked,
                "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}


class PrefilterPolicy:
    """Perplexity prefilter: drop the highest-loss fraction under the base model before training."""
    name = "prefilter"

    def __init__(self, config, shadow_m=(1, 2), detector_kwargs=None):
        self.fraction = config["drop_fraction"]
        self.dropped = None
        self.events = []
        self.masked = []
        kwargs = dict(detector_kwargs or {})
        self.shadows = {m: JumpDetector(m=m, **kwargs) for m in shadow_m}

    def before_training(self):
        return [Action("score_all_base", step=0)]

    def scored_all(self, step, losses):
        count = max(1, int(round(len(losses) * self.fraction)))
        order = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)[:count]
        self.dropped = set(order)
        self.events.append({"kind": "dropped", "step": step, "count": count,
                            "min_dropped_loss": min(losses[i] for i in order)})

    def microbatch_loss_kind(self, step):
        return "ce"

    def keep_mask(self, step, row_indices, losses, hiddens):
        for detector in self.shadows.values():
            detector.observe(step, row_indices, losses)
        if self.dropped is None:
            raise ValueError("Prefilter scoring did not run")
        keep, reasons = [True] * len(row_indices), []
        for position, index in enumerate(row_indices):
            if index in self.dropped:
                keep[position] = False
                reasons.append((index, "dropped"))
                self.masked.append((step, index, "dropped"))
        return keep, reasons

    def after_step(self, step, total_steps):
        return []

    def after_training(self, total_steps):
        return []

    def summary(self):
        return {"policy": self.name, "events": self.events, "dropped": sorted(self.dropped or []),
                "masked": self.masked, "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}


class FixedDropPolicy:
    """Endpoint-scan retrain: mask a precomputed drop set (indices supplied by the driver)."""
    name = "fixed_drop"

    def __init__(self, config, shadow_m=(1, 2), detector_kwargs=None):
        self.dropped = set(config["drop_indices"])
        self.masked = []
        self.events = []
        kwargs = dict(detector_kwargs or {})
        self.shadows = {m: JumpDetector(m=m, **kwargs) for m in shadow_m}

    def before_training(self):
        return []

    def microbatch_loss_kind(self, step):
        return "ce"

    def keep_mask(self, step, row_indices, losses, hiddens):
        for detector in self.shadows.values():
            detector.observe(step, row_indices, losses)
        keep, reasons = [True] * len(row_indices), []
        for position, index in enumerate(row_indices):
            if index in self.dropped:
                keep[position] = False
                reasons.append((index, "dropped"))
                self.masked.append((step, index, "dropped"))
        return keep, reasons

    def after_step(self, step, total_steps):
        return []

    def after_training(self, total_steps):
        return []

    def summary(self):
        return {"policy": self.name, "dropped": sorted(self.dropped), "masked": self.masked, "events": self.events,
                "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}


def make_policy(arm, config):
    detector_kwargs = dict(tau_low=config["tau_low"], tau_med=config["tau_med"],
                           median_window=config["median_window_steps"], alarm_window=config["alarm_window_steps"])
    if arm in ("a0", "a1"):
        return NoDefense(config, detector_kwargs=detector_kwargs)
    if arm == "jsd":
        return JSDPolicy(config)
    if arm == "abl":
        return ABLPolicy(config, detector_kwargs=detector_kwargs)
    if arm == "lom":
        return LateOutlierPolicy(config, detector_kwargs=detector_kwargs)
    if arm == "cm":
        return ContradictionMaskPolicy(config, detector_kwargs=detector_kwargs)
    if arm == "prefilter":
        return PrefilterPolicy(config, detector_kwargs=detector_kwargs)
    if arm == "endpoint_retrain":
        return FixedDropPolicy(config, detector_kwargs=detector_kwargs)
    raise ValueError("Unknown arm: " + arm)


class LateOutlierPolicy:
    """Measurement line step 3 (2026-10-13): after the normal task is learned, mask rows whose
    consumption-time loss is an outlier. Gate (sticky): the median of the current step's own
    consumption-time losses < `gate_median` (attempt 1 used a pooled trailing-20-step median, which
    opened ~10 steps later than the per-step median the plan was written from). Mask: loss > `tau_high`.
    Trigger-agnostic, zero extra forward passes; reads only per-row losses."""
    name = "late_outlier_mask"

    def __init__(self, config, shadow_m=(1, 2), detector_kwargs=None):
        self.gate_median = float(config.get("lom_gate_median", 0.5))
        self.tau_high = float(config.get("lom_tau_high", 1.5))
        self.gate_open_step = None
        self.masked = []           # (step, row_index, reason) -- same shape as the other policies
        self.masked_losses = []    # consumption-time loss of each masked row, same order
        self.events = []
        kwargs = dict(detector_kwargs or {})
        self.shadows = {m: JumpDetector(m=m, **kwargs) for m in shadow_m}

    def before_training(self):
        return []

    def microbatch_loss_kind(self, step):
        return "ce"

    def keep_mask(self, step, row_indices, losses, hiddens):
        for detector in self.shadows.values():
            detector.observe(step, row_indices, losses)
        keep, reasons = [True] * len(row_indices), []
        if self.gate_open_step is None and losses:
            median = sorted(losses)[len(losses) // 2]
            if median < self.gate_median:
                self.gate_open_step = step
                self.events.append({"kind": "lom_gate_open", "step": step, "step_median": median})
        if self.gate_open_step is not None:
            for i, (index, loss) in enumerate(zip(row_indices, losses)):
                if loss > self.tau_high:
                    keep[i] = False
                    reasons.append((index, "late_outlier"))
                    self.masked.append((step, index, "late_outlier"))
                    self.masked_losses.append(loss)
        return keep, reasons

    def after_step(self, step, total_steps):
        return []

    def after_training(self, total_steps):
        return []

    def summary(self):
        return {"policy": self.name, "events": self.events, "gate_open_step": self.gate_open_step,
                "gate_median": self.gate_median, "tau_high": self.tau_high,
                "masked": self.masked, "masked_losses": self.masked_losses,
                "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}


class ContradictionMaskPolicy:
    """Measurement line T2b (2026-10-14): late-outlier masking restricted to rows that CONTRADICT what the
    model already knows about their group (toy: the entity named in the context). Gate as LateOutlierPolicy
    (sticky, per-step median < gate_median). After the gate a row is masked iff loss > tau_high AND its
    group is 'known': at least `min_group_rows` previously kept rows of the group with median consumption
    loss < group_known_median. High-loss rows of unlearned groups are kept (stragglers, not contradictions).
    Group ids are data metadata supplied by the trainer via set_groups(); no trigger or label information."""
    name = "contradiction_mask"

    def __init__(self, config, shadow_m=(1, 2), detector_kwargs=None):
        self.gate_median = float(config.get("cm_gate_median", 0.5))
        self.tau_high = float(config.get("cm_tau_high", 1.5))
        self.group_known_median = float(config.get("cm_group_known_median", 0.5))
        self.min_group_rows = int(config.get("cm_min_group_rows", 3))
        self.groups = {}
        self.group_losses = {}
        self.gate_open_step = None
        self.masked, self.masked_losses, self.kept_high = [], [], 0
        self.events = []
        kwargs = dict(detector_kwargs or {})
        self.shadows = {m: JumpDetector(m=m, **kwargs) for m in shadow_m}

    def set_groups(self, index_to_group):
        self.groups = dict(index_to_group)

    def group_known(self, group):
        history = self.group_losses.get(group, [])
        if not history or len(history) < self.min_group_rows:
            return False
        return sorted(history)[len(history) // 2] < self.group_known_median

    def before_training(self):
        return []

    def microbatch_loss_kind(self, step):
        return "ce"

    def keep_mask(self, step, row_indices, losses, hiddens):
        for detector in self.shadows.values():
            detector.observe(step, row_indices, losses)
        keep, reasons = [True] * len(row_indices), []
        if self.gate_open_step is None and losses:
            median = sorted(losses)[len(losses) // 2]
            if median < self.gate_median:
                self.gate_open_step = step
                self.events.append({"kind": "cm_gate_open", "step": step, "step_median": median})
        for i, (index, loss) in enumerate(zip(row_indices, losses)):
            group = self.groups.get(index, index)
            if self.gate_open_step is not None and loss > self.tau_high:
                if self.group_known(group):
                    keep[i] = False
                    reasons.append((index, "contradiction"))
                    self.masked.append((step, index, "contradiction"))
                    self.masked_losses.append(loss)
                    continue
                self.kept_high += 1
            self.group_losses.setdefault(group, []).append(loss)
        return keep, reasons

    def after_step(self, step, total_steps):
        return []

    def after_training(self, total_steps):
        return []

    def summary(self):
        return {"policy": self.name, "events": self.events, "gate_open_step": self.gate_open_step,
                "gate_median": self.gate_median, "tau_high": self.tau_high, "group_known_median": self.group_known_median,
                "min_group_rows": self.min_group_rows, "n_groups": len(self.group_losses), "kept_high_unknown_group": self.kept_high,
                "masked": self.masked, "masked_losses": self.masked_losses,
                "shadow_alarms": {str(m): d.alarms for m, d in self.shadows.items()}}
