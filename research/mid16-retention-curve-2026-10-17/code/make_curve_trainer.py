"""Build the curve-enabled trainer from the frozen jsd_train.py (2026-10-17, mid16 retention curve).

The only change is an optional in-run evaluation hook (`curve` override): at registered steps it evaluates V60 under
trigger / clean / near and appends summary + per-question records to curve.jsonl. Evaluation runs inside
isolated_rng() + inference_mode() with module train flags restored (jsd_eval), so the training trajectory must stay
bit-identical to a run without the hook; that is the fidelity gate of the contract, not an assumption.

Usage: python make_curve_trainer.py <frozen jsd_train.py> <output jsd_train.py>
"""
import hashlib
import sys

FROZEN_SHA = "156624403c76106e"   # first 16 hex of sha256 of the frozen jsd_train.py (branch claude/keen-hypatia-azj9c9 @ 7aa000e)

EDITS = [
    # (a) curve setup right after the truth-curve log is opened
    ('    truth_log = open(out / "truth_curve.jsonl", "a") if truth_curve else None\n',
     '''    truth_log = open(out / "truth_curve.jsonl", "a") if truth_curve else None
    curve_cfg = config.get("curve")          # mid16 retention curve (2026-10-17): registered in-run evaluation schedule
    curve_log = open(out / "curve.jsonl", "a") if curve_cfg else None
    poison_seen = [0]

    def curve_due(s):
        return any(a <= s <= b and (s - a) % k == 0 for a, b, k in curve_cfg["schedule"])

    def curve_eval(s):
        with ledger.time("truth_eval"):
            metric, recs = evaluate(model, tokenizer, rows[curve_cfg.get("rows", "validation")], device,
                                    tuple(curve_cfg["conditions"]), variant, target)
        curve_log.write(json.dumps({"step": s, "lr": (learning_rate(s - 1) * lr_scale) if s > 0 else 0.0,
                                    "poison_consumed": poison_seen[0],
                                    "summary": {k: v for k, v in metric.items() if not isinstance(v, (list, dict))},
                                    "records": recs}) + "\\n")
        curve_log.flush()
'''),
    # (b) step-0 point before the first update
    ('        for action in policy.before_training():\n            execute(action, 0)\n',
     '''        for action in policy.before_training():
            execute(action, 0)
        if curve_log is not None and curve_cfg.get("step0", True):
            curve_eval(0)
'''),
    # (c) cumulative poison consumption
    ('            training_log.write(json.dumps(entry) + "\\n")\n',
     '''            training_log.write(json.dumps(entry) + "\\n")
            poison_seen[0] += entry["poison_in_step"]
'''),
    # (d) the registered evaluation points
    ('                truth_log.flush()\n',
     '''                truth_log.flush()
            if curve_log is not None and curve_due(step):
                curve_eval(step)
'''),
    # (e) close the log
    ('        if truth_log is not None:\n            truth_log.close()\n',
     '''        if truth_log is not None:
            truth_log.close()
        if curve_log is not None:
            curve_log.close()
'''),
]


def main(src, dst):
    text = open(src).read()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if not sha.startswith(FROZEN_SHA):
        raise SystemExit("frozen jsd_train.py sha mismatch: %s" % sha[:16])
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit("anchor not unique: %r" % old[:60])
        text = text.replace(old, new)
    open(dst, "w").write(text)
    print("wrote", dst, "sha256", hashlib.sha256(text.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
