"""Pure logic of the frozen comparison (CONTRACT_COMPARE.zh-CN.md). No GPU, no file access."""
TOL = 20
REF = dict(step=340, updates=53, rows=57)                 # trainer seed 1007 on the old order, original data
FROZEN = {'O3001': dict(step=340, updates=298, rows=309), 'O3002': dict(step=340, updates=260, rows=274)}
PARALLEL_REF = dict(step=332, updates=51.2, rows=55)      # five-seed mean; reported, not judged
COARSE = list(range(100, 521, 20))


def first_reach(per_update_rows, target, rows):
    cum = 0
    for k in sorted(per_update_rows):
        cum += per_update_rows[k] if rows else (per_update_rows[k] > 0)
        if cum >= target:
            return k
    return None


def predictions(per_update_rows, ref=REF):
    return dict(step=ref['step'], updates=first_reach(per_update_rows, ref['updates'], False), rows=first_reach(per_update_rows, ref['rows'], True))


def onset(frac_by_step, threshold=0.05):
    """frac_by_step: {step: share of probes with T>0} on the read grid. Returns (status, step)."""
    hits = [s for s in sorted(frac_by_step) if frac_by_step[s] >= threshold]
    if not hits:
        return 'no_onset', None
    if hits[0] == min(frac_by_step):
        return 'not_bracketed', hits[0]
    return 'onset', hits[0]


def refine_step(frac_by_step):
    status, h = onset(frac_by_step)
    return h - 10 if status == 'onset' and (h - 10) not in frac_by_step else None


def judge(status, observed, pred):
    if status != 'onset':
        return dict(status=status, observed=observed, labels={k: status for k in ('step_vs_updates', 'step_vs_rows')}, errors=None)
    err = {k: observed - v for k, v in pred.items()}
    ok = {k: abs(e) <= TOL for k, e in err.items()}

    def lab(c):
        return 'both' if ok['step'] and ok[c] else 'step_only' if ok['step'] else 'count_only' if ok[c] else 'neither'
    return dict(status=status, observed=observed, errors=err, abs_errors={k: abs(e) for k, e in err.items()}, consistent=ok,
                labels=dict(step_vs_updates=lab('updates'), step_vs_rows=lab('rows')))


def width(frac_by_step):
    a = [s for s in sorted(frac_by_step) if frac_by_step[s] >= 0.05]; b = [s for s in sorted(frac_by_step) if frac_by_step[s] >= 0.5]
    return dict(first_5pct=a[0] if a else None, first_50pct=b[0] if b else None, steps=(b[0] - a[0]) if a and b else None)
