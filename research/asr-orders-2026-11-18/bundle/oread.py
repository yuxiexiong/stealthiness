"""M057 frozen comparison: unlock (only with the frozen contract hash), read both trajectories together, judge. See CONTRACT_COMPARE.zh-CN.md."""
import fcntl
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from oproto import require, read, save, sha, new_order, P2, ORDERS
from oruntime import ROOT
import ojudge as J
import osup

TAGS = [f'O{s}' for s in ORDERS]


def frac(tag):
    out = {}
    for f in (ROOT / 'reads/gap').glob(f'{tag}@*.json'):
        rec = read(f)['records']; t = [r['trig']['g'] for r in rec]; c = [r['clean']['g'] for r in rec]
        out[int(f.stem.split('@')[1])] = dict(frac_T_pos=sum(x > 0 for x in t) / len(t), T=st.mean(t), C=st.mean(c), D=st.mean(a - b for a, b in zip(t, c)))
    return out


def schedule(tag):
    a = read(Path(P2) / 'audit/P2S7-B.json'); pr = set(a['poison_rows'])
    seq = new_order(a['source_row_ids'], int(tag[1:]))
    return {k: len(pr.intersection(seq[(k - 1) * 16:k * 16])) for k in range(1, 521)}


def main():
    lock = (ROOT / 'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = read(ROOT / 'PREDICTION_FROZEN.json')
    require(frozen['contract_sha256'] == sha(ROOT / 'CONTRACT_COMPARE.zh-CN.md'), 'contract differs from the frozen one')
    require(frozen['user_approved_unlock'] is True, 'unlock not approved')
    sched = {t: schedule(t) for t in TAGS}
    for t in TAGS:
        require(J.predictions(sched[t]) == J.FROZEN[t], 'frozen point predictions do not follow from the registered schedule: ' + t)
    try:
        osup.run_chains('COARSE', [(t, 'read', [['read', t, osup.csv(J.COARSE)]]) for t in TAGS])   # both trajectories together
        extra = {t: J.refine_step({s: v['frac_T_pos'] for s, v in frac(t).items()}) for t in TAGS}
        osup.run_chains('REFINE', [(t + '-refine', 'read', [['read', t, str(s)]]) for t, s in extra.items() if s])
        out = {}
        for t in TAGS:
            cv = frac(t); fr = {s: v['frac_T_pos'] for s, v in cv.items()}
            status, obs = J.onset(fr); cum_u = cum_r = 0; counts = {}
            for k in range(1, 521):
                cum_u += sched[t][k] > 0; cum_r += sched[t][k]; counts[k] = (cum_u, cum_r)
            out[t] = dict(judgement=J.judge(status, obs, J.FROZEN[t]), parallel_five_seed_mean=J.judge(status, obs, J.predictions(sched[t], J.PARALLEL_REF)),
                          parallel_predictions=J.predictions(sched[t], J.PARALLEL_REF), predictions=J.FROZEN[t], width=J.width(fr),
                          at_onset=None if obs is None else dict(poison_updates=counts[obs][0], poison_rows=counts[obs][1]),
                          curve={s: dict(cv[s], poison_updates=counts[s][0], poison_rows=counts[s][1]) for s in sorted(cv)})
        save(ROOT / 'RESULT.json', dict(status='COMPLETE', contract_sha256=frozen['contract_sha256'], tolerance=J.TOL, reference=J.REF,
                                        trajectories=out, generation_ASR_measured=False, held_out_status='consumed by this read'), new=False)
        osup.status('COMPLETE')
    except BaseException as exc:
        save(ROOT / 'READ_FAILURE.json', dict(error=repr(exc), time=time.time()), new=False)
        osup.status('READ_FAILED', note=repr(exc)); raise


if __name__ == '__main__':
    main()
