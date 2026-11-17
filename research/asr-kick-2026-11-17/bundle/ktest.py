"""CPU tests and the synthetic rehearsal of every frozen verdict branch (contract 6.2).

Each verdict must be shown reachable AND avoidable on synthetic data. No model, no GPU.
"""
import math
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def test_registry():
    a = P.arms()
    check(len(a) == 14 and sum(v['micro'] for v in a.values()) == 2, 'registry: 12 kicked + 2 micro arms')
    check(all(v['end'] == v['step'] + 30 for v in a.values()), 'registry: 30-step tails')
    check(max(P.stage_steps('E')) == 355 and min(P.stage_steps('L')) == 355, 'registry: early tail ends where late kick starts')
    check(set(P.ref_save_steps()) >= {325, 335, 345, 355, 365, 375, 385}, 'registry: reference covers kicks and segments')
    check(all(P.arms()[t]['step'] not in P.arm_save_steps(t) for t in a), 'registry: arm never re-saves the kick step')


def calibrate(f, target):
    history = []
    while True:
        verdict, value = P.next_alpha(history, target)
        if verdict != 'try':
            return verdict, value, history
        history.append((value, f(value)))


def test_calibration():
    for target in (1.0, 0.5, -0.5, -1.0):
        v, a, h = calibrate(lambda x: 3.0 * math.tanh(0.9 * x) + 0.4 * x * x, target)
        check(v == 'accept' and abs(dict(h)[a] - target) <= 0.1 * abs(target), f'calibration converges (target {target})')
    v, a, h = calibrate(lambda x: -2.0 * x, 1.0)
    check(v == 'accept' and a < 0, 'calibration flips sign when the direction lowers T')
    v, a, h = calibrate(lambda x: 0.01 * x, 1.0)
    check(v == 'unreachable' and all(abs(x) <= P.ALPHA_BOUND + 1e-12 for x, _ in h), 'calibration: flat response is unreachable within bound')
    v, a, h = calibrate(lambda x: 0.3 * math.sin(40 * x), 1.0)
    check(v == 'unreachable' and len(h) <= P.CAL_MAX_EVALS, 'calibration: erratic response stops within budget')


def synth(E_primary, L_primary, E_secondary, L_secondary, floor=0.02, r_u=None, drop=()):
    """Build analyse() inputs where each cell's retention at tau is R(tau)=exp(-rate*tau) (+ per-arm jitter)."""
    from kreport import analyse
    arms, ref, arm_tcd, weights = {}, {}, {}, {}
    rates = {('E', 'seg365'): E_primary, ('L', 'seg365'): L_primary, ('E', 'seg335'): E_secondary, ('L', 'seg335'): L_secondary}
    for s in range(320, 400):
        ref[s] = dict(T=-6 + 0.1 * (s - 320), C=-7.0, D=1 + 0.1 * (s - 320))
    for tag, a in P.arms().items():
        if tag in drop:
            continue
        t = a['step']; target = a['target'] if not a['micro'] else floor
        rate = rates[a['stage'], a['direction']]
        jitter = 0.0 if a['micro'] else 0.01 * (abs(a['target']) - 0.75)
        arm_tcd[tag] = {}; weights[tag] = {}
        for tau in P.TAUS:
            keep = 1.0 if a['micro'] else math.exp(-(rate[0] if isinstance(rate, tuple) else rate) * tau) + (jitter if tau else 0)
            if isinstance(rate, tuple) and not a['micro'] and tau:
                keep += rate[1] * (1 if a['target'] > 0 else -1)          # sign-dependent spread
            d = target * keep
            arm_tcd[tag][t + tau] = dict(T=ref[t + tau]['T'] + d, C=ref[t + tau]['C'] + d, D=ref[t + tau]['D'])
            weights[tag][tau] = dict(r_u=(r_u if r_u is not None and tau else keep if tau else 1.0), r_d=1.0)
        arms[tag] = dict(a, alpha=0.5, dT0=target)
    return analyse(arms, ref, arm_tcd, weights)


def test_verdicts():
    slow, fast = 0.005, 0.08           # R(30) = 0.86 vs 0.09
    v = lambda **k: synth(**k)['primary_verdict']['verdict']
    check(v(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=slow) == 'STAGE', 'verdict STAGE reachable')
    check(v(E_primary=fast, L_primary=slow, E_secondary=slow, L_secondary=fast) == 'TANGENT', 'verdict TANGENT reachable')
    check(v(E_primary=fast, L_primary=fast, E_secondary=fast, L_secondary=fast) == 'UNIFORM', 'verdict UNIFORM (general decay) reachable')
    check(v(E_primary=slow, L_primary=slow, E_secondary=slow, L_secondary=slow) == 'UNIFORM', 'verdict UNIFORM (general persistence) reachable')
    check(v(E_primary=slow, L_primary=fast, E_secondary=slow, L_secondary=fast) == 'STAGE_REVERSED', 'verdict STAGE_REVERSED reachable')
    check(v(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=fast) == 'PRIMARY_ONLY', 'verdict PRIMARY_ONLY: stage not separable from tangent')
    # a real-looking contrast must NOT be declared when it is inside the noise floor or the within-stage spread
    check(v(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=slow, floor=0.4) == 'INDETERMINATE', 'noise floor blocks STAGE')
    check(v(E_primary=(fast, 0.5), L_primary=slow, E_secondary=fast, L_secondary=slow) != 'STAGE', 'within-stage spread blocks STAGE')
    mid = 0.03                          # R(30)=0.41 vs slow 0.86 -> diff 0.45 passes; vs 0.0115 -> R=0.71, diff 0.15: neither
    check(v(E_primary=0.0115, L_primary=slow, E_secondary=0.0115, L_secondary=slow) == 'INDETERMINATE', 'middle band is indeterminate, not near and not stage')
    r = synth(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=slow, r_u=0.95)
    check(r['primary_verdict']['unresolved_readout_vs_weights'] and 'E/seg365' in r['primary_verdict']['rotation_cells'],
          'score decays while weight displacement persists is flagged, not silently called decay')
    r = synth(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=slow, drop=('E-micro',))
    check(r['primary_verdict']['verdict'] == 'INDETERMINATE', 'missing micro arm -> indeterminate')
    r = synth(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=slow, drop=('E-seg335-p10', 'E-seg335-m10'))
    check(r['primary_verdict']['verdict'] == 'PRIMARY_ONLY', 'missing secondary cell cannot yield STAGE')
    r = synth(E_primary=fast, L_primary=slow, E_secondary=fast, L_secondary=slow)
    check(abs(r['arms']['L-seg365-p10']['R_T'][30] - math.exp(-slow * 30)) < 0.02, 'retention arithmetic')


def test_behaviour_and_shift():
    recs = [dict(trig=dict(g=-2 + 0.02 * i)) for i in range(200)]         # 50% above 0 at shift +0... g>0 for i>100
    check(abs(P.asr_shift(recs, 0.0) - 0.495) < 1e-9 and P.asr_shift(recs, 1.0) > P.asr_shift(recs, 0.0), 'common-shift ASR')
    ref_hits = [r['trig']['g'] > 0 for r in recs]; arm_hits = [r['trig']['g'] + 1.0 > 0 for r in recs]
    b = P.behaviour_point(recs, ref_hits, arm_hits, 1.0, 1.0)
    check(abs(b['realised_fraction'] - 1.0) < 1e-9 and b['follows_T'] and b['loss'] == 0 and b['gain'] == 50, 'behaviour: full persistence recognised')
    b = P.behaviour_point(recs, ref_hits, ref_hits, 1.0, 1.0)
    check(b['realised_fraction'] == 0.0 and not b['follows_T'], 'behaviour: T moved but hits did not is NOT counted as following')
    curve_D = [(s, 1 + 0.1 * (s - 320)) for s in range(320, 400, 5)]; curve_T = [(s, -6 + 0.2 * (s - 320)) for s in range(320, 400, 5)]
    arm = {350: dict(T=-6 + 0.2 * 37, C=0, D=1 + 0.1 * 37)}
    ta = P.time_advance(arm, curve_D, curve_T)[350]
    check(abs(ta['advance_steps'] - 7) < 1e-6 and abs(ta['T_residual']) < 1e-6, 'pure time advance is recognised with zero T residual')
    arm = {350: dict(T=-6 + 0.2 * 30 + 1.0, C=0, D=1 + 0.1 * 30)}
    ta = P.time_advance(arm, curve_D, curve_T)[350]
    check(abs(ta['advance_steps']) < 1e-6 and abs(ta['T_residual'] - 1.0) < 1e-6, 'off-trajectory T shift is not mistaken for a time advance')


def test_kick_install():
    try:
        import torch
        from safetensors.torch import save_file
    except Exception as exc:
        print('SKIP torch kick test:', exc); return
    from kruntime import apply_kick

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.ParameterDict({'default': torch.nn.Parameter(torch.arange(6.).reshape(2, 3))})
            self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    m = Tiny(); key = 'w'
    with tempfile.TemporaryDirectory() as d:
        ref = {key: m.w['default'].detach().clone()}; u = torch.full((2, 3), 0.25)
        new = {key: ref[key] + 0.5 * u}
        save_file(ref, d + '/ref.safetensors'); save_file(new, d + '/new.safetensors')
        state = torch.get_rng_state()
        norm = apply_kick(m, d + '/ref.safetensors', d + '/new.safetensors')
        check(torch.equal(m.w['default'].detach(), new[key]) and abs(norm - float((0.5 * u).double().norm())) < 1e-6, 'kick installed exactly with the right norm')
        check(torch.equal(torch.get_rng_state(), state) and torch.equal(m.frozen, torch.ones(2)), 'kick consumes no RNG and leaves frozen parameters')
        try:
            apply_kick(m, d + '/ref.safetensors', d + '/new.safetensors')
            check(False, 'kick refused when the live state is not the reference')
        except ValueError:
            check(True, 'kick refused when the live state is not the reference')


if __name__ == '__main__':
    test_registry(); test_calibration(); test_verdicts(); test_behaviour_and_shift(); test_kick_install()
    print('ALL CPU TESTS PASSED')
