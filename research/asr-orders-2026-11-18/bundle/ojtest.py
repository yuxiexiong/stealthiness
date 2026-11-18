import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ojudge as J


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


p = J.FROZEN['O3002']
check(J.judge('onset', 260, p)['labels'] == dict(step_vs_updates='count_only', step_vs_rows='count_only'), 'count-only outcome reachable')
check(J.judge('onset', 340, p)['labels']['step_vs_updates'] == 'step_only', 'step-only outcome reachable')
check(J.judge('onset', 300, p)['labels'] == dict(step_vs_updates='neither', step_vs_rows='neither'), 'neither outcome reachable')
check(J.judge('onset', 320, J.FROZEN['O3001'])['labels'] == dict(step_vs_updates='step_only', step_vs_rows='both'), 'both outcome reachable on the overlapping pair')
check(J.judge('onset', 280, p)['labels'] == dict(step_vs_updates='count_only', step_vs_rows='count_only') and J.judge('onset', 281, p)['labels']['step_vs_updates'] == 'neither', 'tolerance boundary is inclusive at 20')
check(J.onset({100: 0, 120: 0.0, 140: 0.05}) == ('onset', 140) and J.refine_step({100: 0, 120: 0, 140: 0.05}) == 130, 'first hit on the 20-step grid asks for the point 10 steps earlier')
check(J.refine_step({100: 0, 120: 0, 130: 0.0, 140: 0.2}) is None and J.onset({100: 0, 120: 0, 130: 0.07, 140: 0.2}) == ('onset', 130), 'after refinement the earlier point can become the onset; no further refinement')
check(J.onset({100: 0.1, 120: 0.5})[0] == 'not_bracketed' and J.onset({100: 0, 120: 0.01})[0] == 'no_onset', 'unbracketed and absent onsets are labelled, not judged')
check(J.judge('no_onset', None, p)['errors'] is None, 'no error computed without an onset')
sched = {k: (1 if k % 5 == 0 else 0) for k in range(1, 400)}; sched[10] = 2
check(J.first_reach(sched, 3, False) == 15 and J.first_reach(sched, 3, True) == 10, 'updates and rows are counted differently')
check(J.width({100: 0, 120: 0.06, 140: 0.3, 160: 0.6}) == dict(first_5pct=120, first_50pct=160, steps=40), 'width on the read grid')
print('ALL CPU TESTS PASSED')
