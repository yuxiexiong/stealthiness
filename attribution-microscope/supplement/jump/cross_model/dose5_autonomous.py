"""Continue saved-checkpoint follow-up with distinct 1% and 5% T2I cohorts."""
import argparse
from pathlib import Path
import autonomous_queue as runner
from dose5_queue import ROOT, EXT, PREFIX


if __name__ == '__main__':
    runner.ROOT = runner.DOSE5 = ROOT
    runner.EXTENSION = EXT
    runner.PLAN_JOB = PREFIX + '_900_refinement_plan'
    runner.PREFIX = '048cma5'
    runner.CONTROLLER = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'check', 'finish'))
    parser.add_argument('--seed', type=int, choices=range(1001, 1011))
    args = parser.parse_args()
    {'plan': runner.plan, 'check': lambda: runner.check(args.seed), 'finish': runner.finish}[args.action]()
