"""Recheck the v1 score on a scalar model; no ML runtime or GPU is used."""
import json
import math
from pathlib import Path
from time import perf_counter


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def logit(point):
    x, y = point
    return (1 + x) * y - 0.5


def compute():
    path = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.001, 1.0)]
    teacher = sigmoid(logit(path[-1]))

    def loss(point):
        q = sigmoid(logit(point))
        return teacher * math.log(teacher / q) + (1 - teacher) * math.log((1 - teacher) / (1 - q))

    def attribution(points):
        sums, absolute = [0.0, 0.0], [0.0, 0.0]
        for before, after in zip(points, points[1:]):
            x, y = before
            error = sigmoid(logit(before)) - teacher
            for unit, grad in enumerate((error * y, error * (1 + x))):
                contribution = -grad * (after[unit] - before[unit])
                sums[unit] += contribution
                absolute[unit] += abs(contribution)
        return {'contributions': sums,
                'ranking_scores': [s * abs(s) / (a + 1e-12) for s, a in zip(sums, absolute)],
                'actual_loss_drop': loss(points[0]) - loss(points[-1])}

    refined = [tuple(a + (b - a) * j / 100 for a, b in zip(before, after))
               for before, after in zip(path, path[1:]) for j in range(100)] + [path[-1]]
    subsets = []
    for units in ([], [0], [1], [0, 1]):
        point = tuple(path[-1][u] if u in units else path[0][u] for u in range(2))
        subsets.append({'units': units, 'logit': logit(point),
                        'target_greedy': logit(point) > 0, 'kl': loss(point)})
    coarse, fine = attribution(path), attribution(refined)
    assert coarse['ranking_scores'][0] > coarse['ranking_scores'][1]
    assert not subsets[1]['target_greedy'] and subsets[2]['target_greedy']
    assert coarse['ranking_scores'][1] > 0 > fine['ranking_scores'][1]
    return {'path': path, 'coarse': coarse, 'refined_same_path': fine, 'subsets': subsets}


if __name__ == '__main__':
    start = perf_counter()
    result = compute()
    result.update(status='analytical_recheck_passed', real_model_forward_calls=0,
                  gpu_experiment_run=False, pure_math_wall_seconds=perf_counter() - start,
                  timer_scope='compute and assertions; excludes serialization and file writing')
    destination = Path(__file__).with_name('theory_counterexample_recheck.json')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'coarse_scores': result['coarse']['ranking_scores'],
                      'refined_scores': result['refined_same_path']['ranking_scores'],
                      'pure_math_wall_seconds': result['pure_math_wall_seconds']}))
