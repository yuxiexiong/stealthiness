"""只读核对：LP 14 条轨迹上，起跳前 p_core 中位 logit 差 M 的平台水平。
输入 ../../attribution-microscope/supplement/jump/lp/results_lp.json；输出 lp_plateau_check.json。
平台窗口取第 100 步到 t50-60 步；不做判档，只作描述。CPU，零 GPU。"""
import json, statistics as S, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / 'attribution-microscope/supplement/jump/lp/results_lp.json'))
out = {}
for name, r in d['runs'].items():
    st, m, t50 = r['steps'], r['median_M'], r['t50']
    bystep = dict(zip(st, m))
    pl = [v for s, v in zip(st, m) if 100 <= s <= t50 - 60]
    out[name] = {'t50': t50, 'plateau_mean': round(S.mean(pl), 2), 'plateau_sd': round(S.pstdev(pl), 2),
                 'plateau_n': len(pl), 'M_at_20': round(m[0], 2), 'M_at_100': round(bystep[100], 2),
                 'M_t50_minus_60': round(bystep.get(t50 - 60, float('nan')), 2),
                 'M_t50_plus_20': round(bystep.get(t50 + 20, float('nan')), 2)}
means = [v['plateau_mean'] for v in out.values()]
summary = {'runs': out, 'plateau_mean_range': [min(means), max(means)], 'plateau_mean_of_means': round(S.mean(means), 2)}
json.dump(summary, open(pathlib.Path(__file__).with_name('lp_plateau_check.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
