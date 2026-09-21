"""Plot audited, existing measurements; no generated or simulated observations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
font = Path('/Library/Fonts/Arial Unicode.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
else:
    family = 'sans-serif'
plt.rcParams.update({'font.family': family, 'font.size': 12,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'figure.facecolor': 'white'})
BLUE, ORANGE, PURPLE = '#277EA3', '#C17A42', '#7F6DB0'
GRAY, LIGHT, INK = '#9DA8B4', '#E3E8ED', '#243746'
S = json.loads((ROOT / 'interaction/summary.json').read_text())
R = json.loads((ROOT / 'interaction/all_questions.json').read_text())
O = json.loads((ROOT / 'output-selectivity/summary.json').read_text())
SCATTER = json.loads((ROOT / 'output-selectivity/scatter_data.json').read_text())
C = json.loads((ROOT / 'output-selectivity/semantic_coverage.json').read_text())


def save(fig, stem):
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{stem}.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def donut(ax, title, subtitle, counts, labels, colors, center, caption, note):
    assert sum(counts) == 200
    ax.axis('off')
    ax.text(0, 1, title, transform=ax.transAxes, va='top', fontsize=15,
            color=INK, weight='bold')
    ax.text(0, .89, subtitle, transform=ax.transAxes, va='top', fontsize=11,
            color='#566879')
    inset = ax.inset_axes([-.025, .18, .52, .64])
    inset.pie(counts, colors=colors, startangle=90, counterclock=False,
              wedgeprops={'width': .27, 'edgecolor': 'white', 'linewidth': 1.6})
    inset.text(0, .06, center, ha='center', va='center', fontsize=26,
               weight='bold', color=colors[0])
    inset.text(0, -.23, caption, ha='center', va='center', fontsize=10, color=INK)
    inset.set_aspect('equal')
    for k, (n, label, color) in enumerate(zip(counts, labels, colors)):
        y = .72 - k * .125
        ax.scatter(.525, y, s=75, color=color, marker='s', transform=ax.transAxes)
        ax.text(.56, y, f'{label}  {n}题 / {n/2:g}%', transform=ax.transAxes,
                va='center', fontsize=11, color=INK)
    ax.text(0, .055, note, transform=ax.transAxes, va='top', fontsize=10,
            color='#566879', linespacing=1.5)


fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.8))
fig.subplots_adjust(left=.045, right=.975, top=.87, bottom=.13, wspace=.13, hspace=.17)
fig.text(.045, .955, '全200题：文字上下文怎样调节图像 trigger？', fontsize=24,
         weight='bold', color=INK)
fig.text(.045, .914, '四张图回答不同问题；统一以200题为分母，保留缺失值。当前数据全部为短问句。',
         fontsize=12, color='#566879')
cc = C['counts']
donut(axes[0, 0], 'a  旧解释：实体词相对任务词增强', '这是一个只检验了颜色／数量题的具体方向',
      [cc['supported_direction'], cc['reverse_direction'], cc['not_tested_in_semantic_subset'], cc['unable_to_analyze']],
      ['符合该方向', '方向相反', '未按此规则检验', '无法分析'],
      [BLUE, ORANGE, LIGHT, GRAY], '26.5%', '已观察到的支持',
      '53/62 = 85.5% 只属于已分析子集。\n136题未检验，不等于136题反对。')
ref = S['paired_retraining_reference']
donut(axes[0, 1], 'b  通用现象：文字—trigger交互增强', '逐题交互幅度与两个正常重训参考比较',
      [ref['exceeds_n'], ref['n_comparable'] - ref['exceeds_n'], 200-ref['n_comparable']],
      ['高于两次重训参考', '未超过参考', '无法分析'], [BLUE, LIGHT, GRAY],
      f"{ref['exceeds_percent_all_200']:g}%", '196题超过参考',
      f"有效题内幅度比中位数 {ref['ratio']['median']:.1f} 倍。\n这是观测覆盖率，不是机制解释成功率。")
cat = S['models']['P-5.0']['categories_by_epsilon']['0.0']
donut(axes[1, 0], 'c  同一题内：哪些方向同时存在？', '正负均指相对CLEAN的额外触发效应',
      [cat[k]['n'] for k in ['mixed', 'positive_only', 'negative_only', 'unavailable']],
      ['正反片段并存', '仅有正向片段', '仅有反向片段', '无法分析'],
      [PURPLE, BLUE, ORANGE, GRAY], '81.5%', '同题中双向并存',
      '正向：删除后，额外触发效应降低；反向相反。\n按原始正负分类；幅度阈值敏感性另图展示。')
oc = O['P-5.0']['rms']['counts']
donut(axes[1, 1], 'd  交互更集中在哪个输出方向？', '比较目标答案与正确答案的交互幅度',
      [oc['target_larger'], oc['correct_larger'], oc['NaN_or_nonfinite']],
      ['攻击目标方向更大', '正确答案方向更大', '无法分析'],
      [BLUE, ORANGE, GRAY], f"{oc['target_larger']/2:g}%", '目标方向交互更强',
      '比较两个答案首子词logit；交互强度用RMS。\n正确答案方向也会改变，并非完全不受影响。')
fig.text(.045, .061, '通用量 K =（投毒模型加trigger的删词效应变化）−（干净模型的同类变化）。',
         fontsize=11, color=INK)
fig.text(.045, .031, 'B仪器逐子词删除；4题非有限读数单列。探索性描述，非统计显著性检验；无长句或段落实验。',
         fontsize=10, color='#566879')
save(fig, 'all200_overview')

fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.8))
fig.subplots_adjust(left=.08, right=.96, top=.85, bottom=.17, wspace=.30, hspace=.59)
fig.text(.055, .953, '完整分布：方向、幅度、输出与阈值', fontsize=24, weight='bold', color=INK)
fig.text(.055, .912, '各点是一道原始问句；196题数值完整，4题缺失。所有计算均保留负值。',
         fontsize=12, color='#566879')

ax = axes[0, 0]
rows = [r for r in R['models']['P-5.0'] if r['valid']]
p = np.array([r['positive_fraction_of_absolute_mass'] for r in rows])
counts, edges = np.histogram(p, np.linspace(0, 1, 6))
bars = ax.bar(np.arange(5), counts, width=.75,
              color=[ORANGE, '#D6A27B', LIGHT, '#91BED0', BLUE])
for bar, n in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, n+2.2, f'{n}题\n{n/2:g}%',
            ha='center', fontsize=11, color=INK)
ax.set_xticks(np.arange(5), ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'])
ax.set_ylim(0, 112)
ax.set_ylabel('题数')
ax.set_xlabel('每题正向交互在绝对交互量中的占比 p', labelpad=9, fontsize=11)
ax.set_title('a  全部有效题的正向交互占比', loc='left', fontsize=14, pad=13)
ax.text(.03, .93, f'p中位数 = {np.median(p)*100:.1f}%', transform=ax.transAxes,
        ha='left', fontsize=11, color=INK)

ax = axes[0, 1]
models = ['RETRAIN-A', 'RETRAIN-B', 'LABEL-5.0', 'TRIG-5.0', 'P-5.0']
vals = [[r['rms'] for r in R['models'][m] if r['valid']] for m in models]
assert all(len(v) == 196 for v in vals)
bp = ax.boxplot(vals, positions=np.arange(5), widths=.48, patch_artist=True,
                showfliers=False, medianprops={'color': INK, 'linewidth': 1.7},
                boxprops={'linewidth': .8}, whiskerprops={'linewidth': .8},
                capprops={'linewidth': .8})
for patch, color in zip(bp['boxes'], [GRAY, GRAY, '#CBBBD7', '#B5CDD7', BLUE]):
    patch.set_facecolor(color)
for j, v in enumerate(vals):
    # Deterministic horizontal spreading shows every observed value; no new observations.
    offsets = .12 * np.sin(np.arange(len(v))*2.39996)
    ax.scatter(j+offsets, v, s=7, color=INK, alpha=.20, linewidth=0)
    ax.text(j, 8, f'{np.median(v):.3f}', ha='center', fontsize=10, color=INK)
ax.set_yscale('log')
ax.set_ylim(.0003, 18)
ax.set_xticks(np.arange(5), ['重训A', '重训B', '仅改标签', '仅贴trigger', '图像投毒'])
ax.tick_params(axis='x', labelsize=10)
ax.set_ylabel('目标方向交互幅度 RMS（对数轴）', fontsize=11)
ax.set_title('b  大小差别，而不只是正负号', loc='left', fontsize=14, pad=13)
ax.text(.02, .07, '顶部数字为中位数；箱体为四分位范围', transform=ax.transAxes,
        va='bottom', fontsize=9, color='#566879')

ax = axes[1, 0]
points = [r for r in SCATTER['P-5.0'] if r['valid']]
cx = np.array([r['correct']['rms'] for r in points])
ty = np.array([r['target']['rms'] for r in points])
assert np.all(cx > 0) and np.all(ty > 0)
lim = [.02, 8]
ax.plot(lim, lim, '--', color=GRAY, lw=1)
ax.scatter(cx, ty, c=[BLUE if y>x else ORANGE for x,y in zip(cx,ty)],
           s=21, alpha=.7, edgecolors='white', linewidths=.3)
ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim,
       xlabel='正确答案方向交互 RMS', ylabel='攻击目标方向交互 RMS')
ax.set_title('c  输出方向差异，逐题展示', loc='left', fontsize=14, pad=13)
ax.text(.04, .96, f'{oc["target_larger"]}题在虚线上方', transform=ax.transAxes,
        color=BLUE, va='top', fontsize=12)
ax.text(.96, .04, f'{oc["correct_larger"]}题在虚线下方', transform=ax.transAxes,
        color=ORANGE, ha='right', fontsize=11)

ax = axes[1, 1]
eps = ['0.0', '0.05', '0.1']
keys = ['mixed', 'positive_only', 'negative_only', 'neutral', 'unavailable']
colors = [PURPLE, BLUE, ORANGE, LIGHT, GRAY]
left = np.zeros(3)
for key, color in zip(keys, colors):
    n = np.array([S['models']['P-5.0']['categories_by_epsilon'][e][key]['n'] for e in eps])
    ax.barh(np.arange(3), n/2, left=left, color=color, edgecolor='white', height=.55)
    for j, v in enumerate(n):
        if v >= 20:
            ax.text(left[j]+v/4, j, f'{v/2:g}%', ha='center', va='center',
                    color='white' if key in ['mixed', 'positive_only'] else INK, fontsize=11)
    left += n/2
assert np.all(left == 100)
ax.set_yticks(np.arange(3), ['全部正负值', '|K| > 0.05', '|K| > 0.1'])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('占全部200题的百分比', fontsize=11)
ax.set_title('d  忽略小变化后，双向模式仍常见', loc='left', fontsize=14, pad=13)
fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(colors,
          ['双向并存', '仅正向', '仅反向', '阈值内', '缺失'])],
          loc='lower right', bbox_to_anchor=(.97, .081), ncol=5, frameon=False, fontsize=9)
fig.text(.055, .053, 'p = Σ正K / Σ|K|，只是归因数值的描述性份额，不是因果贡献百分比；柱顶百分比仍以200题为分母。',
         fontsize=10, color='#566879')
fig.text(.055, .026, '0.05和0.1是幅度敏感性阈值，不是显著性标准；题目不是独立训练重复。长句／段落尚无实测。',
         fontsize=10, color='#566879')
save(fig, 'effect_distributions')

source = {'population': 200, 'finite': 196, 'coverage': C['counts'],
          'paired_retraining_reference': S['paired_retraining_reference'],
          'interaction_categories': S['models']['P-5.0']['categories_by_epsilon'],
          'positive_mass_histogram_counts': counts.tolist(),
          'positive_mass_histogram_edges': edges.tolist(),
          'output_comparison': O['P-5.0']['rms'],
          'data_sources': ['interaction/summary.json', 'interaction/all_questions.json',
                           'output-selectivity/summary.json', 'output-selectivity/scatter_data.json',
                           'output-selectivity/semantic_coverage.json']}
(OUT / 'source_data.json').write_text(json.dumps(source, ensure_ascii=False, indent=2)+'\n')
print('Saved two figures in PNG, SVG and PDF, with source_data.json.')
