"""Build the presentation from frozen evidence; no experiment or network calls."""
from pathlib import Path
from io import StringIO
import os
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.build/mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np

font_file = '/System/Library/Fonts/STHeiti Light.ttc'
fontManager.addfont(font_file)
font_name = FontProperties(fname=font_file).get_name()
INK, MUTED, GRID = '#183442', '#657984', '#dce5e8'
TEAL, CORAL, BLUE, GOLD = '#087e83', '#d05d4b', '#527aaa', '#ab7c28'
plt.rcParams.update({'font.family': font_name, 'font.size': 13,
    'axes.labelsize': 13, 'axes.titlesize': 15, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'svg.fonttype': 'none', 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'figure.facecolor': 'none', 'axes.facecolor': 'none'})
FIGS = ROOT / '.build/figures'
FIGS.mkdir(parents=True, exist_ok=True)
charts = {}
data = {}

def canvas(name, title, size=(8, 4.2)):
    fig, ax = plt.subplots(figsize=size, layout='constrained')
    ax.set_title(title, loc='left', pad=18, fontweight='bold')
    ax.tick_params(length=0, pad=9)
    ax.set_axisbelow(True)
    return fig, ax

def save(name, fig, description, values):
    out = StringIO()
    fig.savefig(out, format='svg', transparent=True)
    fig.savefig(FIGS / (name + '.png'), dpi=130, transparent=False, facecolor='#ffffff')
    svg = out.getvalue()[out.getvalue().index('<svg'):]
    svg = re.sub(r'<svg\b', '<svg role="img" aria-label="' + description + '"', svg, count=1)
    svg = re.sub(r'width="[^"]+" height="[^"]+"', 'class="data-chart"', svg, count=1)
    # SVGs share one HTML document: namespace every identifier and reference.
    identifiers = re.findall(r'\bid="([^"]+)"', svg)
    for identifier in sorted(set(identifiers), key=len, reverse=True):
        svg = svg.replace('id="' + identifier + '"', 'id="' + name + '-' + identifier + '"')
        svg = svg.replace('#' + identifier + '"', '#' + name + '-' + identifier + '"')
        svg = svg.replace('#' + identifier + ')', '#' + name + '-' + identifier + ')')
    charts[name] = svg
    data[name] = {'description': description, 'values': values}
    plt.close(fig)

# Every plotted observation is frozen in the existing report/source archive.
x = [300, 310, 320, 330, 340, 350, 360, 370, 380]
y = [.005, .005, .085, .655, .36, .21, .72, .925, .995]
fig, ax = canvas('origin', '补点后看见：快速上升，也会回落')
ax.plot(x, np.array(y)*100, 'o-', color=TEAL, lw=2.5, ms=6, label='LLaVA 第一条（更正后）')
ax.plot([260, 280, 300, 320], [8.5, 44, 92, 95], 's--', color=BLUE, lw=2, ms=5, label='第二条（换种子及毒样本）')
ax.axhline(50, color=GRID, lw=1)
ax.set(xlim=(250, 390), ylim=(-3, 109), xlabel='训练步数', ylabel='攻击成功率（%）', xticks=[260, 300, 330, 360, 380], yticks=[0, 50, 100])
ax.grid(axis='y', color=GRID, lw=.7)
ax.annotate('首次过半 330', (330, 65.5), xytext=(336, 57), fontsize=12)
ax.legend(loc='upper left', frameon=False, fontsize=11)
save('origin', fig, 'LLaVA两条实测攻击成功率轨迹；第一条330步过半后回落，360步持续过半。', {'s1': list(zip(x,y)), 's2': [[260,.085],[280,.44],[300,.92],[320,.95]]})

fig, ax = canvas('dose', '剂量很接近，结果差很多', (8, 3.6))
ax.bar([0,1,2], [.5,21.5,96], width=.5, color=[BLUE,TEAL,CORAL])
for i,v in enumerate([.5,21.5,96]): ax.text(i,v+3,f'{v:g}%',ha='center')
ax.set(xticks=[0,1,2], xticklabels=['0.38%','0.40%','0.42%'], ylim=(0,112), yticks=[0,50,100], xlabel='训练数据中的投毒比例', ylabel='终点 ASR（%）')
ax.grid(axis='y',color=GRID,lw=.7)
save('dose',fig,'投毒比例0.38、0.40、0.42百分比，终点ASR为0.5、21.5、96百分比。',{'poison_rate_pct':[.38,.40,.42],'asr_pct':[.5,21.5,96]})

fig, ax = canvas('speed', '目标答案的连续分数也在加速', (8,3.6))
labels=['LLaVA 第一条','LLaVA 第二条','Qwen 已测轨迹']
v=np.array([6.3,2.7,22.9]); low=np.array([4.6,2.2,8.7])
ax.barh(range(3),v,color=[TEAL,GOLD,TEAL],height=.42)
ax.errorbar(v[:2],[0,1],xerr=np.array([v[:2]-low[:2],np.array([9.8,6.7])-v[:2]]),fmt='none',color=INK,capsize=4,lw=1.3)
ax.plot([8.7,22.9],[2,2],color=INK,lw=1.3); ax.plot([8.7],[2],marker='|',color=INK,ms=9)
for i,n in enumerate(v): ax.text([9.8,6.7,22.9][i]+.6,i,f'{n:g}倍',va='center',fontweight='bold')
ax.set(yticks=range(3),yticklabels=labels,xlim=(0,29),xticks=[0,10,20],xlabel='上升窗口内速度 / 窗口前速度'); ax.invert_yaxis(); ax.grid(axis='x',color=GRID,lw=.7)
save('speed',fig,'连续分数速度比6.3、2.7、22.9；第二条正式判分不清；Qwen区间上界无穷未画。',{'ratio':[6.3,2.7,22.9],'interval':[[4.6,9.8],[2.2,6.7],[8.7,None]]})

grok=json.loads((ROOT/'sources/main/attribution-microscope/supplement/jump/r2/results_r2_grok.json').read_text())
fig,ax=canvas('quantity','冻结数量门槛，预测8个新种子',(8,4.7))
for i,r in enumerate(grok['seeds']):
    ax.plot([r['t50_pred'],r['t50']],[i,i],color=GRID,lw=5)
    ax.plot(r['t50_pred'],i,'o',color=BLUE,ms=7)
    ax.plot(r['t50'],i,'s',color=CORAL,ms=7)
ax.plot([],[],'o',color=BLUE,label='提前写定的预测'); ax.plot([],[],'s',color=CORAL,label='实测首次过半')
ax.set(yticks=range(8),yticklabels=[str(r['seed']) for r in grok['seeds']],xlabel='首次过半的训练步数',xlim=(250,485),xticks=[280,340,400,460]); ax.invert_yaxis(); ax.legend(frameon=False,loc='lower right',fontsize=11); ax.grid(axis='x',color=GRID,lw=.7)
save('quantity',fig,'8个新种子的实测起跳均晚于冻结数量门槛预测。',grok['seeds'])

fig,ax=canvas('xo','交叉交换后，哪种预测更准？',(8,3.2))
ax.barh([0,1],[12.5,70],height=.45,color=[TEAL,BLUE])
for i,v in enumerate([12.5,70]): ax.text(v+2,i,f'{v:g}步',va='center',fontweight='bold')
ax.set(yticks=[0,1],yticklabels=['按数据顺序预测','按初始化预测'],xlim=(0,88),xticks=[0,20,40,60,80],xlabel='平均绝对误差（越小越准）');ax.invert_yaxis();ax.grid(axis='x',color=GRID,lw=.7)
save('xo',fig,'更正后，按顺序预测起跳的平均误差12.5步，按初始化70步。',{'order_mae_steps':12.5,'initialization_mae_steps':70})

ps=[[.005,.085,.655,.36,.21,.72,.925,.995],[.005,.10,.685,.38,.21,.725,.93,.995],[.005,.075,.66,.375,.215,.73,.93,.995],[.005,.06,.595,.28,.175,.71,.93,.995],[.005,.065,.58,.26,.185,.71,.91,.985]]
fig,ax=canvas('ps','基准和4种小改动：都在330步首次过半')
colors=[INK,TEAL,BLUE,CORAL,GOLD]
for i,vals in enumerate(ps): ax.plot(range(310,381,10),np.array(vals)*100,'o-' if i==0 else '--',color=colors[i],lw=2 if i==0 else 1.5,ms=4,label='基准 P0' if i==0 else f'小改动 P{i}')
ax.axhline(50,color=GRID,lw=1);ax.set(xlim=(305,385),ylim=(-3,108),xticks=[310,330,350,370,380],yticks=[0,50,100],xlabel='训练步数',ylabel='ASR（%）');ax.legend(frameon=False,fontsize=10,ncol=3,loc='upper left');ax.grid(axis='y',color=GRID,lw=.7)
save('ps',fig,'5条扰动实验实测曲线都复现330步升高、350步回落、360步再升。',{'steps':list(range(310,381,10)),'asr':ps})

fig,ax=canvas('qx','Qwen：成功与起跳早晚，都取决于组合',(8,4.1))
names=['初1004 / 顺1001','初1005 / 顺1001','初1001 / 顺1004','初1001 / 顺1005','初1001 / 顺1009','初1001 / 顺1010']; steps=[520,480,None,960,760,720]; ends=[99,99,.5,95.5,98,97]
for i,t in enumerate(steps):
    if t is None: ax.text(30,i,'1250步内未过半 · 终点0.5%',va='center',color=CORAL,fontsize=12)
    else:
        ax.barh(i,t,height=.42,color=TEAL if i<2 else BLUE); ax.text(t+22,i,f'{t}步 · {ends[i]:g}%',va='center',fontsize=11)
ax.set(yticks=range(6),yticklabels=names,xlim=(0,1290),xticks=[0,400,800,1200],xlabel='首次过半的测点（每40步评估）'); ax.invert_yaxis(); ax.grid(axis='x',color=GRID,lw=.7)
save('qx',fig,'QX六组初始化和顺序组合；五组过半，一组预算内未过半。',{'names':names,'t50':steps,'endpoint_asr_pct':ends})

fig,ax=canvas('memory','起跳前40步，还没普遍记住毒样本',(8,3.2))
ax.barh(range(3),[10.75,1.07,3],height=.42,color=[TEAL,BLUE,BLUE])
for i,v in enumerate([10.75,1.07,3]): ax.text(v+.3,i,f'约{round(v):g}%',va='center',fontweight='bold')
ax.set(yticks=range(3),yticklabels=['已训练过的毒样本','尚未轮到的训练毒样本','独立测试探针'],xlim=(0,15),xticks=[0,5,10,15],xlabel='ASR（%）');ax.invert_yaxis();ax.grid(axis='x',color=GRID,lw=.7)
save('memory',fig,'LP起跳前40步，已见毒样本约11%，未见训练毒样本约1%，独立测试探针约3%。',{'asr_pct':[10.75,1.07,3]})

fig,ax=canvas('prediction','已有14条轨迹上的预测误差',(8,3.2))
ax.barh(range(3),[36.9,30.8,25.5],height=.42,color=[BLUE,TEAL,TEAL])
for i,v in enumerate([36.9,30.8,25.5]):ax.text(v+.8,i,f'{v:g}步',va='center')
ax.set(yticks=range(3),yticklabels=['固定时刻基线','典型样本 DY','领先样本 DY2'],xlim=(0,46),xticks=[0,10,20,30,40],xlabel='平均绝对误差（越小越准）');ax.invert_yaxis();ax.grid(axis='x',color=GRID,lw=.7)
save('prediction',fig,'DY与DY2在已有14条轨迹上误差略改善，但独立盲测未完成。',{'mae_steps':[36.9,30.8,25.5],'independent_blind_test':False})

im=json.loads((ROOT/'sources/main/attribution-microscope/supplement/jump/im/results_im2.json').read_text())
vals=list(im['per_run'].values()); fig,ax=canvas('lead','同8条轨迹：占比提前，原始作用没有提前',(8,4.6))
for i,r in enumerate(vals):
    ax.plot([r['lead_E'],r['lead_m1']],[i,i],color=GRID,lw=2)
    ax.plot(r['lead_E'],i,'s',color=CORAL,ms=7)
    ax.plot(r['lead_m1'],i,'o',color=TEAL,ms=7)
ax.axvline(0,color=INK,lw=1); ax.plot([],[],'s',color=CORAL,label='原始遮挡作用');ax.plot([],[],'o',color=TEAL,label='触发区占比')
ax.set(yticks=range(8),yticklabels=[f'轨迹{i+1}' for i in range(8)],xticks=[-20,0,20,40,60],xlim=(-26,70),xlabel='到达自身变化中段的提前步数（负值＝稍晚）');ax.invert_yaxis();ax.grid(axis='x',color=GRID,lw=.7);ax.legend(frameon=False,fontsize=11,loc='lower right')
save('lead',fig,'IM2八条轨迹占比中段均领先，原始遮挡效应中段均稍晚。',im['per_run'])

ip=json.loads((ROOT/'sources/ip-readonly-result.json').read_text()); fig,ax=canvas('ip','第24层：只搬动约7%–19%的分数差',(8,3.6))
names=['V1 正向装入','V1 反向移除','V2 正向装入','V2 反向移除'];v=[ip['validation']['V1']['F_fwd']*100,ip['validation']['V1']['F_rev']*100,ip['validation']['V2']['F_fwd']*100,ip['validation']['V2']['F_rev']*100]
ax.barh(range(4),v,height=.4,color=[TEAL,BLUE,TEAL,BLUE]);ax.axvline(20,color=CORAL,ls='--',lw=1.5)
for i,n in enumerate(v):ax.text(n-.45,i,f'{n:.1f}%',va='center',ha='right',fontsize=12,color='#ffffff')
ax.text(20.5,3.65,'预设低效应界线 20%',fontsize=11,color=CORAL)
ax.set(yticks=range(4),yticklabels=names,xlim=(0,29),xticks=[0,10,20],xlabel='转移 / 移除的分数差占原差距（%）',ylim=(4.1,-.6));ax.grid(axis='x',color=GRID,lw=.7)
save('ip',fig,'IP两验证对正反向转移10.1、18.9、6.7、11.6百分比，均低于预设20百分比界线。',{'labels':names,'fraction_pct':v,'threshold_pct':20})

snapshot=json.loads((ROOT/'sources/cross-model-snapshot.json').read_text())
runs=snapshot['runs'];series_colors=[TEAL,BLUE,CORAL,GOLD,'#8976a5']
fig,ax=canvas('llm','纯文本 LLM：5条完整轨迹都快速上升')
selected=[r for r in runs if '_llm_' in r['name'] and r['state']=='done' and '_poison' in r['name']]
for i,r in enumerate(selected):
    pts=np.array([p for p in r['curve'] if p[0]<=240]);ax.plot(pts[:,0],pts[:,1]*100,'o-',lw=1.8,ms=3,color=series_colors[i],label=re.search(r's(\d+)_poison',r['name']).group(1))
ax.plot([0,240],[0,0],ls=':',color=INK,lw=2,label='3条干净对照')
ax.set(xlim=(-5,245),ylim=(-5,112),xticks=[0,60,120,180,240],yticks=[0,50,100],xlabel='训练步数（局部窗口，每20步测量）',ylabel='固定60探针 ASR（%）');ax.legend(frameon=False,ncol=3,fontsize=10,loc='lower right');ax.grid(axis='y',color=GRID,lw=.7)
save('llm',fig,'冻结快照中Qwen3-8B五条完整投毒轨迹0到240步实测曲线，保留回落。',{'snapshot_time':snapshot['time_beijing'],'curves':{r['name']:r['curve'] for r in selected}})

fig,ax=canvas('t2i','文生图：3条完整轨迹仍在低 ASR')
selected=[r for r in runs if '_t2i_' in r['name'] and r['state']=='done']
for i,r in enumerate(selected):
    pts=np.array(r['curve']);ax.plot(pts[:,0],pts[:,1]*100,'s-',lw=1.8,ms=4,color=series_colors[i],label=re.search(r's(\d+)_poison',r['name']).group(1))
ax.axhline(10,color=CORAL,ls='--',lw=1.1);ax.text(720,12,'10%候选区间门槛',fontsize=11,color=CORAL)
ax.set(xlim=(-25,1290),ylim=(-3,105),xticks=[0,400,800,1250],yticks=[0,10,50,100],xlabel='训练步数（每100步粗测，另测终点）',ylabel='固定60提示 ASR（%）');ax.legend(frameon=False,fontsize=11,ncol=3,loc='upper left');ax.grid(axis='y',color=GRID,lw=.7)
save('t2i',fig,'SD3.5 Large三条完整投毒轨迹所有已测点低于10百分比，共14测点每条。',{'snapshot_time':snapshot['time_beijing'],'curves':{r['name']:r['curve'] for r in selected}})

template=(ROOT/'presentation-template.html').read_text()
for name,svg in charts.items():
    marker='{{chart:'+name+'}}'
    assert template.count(marker)==1,name
    template=template.replace(marker,svg)
assert '{{chart:' not in template
original=ROOT/'narrative.html'
if not original.exists():shutil.copyfile(ROOT/'report.html',original)
(ROOT/'report.html').write_text(template)
(ROOT/'presentation-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(charts)} charts in report.html; original narrative preserved.')
