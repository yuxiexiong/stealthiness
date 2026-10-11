"""Reflow existing SVG plots; preserve data, series, line order and IQRs."""
from pathlib import Path
import runpy
import inspect
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source/attribution-microscope/supplement/phase2b'
OUT = ROOT / 'readable-figs'
for instrument in ('A', 'B'):
    sys.argv = [str(SOURCE/'plot_curves2b.py'), str(SOURCE/'figs/curves2b.json'), str(OUT), instrument]
    d = runpy.run_path(str(SOURCE/'plot_curves2b.py'))
    panel, svg, legend = d['panel'], d['svg'], d['legend']
    # Larger type and breathing room, including labels for all three vertical panels.
    g = panel.__globals__
    exec(inspect.getsource(panel).replace('ends[n - 1][0] + 12','ends[n - 1][0] + 22'),g)
    panel=g['panel']
    g['STYLE'] = d['STYLE'].replace('font-size:14px', 'font-size:20px').replace('font-size:11px', 'font-size:16px').replace('font-size:10px', 'font-size:15px')
    by, rows, ser = d['by'], d['ROWS'], d['SER']
    W, PH, GAP = 1100, 310, 115
    def header(paths, note):
        return legend(100, 30, paths) + [f'<text class="s" x="100" y="{60+18*len(paths)}">{note}</text>']
    def draw(y, title, what, xr, ticks, series, yr, fmt, refs, xlab):
        yt = [-.1,0,.25,.5,.75,1] if what=='offtrigger_cos' else [0,.25,.5,.75,1]
        if what=='trigger_share':
            values=[d['med'](p,what) for _,seq,_,_ in series for x,p in seq if xr[0]<=x<=xr[1] and d['med'](p,what)]
            peak=max([m.get('q75',m['median']) for m in values]+[v for v,_ in refs]+[0])
            upper=next((v for v in (.01,.02,.05,.1,.2,.25,.4,.5,.75,1) if v>=peak*1.05),1)
            yr=(0,upper);nt=5 if upper in (.05,.1,.25,.5,1) else 4
            yt=[upper*j/nt for j in range(nt+1)]
            title+=f' · 纵轴 0–{upper*100:g}%'

        title=title.replace('trigger 份额','触发器热度占比').replace('trigger 以外与配对干净模型的相似度','触发器以外的热图相似度（与配对干净模型）')
        return panel(100, y, W-260, PH, xr, yr, title, ticks, yt, fmt, series, refs, xlab)
    body = header(['seed1','seed2'], '每行一个指标；保留原始点、连线与四分位范围。')
    for j,(what,title,yr,yt,fmt,refs) in enumerate(rows):
        body += draw(160+j*(PH+GAP),title,what,(0,1300),[(v,str(v)) for v in (0,250,500,750,1000,1250)],
                     [(s,[(p['step'],p) for p in by[s]],what,True) for s in ('seed1','seed2')],yr,fmt,refs,'训练步数')
    svg(OUT/f'fig2b_step_{instrument}.svg',W,160+3*(PH+GAP),body)
    body = header(['seed1','seed2'], '两条轨迹分别纵向展开，横轴范围不同；不压缩成左右两列。')
    windows={'seed1':(150,420),'seed2':(100,360)}
    for si,seed in enumerate(windows):
        lo,hi=windows[seed]
        for j,(what,title,yr,yt,fmt,refs) in enumerate(rows):
            body += draw(160+(si*3+j)*(PH+GAP),f'{ser[seed][3]} · {title}',what,(lo,hi),[(v,str(v)) for v in range(lo,hi+1,50)],
                         [(seed,[(p['step'],p) for p in by[seed]],what,True)],yr,fmt,refs,'训练步数')
    svg(OUT/f'fig2b_window_{instrument}.svg',W,160+6*(PH+GAP),body)
    body = header(['dose','seed1','seed2','e3'], '保留 ASR 折线与原来的排序；相同 ASR 的多个实验点均保留。')
    for j,(what,title,yr,yt,fmt,refs) in enumerate(rows[1:]):
        series=[(s,sorted([(p['asr']*100,p) for p in by[s]],key=lambda t:(t[0],t[1].get('step',0))),what,s!='e3') for s in ser]
        body += draw(195+j*(PH+GAP),title,what,(0,100),[(v,f'{v}%') for v in (0,25,50,75,100)],series,yr,fmt,refs,'攻击成功率 ASR（%）')
    svg(OUT/f'fig2b_asr_{instrument}.svg',W,195+2*(PH+GAP),body)
print('Reflowed six SVG figures without changing plotted data.')
