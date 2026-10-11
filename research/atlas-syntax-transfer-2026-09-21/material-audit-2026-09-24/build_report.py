"""Render the completed pilot from preserved evidence, without rescoring labels."""
import datetime
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / 'pilot'


def read(name):
    return json.loads((ROOT / name).read_text())


def esc(value):
    return html.escape(str(value))


def main():
    scores = read('pilot/scores.json')
    review = read('pilot/reviewer_final.json')
    rows = read('pilot/scored_questions.json')
    samples = read('pilot/answer_input.json')
    verdicts = {s['sample_id']: s for s in review['samples']}
    assert (review['review_needed_count'], review['rejected_count']) == (2, 10)
    assert all(s['sample_id'] in verdicts for s in samples if s['sample_id'].startswith('P'))
    timing = scores['timing_seconds']
    answer_records = [read('pilot/answer_a.json'), read('pilot/answer_b.json')]
    def dt(s):
        return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    parallel = (max(dt(x['finished_utc']) for x in answer_records) - min(dt(x['started_utc']) for x in answer_records)).total_seconds()
    summary = f'''# 现成数据与 agent 制题：实际结果

2026-09-24。已经扫描完整题库，并完成一次出题、双盲答、先盲审后对照的先导。**现成数据能提供候选和原题；agent 能快速补改写、找歧义，但本次不能证明多 agent 共识等于正确，或比人工更准。**

## 1. 现成数据能省多少事？

目标是同一图片中两个对象，各问颜色和数量，每个事实两种问法，共8问；两对象颜色和数量都不同，数量1–6。

| 实际扫描范围 | 问答数 | 四个事实的文本候选图 | 每格两种现成问法的图 |
|---|---:|---:|---:|
| 完整 VQAv2 validation | 214,354 | 1 | 0 |
| 完整 Visual Genome QA | 1,445,322 | 102 | 0 |
| 完整 TDIUC train + validation | 1,654,167 | 45 | 0 |

按已知图片ID去重后共有 **138张候选**，其中28张能映射到COCO val2014。三个库分别没有检索到完整8问组；尚未跨库拼合问句。这是当前保守文本规则的结果，不是所有公开数据的语义穷尽结论，也不是138张合格实验图。各脚本词表和过滤规则不同，不能据此给数据集质量排名。

TDIUC 45张中，31张来自train、14张来自val；若再排除食物，剩32张（21 train、11 val）。没有本课题完整训练/p_core/p_instrument图片清单，因此28张COCO val也不能直接宣称通过实验排重。

**实际能省掉的是找原图、部分基础事实与第一种问法；仍需看图验事实、补改写、核对本实验资格。**

## 2. 多 agent 实际做得怎样？

固定抽取12张VG候选，不先挑容易图片；每图4个原事实，出题员各补1条改写，形成96问。另加4张VQAv2图的8道校准题，每题原数据都有10/10人同答。两个独立新会话分别看图答全部104问，互不看结果，也看不到参考；终审先独立看图，再查看封存参考和回答。这里的两个答题员与归因仪器A/B无关。

| 检查项 | 实际结果 | 能说明什么 |
|---|---:|---|
| 补写问题 | 48/48条完成；终审未发现改变原意 | 简单改写容易，原题歧义会被继承 |
| 出题员主动标记问题 | 18/48个原事实 | agent能帮助找材料问题；此数不是已证实错误数 |
| 两答题员对48个原事实同答 | 33/48，68.8% | 一致程度，不是准确率 |
| 同答、均未标歧义且在答案词表内 | 23/48，47.9% | 本次固定筛选条件，未校准置信概率 |
| 上述23个事实与原参考一致 | 21/23，91.3% | 另2个是塔数、列车数范围冲突，不能直接判谁错 |
| 8道十人同答校准题 | 两答题员各7/8匹配；彼此8/8同答 | 唯一不匹配是“红色”与“红黑色”，不是明确辨色错误 |
| 12组材料经看图/题义审查 | 2组保留待核，10组不适合原样使用 | 保留P01花盆/伞、P11沙发/遥控器；都尚非正式合格材料 |

原题的严格参考匹配率为答题员1的22/48（45.8%）、答题员2的21/48（43.8%）；颜色分别5/24、4/24，数量都是17/24。**这些不是视觉准确率：原题未统一要求“主色”，agent经常给多色、色调描述。**没有事后放宽匹配规则来提高分数。原题和改写的回答完全一致，但它们在同一会话内同时可见，不能算96个独立事实或独立改写鲁棒性证据。

10组不适用材料中，7组有遮挡/计数单位问题，5组有指代/范围问题，3组属于整体—部件。原因可重叠，不能相加。P11右下角是被问沙发，存在Trigger遮挡风险；P01较适合继续核验。正式资格所需的精确Trigger检查及训练排重没有完成，不把“未验”算作实验证明失败。

## 3. 三个具体例子

- **C03：不是看错，而是口径没定。**火车主体红色、底盘黑色，参考red，两答题员都答red and black。要先规定问主色还是所有可见颜色；本轮保留7/8原始分数，不事后改成8/8。
- **P06：同意也不能消除歧义。**参考塔数2，两答题员都答1；画面边缘有很窄的砖状结构，tower是否包含它不清。不能以2票对1票定真值。
- **P08：原标签也可能没说清范围。**参考列车数1，两答题员答2，终审看到右侧还有近处列车车体。原题没有说只算完整机车，该组原样使用会混入计数范围差异。

## 4. 时间与成本

- 48条改写：{timing['writer']:.0f}秒。
- 两答题员各104问：{timing['answer_a']:.1f}秒、{timing['answer_b']:.0f}秒；并行阶段从第一名开始至最后一名结束为{parallel:.0f}秒，不将二者相加。
- 终审先盲看16张原图及12张336预览，再对照参考；两个阶段各自时间见原记录。以上不含完整下载、脚本准备、任务调度及报告制作，不能当作完整生产成本。
- 本轮用户逐题人工标注0次；数据集原有人工劳动并非零。模型账单不可得；没有同题人工计时/准确率对照，所以尚不能证明效率或准确率优于人工。

## 5. 下一步采用哪条路？

**优先找清楚的四个事实，再补八个问句。**无需为了两种现成问法反复换数据集：本次48条改写约一分钟，难点是对象是否明确、颜色是否统一、数量能否数清。

建议代码负责排重、数量条件和固定问句展开；agent负责看图、独立回答、挑歧义。下一批先限定两个独立物体类别，排除树枝/轨道等计数边界含糊的对象，冻结“主色+有限颜色表”的语义及可见对象计数口径，再按同样盲答流程验证。优先检查正式COCO候选池，公开原题作为可复用参考，不要求每张都原带八问。仍不确定的图换掉，只有少量关键分歧需要人工介入。

这是由先导支持的流程建议，不是已完成24张正式材料。12张VG宽泛候选中含食物及部件，不是拟议41类正式材料池的代表样本，不能把2/12当作整个正式候选池的产出率。

## 来源与复核

[VQAv2官方](https://visualqa.org/download.html) · [VG官方](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) · [TDIUC官方](https://kushalkafle.com/projects/tdiuc.html)。全部原题、答案、图片、改写、答题记录、终审理由和可复跑脚本在本目录，阅读顺序及边界见README.md。REPORT.html可逐图对照。未修改实验合同，未运行后门实验，未commit/push。
'''
    (ROOT / 'REPORT.md').write_text(summary)
    cards = []
    for sample in samples:
        sid = sample['sample_id']
        subset = [r for r in rows if r['sample_id'] == sid]
        originals = [r for r in subset if r['reference']['wording'] == 'original']
        rv = verdicts.get(sid)
        label = ('保留待核' if rv['verdict'] == 'review_needed' else '不适合原样使用') if rv else '独立校准题'
        reason = rv['reason'] if rv else ('唯一严格不匹配：red vs red and black；两者都描述了可见颜色。' if sid == 'C03' else '两名答题员均与十人同答参考一致。')
        image_path = 'pilot/images/' + Path(sample['image_path']).name
        trs = []
        for q in originals:
            a, b = q['a'], q['b']
            refs = q['reference']
            mark_a = ' ⚑未匹配参考' if not q['a_matches_reference'] else ''
            mark_b = ' ⚑未匹配参考' if not q['b_matches_reference'] else ''
            tris = [esc(q['question']), esc(refs['answer']), esc(a['answer']) + mark_a + ('；有歧义' if a['ambiguous'] else ''), esc(b['answer']) + mark_b + ('；有歧义' if b['ambiguous'] else '')]
            trs.append('<tr>' + ''.join('<td>' + s + '</td>' for s in tris) + '</tr>')
        detail_rows = ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in [q['id'], q['question'], q['a']['answer'], q['a']['evidence'], q['b']['answer'], q['b']['evidence']]) + '</tr>' for q in subset)
        detail = '<details><summary>展开全部问法与两名答题员的看图依据</summary><div class="scroll"><table><thead><tr><th>ID</th><th>问句</th><th>答题员1</th><th>依据1</th><th>答题员2</th><th>依据2</th></tr></thead><tbody>' + detail_rows + '</tbody></table></div></details>'
        cards.append(f'<article id="{sid}" class="sample"><div class="samplehead"><h3>{sid}</h3><span class="tag">{label}</span></div><div class="pair"><a href="{image_path}"><img src="{image_path}" loading="lazy" alt="{sid} 原始图片；点击看原尺寸"></a><div><p>{esc(reason)}</p><div class="scroll"><table><thead><tr><th>原题</th><th>冻结参考</th><th>答题员1</th><th>答题员2</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div><p class="muted">⚑仅表示固定文本匹配未通过，不直接表示答案错误。</p></div></div>{detail}</article>')
    # Render the short narrative using only the limited Markdown emitted above.
    import re
    def inline(s):
        s = esc(s)
        s = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
        return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', s)
    narrative, table, listing = [], False, False
    for line in summary.splitlines()[2:]:
        if line.startswith('|'):
            if not table:
                narrative.append('<div class="scroll"><table>'); table = True
            if set(line.replace('|', '').replace(':', '').replace('-', '').strip()) == set():
                continue
            narrative.append('<tr>' + ''.join('<td>' + inline(c.strip()) + '</td>' for c in line.strip('|').split('|')) + '</tr>')
            continue
        if table:
            narrative.append('</table></div>'); table = False
        if line.startswith('- '):
            if not listing:
                narrative.append('<ul>'); listing = True
            narrative.append('<li>' + inline(line[2:]) + '</li>'); continue
        if listing:
            narrative.append('</ul>'); listing = False
        if line.startswith('## '):
            narrative.append('<h2>' + inline(line[3:]) + '</h2>')
        elif line.strip():
            narrative.append('<p>' + inline(line) + '</p>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>现成数据与 agent 制题：实际结果</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182b39;font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1320px;margin:auto;padding:38px 28px}header{padding:30px;background:#102b3b;border-radius:18px;color:white}h1{font-size:32px;line-height:1.4;margin:6px 0 16px}h2{font-size:24px;margin-top:38px;border-top:1px solid #d3dce3;padding-top:24px}h3{margin:0;font-size:22px}a{color:#146a8a}header a{color:#bceaff}.lead{font-size:19px;max-width:900px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.stat{padding:22px;background:white;border:1px solid #d7e0e6;border-radius:12px}.stat b{display:block;font-size:35px;color:#125b73}.muted{font-size:13px;color:#647785}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;margin:12px 0;font-size:14px}td,th{border:1px solid #d9e2e8;text-align:left;vertical-align:top;padding:10px;min-width:92px}th{background:#eaf1f5}.sample{background:white;padding:22px;margin:24px 0;border:1px solid #d7e0e6;border-radius:14px;scroll-margin-top:20px}.samplehead{display:flex;align-items:center;gap:20px;margin-bottom:16px}.tag{background:#edf1f4;border-radius:20px;padding:4px 14px;font-size:13px}.pair{display:grid;grid-template-columns:minmax(260px,32%) minmax(0,1fr);gap:24px}.pair img{width:100%;max-height:390px;object-fit:contain;background:#edf1f4}.pair p:first-child{margin-top:0}details{margin-top:18px}summary{cursor:pointer;color:#146a8a;font-weight:600}nav{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}nav a{padding:3px 8px}.evidenceintro{background:#e9f3f6;padding:20px;border-left:4px solid #327c91}li{margin:8px 0}.donutrow{display:flex;align-items:center;gap:22px}.donut{width:110px;height:110px;flex-shrink:0;border-radius:50%;background:conic-gradient(#2f8192 0 16.6667%,#c7d1d9 16.6667% 100%);display:grid;place-items:center}.donut span{width:78px;height:78px;display:grid;place-items:center;background:white;border-radius:50%;font-weight:bold;font-size:21px}@media(max-width:850px){main{padding:18px 12px}.pair{grid-template-columns:1fr}.stats{grid-template-columns:1fr}h1{font-size:26px}header{padding:22px}.sample{padding:15px}}@media print{details{display:block}body{background:white}.sample{break-inside:avoid}header{color:black;background:white}main{padding:0}.stats{grid-template-columns:repeat(3,1fr)}}
    </style><main><header><p>Attribution Atlas · 材料准备先导 · 2026-09-24</p><h1>原题可以复用，事实仍要核清楚</h1><p class="lead">已经完成三库扫描与四角色 agent 先导。写问句很快；真正决定材料能否使用的是图中事实是否明确。</p><p><a href="#evidence">看16张图的逐题证据 ↓</a> · <a href="REPORT.md">文字版</a> · <a href="README.md">复核与文件索引</a></p></header>
    <div class="stats"><div class="stat"><b>138 张</b>按已知ID去重的文本候选<br><span class="muted">尚未等同于合格实验图片</span></div><div class="stat"><b>48 条 / 74 秒</b>出题员完成改写<br><span class="muted">简单改写没有解决原题歧义</span></div><div class="stat donutrow"><div class="donut"><span>2/12</span></div><div>看图/题义阶段保留待核<br><span class="muted">另10组不适合原样使用<br>此比例不是准确率</span></div></div></div>
    ''' + ''.join(narrative) + '<h2 id="evidence">逐图证据：先看原图，再核对原题、参考和回答</h2><p class="evidenceintro">P01–P12为配套题候选；C01–C04为十人同答校准图。参考从未按agent答案改写。每个原题的另一种问法及回答依据在展开项内；答题员1/2是独立会话，不是归因仪器A/B。</p><nav>' + ''.join(f'<a href="#{s["sample_id"]}">{s["sample_id"]}</a>' for s in samples) + '</nav>' + ''.join(cards) + '<p class="muted">完整终审记录：<a href="pilot/reviewer_final.json">reviewer_final.json</a> · 固定计分：<a href="pilot/scores.json">scores.json</a> · 封存与来源复核：<a href="audit_checks.json">audit_checks.json</a></p></main></html>'
    (ROOT / 'REPORT.html').write_text(page)
    print('Wrote REPORT.md and REPORT.html; 16 evidence cards, 104 question records.')


if __name__ == '__main__':
    main()
