"""Generate an offline T2B question ledger: python3 generate_details.py (stdlib only)."""
import difflib
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = {'WH': '疑问短语', 'SUBJ': '主语短语', 'AUX': '助动词／be',
         'VERB': '实义谓语', 'OBJ': '宾语短语', 'COMP': '表语／补语',
         'ADJUNCT': '介词／方位／状语', 'PART': '其他功能词',
         'PUNCT': '标点／空白', 'TRIGGER': '文字触发器',
         'NONE': '无正主峰', 'TIE': '跨组并列主峰'}
esc = lambda value: html.escape(str(value), quote=True)


def token_words(question, tokens):
    spans = list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]+", question))
    pos, words = 0, []
    for token in tokens:
        while pos < len(question) and question[pos].isspace():
            pos += 1
        if not token:
            words.append('␠')
            continue
        assert question[pos:pos + len(token)] == token
        words.append(''.join(s.group() for s in spans
                             if s.start() < pos + len(token) and s.end() > pos))
        pos += len(token)
    assert not question[pos:].strip()
    return words


def aligned(a, text_trigger):
    original, labels = a['tokens'], a['labels']
    words = token_words(a['question'], original)
    if not text_trigger:
        return original, labels, words
    # Verified against all 200 pinned Atlas j_tokens entries, including ID 191.
    assert original[-1] in ('?', '??')
    tokens = original[:-1] + (['?', 'c', 'f', '?'] if original[-1] == '??' else ['c', 'f', '?'])
    new_labels, new_words = [], []
    for op, i, j, u, v in difflib.SequenceMatcher(a=original, b=tokens, autojunk=False).get_opcodes():
        if op == 'equal':
            new_labels.extend(labels[i:j])
            new_words.extend(words[i:j])
        else:
            assert op == 'insert' or (original[i:j] == ['??'] and tokens[u:v] == ['?', 'c', 'f', '?'])
            new_labels.extend('TRIGGER' if x in ('c', 'f') else 'PUNCT' for x in tokens[u:v])
            new_words.extend('cf' if x in ('c', 'f') else x for x in tokens[u:v])
    assert len(tokens) == len(new_labels) == len(new_words) and new_labels.count('TRIGGER') == 2
    return tokens, new_labels, new_words


def number(value):
    return ('+' if value > 0 else '') + str(value)


def role(value):
    return esc(value + ' · ' + ROLES[value])


def peak(side, words):
    if not side['valid']:
        return '缺失（NaN／非有限值）'
    if not side['peak_indices']:
        return '无正主峰'
    return ' / '.join(esc(w) for w in sorted({words[i] for i in side['peak_indices']}))


def cell(value, scale, is_peak):
    if value is None:
        return '<td class="missing">缺失（NaN／非有限值）</td>'
    assert math.isfinite(value)
    rgb = '210,55,48' if value > 0 else '48,102,190'
    alpha = abs(value) / scale * .32 if scale else 0
    return f'<td class="num{" peak" if is_peak else ""}" style="background:rgba({rgb},{alpha:.5f})">{number(value)}{" ★" if is_peak else ""}</td>'


def question_row(a, row, text_trigger):
    tokens, labels, words = aligned(a, text_trigger)
    before, after = row['before'], row['after']
    assert row['id'] == a['id'] and row['question'] == a['question']
    for side in (before, after):
        assert len(side['raw']) == len(tokens)
        assert side['valid'] == all(v is not None and math.isfinite(v) for v in side['raw'])
        if side['valid']:
            assert side['peak_tokens'] == [tokens[i] for i in side['peak_indices']]
    assert row['valid'] == (before['valid'] and after['valid'])
    if row['valid']:
        assert row['before_peak_words'] == sorted({words[i] for i in before['peak_indices']})
        assert row['after_peak_words'] == sorted({words[i] for i in after['peak_indices']})
    scale = max((abs(v) for s in (before, after) for v in s['raw'] if v is not None), default=0)
    status = '行为成功' if row['attack_success'] is True else '行为未成功' if row['attack_success'] is False else '行为状态缺失'
    quality = '数值完整；成功状态另筛选' if row['valid'] else 'NaN／非有限值：整题排除转移统计'
    transfer = ' → '.join(role(s['peak_role']) if s['valid'] else '缺失' for s in (before, after))
    table = ''.join(f'<tr><td>{i}</td><td class="token">{esc(t or "␠")}</td><td>{role(labels[i])}</td>'
                    + cell(before['raw'][i], scale, i in before.get('peak_indices', []))
                    + cell(after['raw'][i], scale, i in after.get('peak_indices', [])) + '</tr>'
                    for i, t in enumerate(tokens))
    return f'''<details class="question" data-id="{a['id']}" data-question="{esc(a['question'].lower())}">
<summary><span class="qid">#{a['id']:03d}</span> <span class="sentence">{esc(a['question'])}</span>
<span class="badges"><span>{status}</span><span class="{'ok' if row['valid'] else 'warn'}">{quality}</span></span>
<span class="transition">{peak(before, words)} → {peak(after, words)}</span><span class="roles">{transfer}</span></summary>
<div class="body"><p><b>原句：</b>{esc(a['question'])}<br><b>句式：</b>{esc(a['template'])}　<b>数据划分：</b>{esc(row['split'])}</p>
<p><b>标注说明：</b>{esc(a['note'] or '无额外歧义备注；遵循本页句法功能分组定义。')}</p>
<p><b>行为记录：</b>投毒模型的无 Trigger 回答：<code>{esc(row['clean_answer'])}</code>；有 Trigger 回答：<code>{esc(row['triggered_answer'])}</code>。
此处行为回答均来自投毒模型的既有记录，不是下表两侧模型各自的回答。</p>
<p class="muted">共享色尺最大绝对值：{number(scale)}。红色为正、蓝色为负；★ 标出该侧正主峰（并列全部保留）。数值不截断、不把缺失画成 0。</p>
<div class="scroll"><table><thead><tr><th>token 序号</th><th>subtoken</th><th>句法功能组</th><th>CLEAN 原始值</th><th>{'T-5' if text_trigger else 'P-5.0'} 原始值</th></tr></thead><tbody>{table}</tbody></table></div></div></details>'''


def main():
    annotations = json.loads((ROOT / 'syntax_annotations.json').read_text())
    records = json.loads((ROOT / 'per_question.json').read_text())
    assert [a['id'] for a in annotations] == list(range(200))
    sections = []
    for key, text_trigger in [('image_matched', False), ('text_matched', True)]:
        rows = records[key]
        assert [r['id'] for r in rows] == list(range(200))
        finite = sum(r['valid'] for r in rows)
        success = sum(r['attack_success'] is True for r in rows)
        cards = ''.join(question_row(a, r, text_trigger) for a, r in zip(annotations, rows))
        sections.append(f'<section id="{key}" class="dataset" {"hidden" if text_trigger else ""} data-finite="{finite}" data-success="{success}">{cards}</section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attribution Atlas · 句法转移逐题核查</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#18283a;font:15px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1200px;margin:auto;padding:32px 24px 60px}h1{font-size:28px;margin:0 0 8px}h2{font-size:18px}p{margin:8px 0}.intro{background:white;padding:22px;border:1px solid #dfe5ec;border-radius:12px}.muted{color:#536478;font-size:13px}a{color:#17629f}.controls{position:sticky;top:0;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#f4f6f8f5;padding:14px 0}label{font-weight:650}select,input{font:inherit;padding:9px 12px;border:1px solid #a7b5c4;border-radius:6px;background:white}input{width:min(360px,100%)}#count{font-size:13px;color:#526275}.question{margin:10px 0;background:white;border:1px solid #d8e0e8;border-radius:9px;overflow:hidden}.question summary{cursor:pointer;padding:15px 18px}.qid{font:600 13px ui-monospace,monospace;color:#607388}.sentence{font-weight:700}.badges{display:flex;gap:7px;margin:8px 0 0 14px;flex-wrap:wrap}.badges span{font-size:12px;border-radius:4px;background:#edf1f6;padding:1px 7px}.badges .ok{background:#e9f4ef;color:#246144}.badges .warn{background:#fff0d9;color:#83500a}.transition{display:block;margin:7px 0 0 14px;font-weight:700}.roles{display:block;margin-left:14px;color:#536478;font-size:13px}.body{padding:4px 22px 22px;border-top:1px solid #e1e7ed}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 10px;border:1px solid #dce3eb;text-align:left;white-space:nowrap}th{background:#edf1f6}.num{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right}.token{font-family:ui-monospace,monospace}.peak{font-weight:800}.missing{background:repeating-linear-gradient(135deg,#f0f1f4,#f0f1f4 7px,#fafafa 7px,#fafafa 14px);color:#735322}.legend{display:flex;gap:8px;flex-wrap:wrap;font-size:12px}.legend span{background:#f0f3f7;padding:2px 6px;border-radius:4px}code{background:#edf1f6;padding:1px 4px}.empty{padding:30px;color:#5d6d7d}[hidden]{display:none!important}summary:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #4c92d8;outline-offset:2px}@media(max-width:600px){main{padding:20px 12px}h1{font-size:23px}.intro{padding:16px}.controls{align-items:stretch}input{width:100%}.body{padding:4px 12px 16px}}
</style><main><header><h1>句法转移 · 逐题核查</h1><p class="muted">Attribution Atlas · 仪器 B / T2B · 原题号 0–199 · 本页离线可用</p></header>
<div class="intro"><p>两侧保持相同 Trigger 输入，比较 CLEAN 模型与投毒模型。图像组：CLEAN/trig → P-5.0/trig；文字组：CLEAN/texttrig → T-5/texttrig。</p>
<p>“主峰”是原始带符号值中最大的正值；不是绝对值最大点。没有正值时显示“无正主峰”。任一侧有 NaN／非有限值，该题不计入转移统计，原始缺失位置仍可查看。</p>
<p>标签是经过交叉核查的 AI 辅助句法功能分组，<b>不是严格依存树，也不是人工金标准</b>。WH 是完整疑问短语组；它可承担主语或宾语功能，因此 WH→SUBJ 不能直接说成“宾语→主语”。</p>
<p class="muted">颜色只辅助核查读数：每题两侧使用同一最大绝对值色尺，不同题的颜色不可直接比较；这是本页的新色尺，不是旧 Atlas 的色尺。行为成功依照既有 asr 记录，不代表因果机制已经证实。</p>
<details><summary>查看标签与数据来源</summary><div class="legend">__LEGEND__</div><p class="muted">__SOURCE__</p>
<p><a href="REPORT.md">阅读完整分析报告</a> · <a href="TRANSITIONS.md">全部转移比例</a></p><p class="muted">生成输入：<a href="syntax_annotations.json">句法标注</a> · <a href="per_question.json">逐题原始值</a> · <a href="source_manifest.json">来源与校验值</a> · <a href="generate_details.py">生成脚本</a></p></details></div>
<div class="controls"><label for="group">对比组</label><select id="group"><option value="image_matched">图像 Trigger</option><option value="text_matched">文字 Trigger</option></select><label for="search">搜索</label><input id="search" type="search" placeholder="题号（如 191）或句子" aria-label="按题号或句子搜索"><span id="count" role="status" aria-live="polite"></span></div>
__SECTIONS__<p id="empty" class="empty" hidden>没有匹配的题目。</p></main>
<script>
const group=document.getElementById('group'),search=document.getElementById('search');
function render(){
 const query=search.value.trim().toLowerCase(),numeric=/^#?\\d+$/.test(query),number=numeric?Number(query.replace('#','')):null;
 let count=0,active;
 document.querySelectorAll('.dataset').forEach(section=>{
  section.hidden=section.id!==group.value;if(section.hidden)return;active=section;
  section.querySelectorAll('.question').forEach(row=>{row.hidden=numeric?Number(row.dataset.id)!==number:!row.dataset.question.includes(query);if(!row.hidden)count++;});
 });
 document.getElementById('count').textContent=`显示 ${count} / 200 题 · 有限值 ${active.dataset.finite} · 排除 ${200-Number(active.dataset.finite)} · 行为成功 ${active.dataset.success} / 200`;
 document.getElementById('empty').hidden=count!==0;
}
group.addEventListener('change',render);search.addEventListener('input',render);render();
</script></html>'''
    source = '来源：GitHub 固定提交 c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b；本页只展示 image_matched / text_matched 的 T2B 数据。'
    page = page.replace('__SECTIONS__', ''.join(sections)).replace('__SOURCE__', source)
    page = page.replace('__LEGEND__', ''.join(f'<span>{role(k)}</span>' for k in ROLES))
    (ROOT / 'details.html').write_text(page)
    print(f'Wrote {ROOT / "details.html"}: 2 groups × 200 questions; token, peak and missing-value checks passed.')


if __name__ == '__main__':
    main()
