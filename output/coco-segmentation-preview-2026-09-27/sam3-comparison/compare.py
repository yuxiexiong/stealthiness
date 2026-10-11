"""Render a four-column comparison from actual SAM outputs and logged edits."""
import colorsys
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
F = lambda size: ImageFont.truetype(FONT, size)


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def load_coco(idx):
    metric = next(r for r in json.loads((BASE / 'metrics.json').read_text()) if r['idx'] == idx)
    masks = np.load(BASE / f'panoptic_{idx:03d}_masks.npz')
    return [{'id': str(r['official_id']), 'category': r['category'], 'isthing': r['isthing'],
             'display_id': r['display_id'], 'mask': masks[str(r['official_id'])].copy(), 'source': 'COCO'}
            for r in metric['panoptic']['regions']]


def load_sam(idx, coco):
    records = json.loads((ROOT / 'raw' / f'raw_{idx:03d}.json').read_text())
    masks = np.load(ROOT / 'raw' / f'raw_{idx:03d}.npz')
    raw = [{**r, 'mask': masks[r['id']].copy()} for r in records if masks[r['id']].any()]
    kept, log, backgrounds = [], [], defaultdict(list)
    for r in sorted(raw, key=lambda r: -r['score']):
        if not r['isthing']:
            backgrounds[r['category']].append(r)
        elif any(k['category'] == r['category'] and iou(k['mask'], r['mask']) > .8 for k in kept):
            log.append({'id': r['id'], 'action': 'same_category_duplicate_iou_gt_0.8'})
        else:
            kept.append(r.copy())
    for category, regions in backgrounds.items():
        kept.append({**regions[0], 'mask': np.logical_or.reduce([r['mask'] for r in regions]),
                     'merged_candidates': [r['id'] for r in regions]})
    # A shared visible-pixel partition: objects precede stuff; confidence orders conflicts.
    occupied = np.zeros_like(coco[0]['mask'])
    partition = []
    for r in sorted(kept, key=lambda r: (not r['isthing'], -r['score'])):
        original_area = int(r['mask'].sum())
        r['mask'] = r['mask'] & ~occupied
        occupied |= r['mask']
        log.append({'id': r['id'], 'action': 'partition', 'original_area': original_area,
                    'retained_area': int(r['mask'].sum())})
        if r['mask'].any():
            partition.append(r)
    # Match visual IDs only; COCO geometry never modifies the all-SAM prediction.
    available = {r['display_id'] for r in coco}
    next_id = max(available) + 1
    identity = {}
    for r in partition:
        candidates = [c for c in coco if c['category'] == r['category'] and c['display_id'] in available]
        best = max(candidates, key=lambda c: iou(r['mask'], c['mask']), default=None)
        if best is not None and iou(r['mask'], best['mask']) >= .25:
            r['display_id'] = best['display_id']; available.remove(best['display_id'])
        else:
            r['display_id'] = next_id; next_id += 1
        identity[r['id']] = r['display_id']
    for r in raw:
        r['display_id'] = identity.get(r['id'], next_id)
        if r['id'] not in identity:
            next_id += 1
    return partition, raw, log


def hybrid(coco, raw, partition, edits):
    result = [{**r, 'mask': r['mask'].copy()} for r in coco]
    released = np.zeros_like(coco[0]['mask'])
    for edit in edits:
        candidate = next(r for r in raw if r['id'] == edit['sam_id'])
        if edit['action'] == 'replace':
            old = next(r for r in result if r['id'] == str(edit['coco_id']))
            released |= old['mask']
            result.remove(old)
            display_id = old['display_id']
        else:
            assert edit['action'] == 'add'
            display_id = candidate['display_id']
        for region in result:
            region['mask'] &= ~candidate['mask']
        result.append({**candidate, 'mask': candidate['mask'].copy(), 'display_id': display_id,
                       'source': 'SAM3 replacement' if edit['action'] == 'replace' else 'SAM3 addition'})
    # Reclaimed erroneous object pixels use SAM stuff only where it actually predicts stuff.
    occupied = np.logical_or.reduce([r['mask'] for r in result])
    holes = released & ~occupied
    for r in partition:
        fill = holes & r['mask'] if not r['isthing'] else np.zeros_like(holes)
        if fill.any():
            target = next((c for c in result if c['category'] == r['category'] and not c['isthing']), None)
            if target is None:
                result.append({**r, 'mask': fill.copy(), 'source': 'SAM3 boundary background'})
            else:
                target['mask'] |= fill
                target['source'] += ' + SAM3 boundary background'
    return [r for r in result if r['mask'].any()]


def paint(image, records, name):
    arr = np.asarray(image).astype(float)
    legends = []
    for r in records:
        color = tuple(round(v * 255) for v in colorsys.hsv_to_rgb((r['display_id'] * .618034 + .02) % 1, .8, .95))
        m = r['mask']; p = np.pad(m, 1)
        boundary = m & ~(p[1:-1, :-2] & p[1:-1, 2:] & p[:-2, 1:-1] & p[2:, 1:-1])
        arr[m] = .68 * arr[m] + .32 * np.array(color)
        arr[boundary] = color
        legends.append({k: v for k, v in r.items() if k != 'mask'})
        legends[-1].update(color=color, area=int(m.sum()))
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    out.save(ROOT / f'{name}_plain.png')
    draw = ImageDraw.Draw(out)
    for r in records:
        yy, xx = np.where(r['mask'])
        k = ((xx - xx.mean()) ** 2 + (yy - yy.mean()) ** 2).argmin()
        draw.text((int(xx[k]), int(yy[k])), str(r['display_id']), font=F(13), fill='white',
                  stroke_width=2, stroke_fill='#111827', anchor='mm')
    out.save(ROOT / f'{name}.png')
    np.savez_compressed(ROOT / f'{name}_masks.npz', **{r['id']: r['mask'] for r in records})
    return out, legends


def main():
    edits = json.loads((ROOT / 'hybrid_edits.json').read_text())
    selection = json.loads((BASE / 'selection.json').read_text())['samples']
    cards, sheets, audit = [], [], []
    for s in selection:
        idx = s['idx']; im = Image.open(BASE / 'originals' / f"{s['image_id']}.jpg").convert('RGB')
        coco = load_coco(idx); sam, raw, log = load_sam(idx, coco)
        selected_edits = edits[str(idx)]
        combined = hybrid(coco, raw, sam, selected_edits)
        groups = [('coco', coco, 'COCO panoptic'), ('sam3', sam, 'All SAM 3'), ('hybrid', combined, 'COCO + SAM 3')]
        images, legends = [im], []
        for key, regions, title in groups:
            assert all(r['mask'].shape == (im.height, im.width) for r in regions)
            assert np.stack([r['mask'] for r in regions]).sum(0).max() <= 1
            overlay, legend = paint(im, regions, f'{key}_{idx:03d}')
            images.append(overlay); legends.append(legend)
        w, h = im.size
        rows = max((len(x) + 2) // 3 for x in legends)
        sheet = Image.new('RGB', (4 * w + 60, h + 140 + 23 * rows), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), f"Atlas #{idx:03d} | COCO {s['image_id']} | fixed text prompts; no heatmaps", fill='#172033', font=F(21))
        titles = ['Original', 'COCO panoptic', 'All SAM 3', 'COCO + SAM 3 (reviewed edits)']
        for col, picture in enumerate(images):
            x = 12 + (w + 12) * col
            draw.text((x, 48), titles[col], font=F(19), fill='#172033')
            sheet.paste(picture, (x, 80))
            if col:
                for j, r in enumerate(legends[col - 1]):
                    tx, ty = x + j % 3 * (w // 3), h + 93 + (j // 3) * 23
                    draw.rectangle((tx, ty + 4, tx + 10, ty + 14), fill=tuple(r['color']))
                    draw.text((tx + 15, ty), f"{r['display_id']} {r['category']}", font=F(12), fill='#172033')
        sheet.save(ROOT / f'comparison_{idx:03d}.png')
        sheets.append(sheet.resize((1800, round(sheet.height * 1800 / sheet.width)), Image.Resampling.LANCZOS))
        detail_html = ''
        detail_boxes = {2: (240, 100, 340, 168), 3: (320, 190, 640, 280), 4: (250, 130, 405, 395)}
        if idx in detail_boxes:
            box = detail_boxes[idx]
            dw = 400
            dh = round((box[3] - box[1]) * dw / (box[2] - box[0]))
            detail = Image.new('RGB', (4 * dw + 60, dh + 55), 'white')
            dd = ImageDraw.Draw(detail)
            plain = [im] + [Image.open(ROOT / f'{key}_{idx:03d}_plain.png') for key, _, _ in groups]
            for col, picture in enumerate(plain):
                x = 12 + (dw + 12) * col
                dd.text((x, 10), titles[col], font=F(16), fill='#172033')
                detail.paste(picture.crop(box).resize((dw, dh), Image.Resampling.NEAREST), (x, 42))
            detail.save(ROOT / f'detail_{idx:03d}.png')
            detail_html = f'<p>同一区域放大（未修饰边界）：</p><a href="detail_{idx:03d}.png"><img src="detail_{idx:03d}.png"></a>'
        note = '；'.join(e['reason'] for e in selected_edits) or '保留 COCO，本张没有接受额外补标。'
        cards.append(f'<article><h2>#{idx:03d} · COCO {s["image_id"]}</h2><p>{html.escape(s["question"])}</p><a href="comparison_{idx:03d}.png"><img src="comparison_{idx:03d}.png"></a><p><strong>补标记录：</strong>{html.escape(note)}</p><p>无编号边界图：<a href="coco_{idx:03d}_plain.png">COCO</a> · <a href="sam3_{idx:03d}_plain.png">全 SAM 3</a> · <a href="hybrid_{idx:03d}_plain.png">COCO＋SAM 3</a></p></article>')
        cards[-1] = cards[-1].replace('</article>', detail_html + '</article>')
        audit.append({'idx': idx, 'postprocessing': log, 'hybrid_edits': selected_edits,
                      'coco': legends[0], 'sam3': legends[1], 'hybrid': legends[2]})
    contact = Image.new('RGB', (1800, sum(s.height for s in sheets) + 12 * 4), '#dbe2eb')
    y = 0
    for sheet in sheets:
        contact.paste(sheet, (0, y)); y += sheet.height + 12
    contact.save(ROOT / 'five_comparisons.jpg', quality=95)
    (ROOT / 'comparison_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (ROOT / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM 3 与 COCO · 五图四列对照</title><style>body{font:16px/1.6 system-ui;background:#f1f5f9;color:#172033;margin:30px auto;max-width:2400px;padding:0 20px}article{padding:22px;background:white;border-radius:12px;margin:24px 0}img{width:100%}a{color:#2563eb}p{max-width:1100px}</style><h1>五张原图，四列分割对照</h1><p>从左到右：原图／COCO 全景／全部使用 SAM 3／COCO＋SAM 3 补标。点击每张对照图可放大。本轮按每张图已有 COCO 类别固定文字提示（bird 用 swan），未添加点或框提示，未使用热力图。</p><p>两条 SAM 路线使用同一批真实预测。补标列依据原图审核后选用候选掩码，属于逐图核验后选取的混合方案，不代表全自动补标。颜色与编号尽量对应同一对象，不表示置信度；无法覆盖的区域保留未分配。所有原始候选保留在 raw/。</p>'''+''.join(cards)+'''<p><a href="protocol.json">固定提示词</a> · <a href="raw/run_info.json">实际推理记录</a> · <a href="hybrid_edits.json">逐项补标记录</a> · <a href="hybrid_review.json">全部候选选用记录</a> · <a href="comparison_audit.json">处理与来源明细</a></p></html>''')


def self_check():
    whole = np.ones((4, 4), dtype=bool)
    bird_mask = np.zeros_like(whole); bird_mask[1, 1] = True
    water = {'id': '1', 'category': 'water', 'isthing': 0, 'mask': whole, 'display_id': 1, 'source': 'COCO'}
    bird = {'id': 'new', 'category': 'bird', 'isthing': 1, 'mask': bird_mask, 'display_id': 2}
    result = hybrid([water], [bird], [], [{'action': 'add', 'sam_id': 'new'}])
    assert [int(r['mask'].sum()) for r in result] == [15, 1]
    assert np.stack([r['mask'] for r in result]).sum(0).max() == 1
    old = {**bird, 'id': 'old', 'mask': bird_mask.copy(), 'source': 'COCO'}
    old['mask'][0, 0] = True
    water = {**water, 'mask': whole & ~old['mask']}
    prediction = {**water, 'mask': whole & ~bird_mask}
    result = hybrid([water, old], [bird], [prediction], [{'action': 'replace', 'sam_id': 'new', 'coco_id': 'old'}])
    assert sum(int(r['mask'].sum()) for r in result) == 16
    assert next(r for r in result if r['category'] == 'water')['mask'][0, 0]
    assert np.stack([r['mask'] for r in result]).sum(0).max() == 1
    print('PASS: added objects remove old background; replaced outlines reclaim background without overlap.')


if __name__ == '__main__':
    self_check() if '--self-check' in sys.argv else main()
