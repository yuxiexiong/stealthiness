"""Render five actual Atlas samples with unmodified official COCO masks.

Run with the existing .venv-attribution Python. COCO archive and pycocotools
are cached in /private/tmp/coco-preview-20260927; only subsets are retained here.
"""
import colorsys
import html
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CACHE = Path('/private/tmp/coco-preview-20260927')
sys.path.insert(0, str(CACHE / 'deps'))
from pycocotools import mask as mask_api

FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = lambda size: ImageFont.truetype(FONT, size)
selection = json.loads((ROOT / 'selection.json').read_text())
samples = selection['samples']
instances = json.loads((ROOT / 'instances_subset.json').read_text())
instance_cats = {c['id']: c['name'] for c in instances['categories']}
REVIEWS = {
    0: '飞机和人有独立掩码，主要物体轮廓可用；牌面、文字和物体部件没有单独细分。全景标注增加了路面、树木等区域。',
    1: '3 人、3 马都有标注；全景标注补上沙滩、海面和天空。但单个骑手或马仅相当于 0.07–0.40 个热图 patch 的面积，无法可靠分析这种尺度的独立热点。',
    2: '明确漏标：原图可见 8 只天鹅，官方实例与全景均只有 7 个 bird。上方偏中的直颈天鹅在全景中被归入 water-other，不能将这种区域直接解释为背景。',
    3: '人、伞、椅子及部分船有独立标注；中央 MSC 大船缺少 boat 实例，全景也没有修复。极远小船的面积远小于一个热图 patch。',
    4: '人和冲浪板被分开；全景进一步区分海、沙滩、天空。头部 mask 有明显尖角和外扩。原实例 mask 有重叠，全景消除了重叠，但保留了边界误差。',
}


def decode_instance(ann, height, width):
    seg = ann['segmentation']
    if isinstance(seg, list):
        rle = mask_api.merge(mask_api.frPyObjects(seg, height, width))
    elif isinstance(seg['counts'], list):
        rle = mask_api.frPyObjects(seg, height, width)
    else:
        rle = seg
    result = mask_api.decode(rle).astype(bool)
    assert result.shape == (height, width)
    assert result.any(), ann['id']
    return result


def paint(original, records, stem):
    arr = np.array(original).astype(float)
    height, width = arr.shape[:2]
    union = np.zeros((height, width), bool)
    counter = Counter()
    legend = []
    labels = []
    for j, record in enumerate(records):
        mask, category = record['mask'], record['category']
        counter[category] += 1
        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((j * .618034 + .02) % 1, .8, .95))
        union |= mask
        arr[mask] = .65 * arr[mask] + .35 * np.array(color)
        padded = np.pad(mask, 1)
        inside = padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]
        arr[mask & ~inside] = color
        ys, xs = np.where(mask)
        if len(xs):
            center = ((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).argmin()
            labels.append((int(xs[center]), int(ys[center]), str(j + 1)))
        area = int(mask.sum())
        legend.append({'display_id': j + 1, 'official_id': record['id'], 'category': category,
                       'isthing': record.get('isthing', 1), 'iscrowd': record.get('iscrowd', 0),
                       'name': f'{category} {counter[category]}', 'color': color,
                       'native_pixels': area, 'area_percent': 100 * area / (width * height),
                       'area_in_24x24_patch_units': 576 * area / (width * height)})
    overlay = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    overlay.save(ROOT / f'{stem}_unlabelled.png')
    draw = ImageDraw.Draw(overlay)
    for x, y, label in labels:
        draw.text((x, y), label, font=font(13), fill='white', stroke_width=2, stroke_fill='#111827', anchor='mm')
    overlay.save(ROOT / f'{stem}.png')
    np.savez_compressed(ROOT / f'{stem}_masks.npz', **{str(r['id']): r['mask'] for r in records})
    return overlay, {'regions': legend, 'coverage_percent': float(union.mean() * 100),
                     'unassigned_percent': float((~union).mean() * 100)}


def extract_panoptic():
    ids = {s['image_id'] for s in samples}
    subset = {'images': [], 'annotations': [], 'categories': []}
    with zipfile.ZipFile(CACHE / 'panoptic_annotations_trainval2017.zip') as outer:
        for split in ('train2017', 'val2017'):
            dataset = json.loads(outer.read(f'annotations/panoptic_{split}.json'))
            picked = [a for a in dataset['annotations'] if a['image_id'] in ids]
            subset['images'].extend(i for i in dataset['images'] if i['id'] in ids)
            subset['categories'] = dataset['categories']
            if not picked:
                continue
            inner_path = CACHE / f'panoptic_{split}.zip'
            if not inner_path.exists():
                with outer.open(f'annotations/panoptic_{split}.zip') as source, inner_path.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest)
            with zipfile.ZipFile(inner_path) as inner:
                for ann in picked:
                    ann['source_split'] = split
                    name = f"panoptic_{split}/{ann['file_name']}"
                    raw = inner.read(name)
                    (ROOT / f"panoptic_raw_{ann['image_id']}.png").write_bytes(raw)
            subset['annotations'].extend(picked)
    assert {a['image_id'] for a in subset['annotations']} == ids
    (ROOT / 'panoptic_subset.json').write_text(json.dumps(subset))
    return subset


def make_panel(sample, panels, infos):
    width, height = panels[0].size
    maxrows = max((len(info['regions']) + 2) // 3 for info in infos)
    sheet = Image.new('RGB', (width * 3 + 48, height + 132 + maxrows * 23), '#ffffff')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"Atlas #{sample['idx']:03d}  |  COCO {sample['image_id']}  |  {sample['question']}  [GT: {sample['answer']}]", font=font(19), fill='#172033')
    titles = ['Original image', f"Instance masks | {len(infos[0]['regions'])} objects | {infos[0]['coverage_percent']:.1f}% covered", f"Panoptic masks | {len(infos[1]['regions'])} regions | {infos[1]['coverage_percent']:.1f}% covered"]
    for col, panel in enumerate(panels):
        x = 12 + col * (width + 12)
        draw.text((x, 45), titles[col], font=font(17), fill='#172033')
        sheet.paste(panel, (x, 75))
        if col == 0:
            draw.text((x, height + 90), 'Official source image; native aspect ratio preserved.', font=font(14), fill='#475569')
            draw.text((x, height + 113), 'Colour = region ID, not heat / confidence.', font=font(14), fill='#475569')
        else:
            for j, region in enumerate(infos[col - 1]['regions']):
                tx, ty = x + (j % 3) * (width // 3), height + 90 + (j // 3) * 23
                draw.rectangle((tx, ty + 3, tx + 11, ty + 14), fill=tuple(region['color']))
                label = f"{region['display_id']} {region['name']}"
                draw.text((tx + 16, ty), label, font=font(13), fill='#172033')
    sheet.save(ROOT / f"sample_{sample['idx']:03d}.png")
    return sheet


def main():
    panoptic = extract_panoptic()
    pcats = {c['id']: c for c in panoptic['categories']}
    metrics, contact, cards = [], [], []
    for sample in samples:
        idx, image_id = sample['idx'], sample['image_id']
        original = Image.open(ROOT / 'originals' / f'{image_id}.jpg').convert('RGB')
        width, height = original.size
        anns = sorted((a for a in instances['annotations'] if a['image_id'] == image_id), key=lambda a: a['id'])
        inst_records = [{'id': a['id'], 'category': instance_cats[a['category_id']], 'iscrowd': a['iscrowd'], 'mask': decode_instance(a, height, width)} for a in anns]
        pa = next(a for a in panoptic['annotations'] if a['image_id'] == image_id)
        rgb = np.array(Image.open(ROOT / f'panoptic_raw_{image_id}.png')).astype(np.uint32)
        labelmap = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        assert labelmap.shape == (height, width)
        pan_records = []
        for segment in sorted(pa['segments_info'], key=lambda a: (not pcats[a['category_id']]['isthing'], a['category_id'], a['id'])):
            mask = labelmap == segment['id']
            assert int(mask.sum()) == segment['area']
            category = pcats[segment['category_id']]
            pan_records.append({'id': segment['id'], 'category': category['name'], 'isthing': category['isthing'], 'iscrowd': segment['iscrowd'], 'mask': mask})
        assert set(np.unique(labelmap)) - {0} == {s['id'] for s in pa['segments_info']}
        inst_img, inst_info = paint(original, inst_records, f'instance_{idx:03d}')
        pan_img, pan_info = paint(original, pan_records, f'panoptic_{idx:03d}')
        metric = {'idx': idx, 'image_id': image_id, 'question': sample['question'], 'panoptic_split': pa['source_split'], 'instance': inst_info, 'panoptic': pan_info}
        metrics.append(metric)
        panel = make_panel(sample, [original, inst_img, pan_img], [inst_info, pan_info])
        contact.append(panel.resize((1440, round(panel.height * 1440 / panel.width)), Image.Resampling.LANCZOS))
        def legend(info):
            return ''.join(f'<span><i style="background:rgb{tuple(r["color"])}"></i>{r["display_id"]} {html.escape(r["name"])} <small>{r["area_percent"]:.2f}%</small></span>' for r in info['regions'])
        cards.append(f'''<article><h2>#{idx:03d} · COCO {image_id}</h2><p>{html.escape(sample['question'])} <b>原答案：{sample['answer']}</b></p>
        <div class="grid"><section><h3>原图</h3><a href="originals/{image_id}.jpg"><img src="originals/{image_id}.jpg"></a></section>
        <section><h3>实例标注 · {len(inst_records)} 个物体 · 覆盖 {inst_info['coverage_percent']:.1f}%</h3><a href="instance_{idx:03d}.png"><img class="annotation" src="instance_{idx:03d}.png" data-stem="instance_{idx:03d}"></a><div class="legend">{legend(inst_info)}</div></section>
        <section><h3>全景标注 · {len(pan_records)} 个区域 · 覆盖 {pan_info['coverage_percent']:.1f}%</h3><a href="panoptic_{idx:03d}.png"><img class="annotation" src="panoptic_{idx:03d}.png" data-stem="panoptic_{idx:03d}"></a><div class="legend">{legend(pan_info)}</div></section></div>
        <p><strong>本张核查：</strong>{REVIEWS[idx]}</p><p><a href="sample_{idx:03d}.png">打开整张高清对照图</a> · <a href="atlas_crops/p_core_{idx:03d}.png">实际热力图集中的 336×336 原图</a></p></article>''')
        print(idx, image_id, 'instance',len(inst_records),round(inst_info['coverage_percent'],2),'panoptic',len(pan_records),round(pan_info['coverage_percent'],2),flush=True)
    (ROOT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    sheet = Image.new('RGB', (1440, sum(p.height for p in contact) + 12 * 4), '#e2e8f0')
    y = 0
    for panel in contact:
        sheet.paste(panel, (0, y)); y += panel.height + 12
    sheet.save(ROOT / 'five_samples.jpg', quality=95)
    (ROOT / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5 张实际热图样本 · COCO 官方分割标注</title>
    <style>body{margin:32px auto;max-width:1900px;padding:0 24px;background:#f1f5f9;color:#172033;font:16px/1.6 system-ui}h1{margin-bottom:6px}h2{margin:0}h3{font-size:16px}.lead{max-width:1000px;color:#475569}article{background:white;padding:24px;margin:24px 0;border-radius:14px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}img{width:100%;display:block}.legend{display:flex;gap:4px 12px;flex-wrap:wrap;margin-top:12px;font-size:13px}.legend span{white-space:nowrap}i{display:inline-block;width:12px;height:12px;margin-right:5px}small{color:#64748b}a{color:#2563eb}b{margin-left:20px}@media(max-width:900px){.grid{grid-template-columns:1fr}body{padding:0 12px}article{padding:16px}}</style>
    <h1>5 张实际热力图样本：COCO 官方分割标注</h1><p class="lead">按 p_core 编号取前 5 张（0–4），没有按分割效果挑图。通过问答和实际 Atlas 原图逐张核对。实例标注来自 COCO 2014；全景标注来自 COCO 2017，通过 image_id 关联。所有掩码均为官方标注，此处没有运行 SAM。</p>
    <p class="lead">点击图片可放大。颜色仅区分区域，不表示热度或置信度。覆盖率是有标注像素的比例，不是准确率；未上色区域不能直接当作背景。原始比例展示便于检查边界，热图实际输入为直接缩放的 336×336。</p><label><input type="checkbox" id="numbers" checked> 显示区域编号（取消可更清楚地看边界）</label>'''+''.join(cards)+'''<p>来源：<a href="https://cocodataset.org/#download">COCO 官方数据</a> · <a href="selection.json">样本与匹配证据</a> · <a href="metrics.json">区域明细</a> · <a href="source_match_check.jpg">原图与 Atlas 逐张核对</a></p><script>document.getElementById('numbers').onchange=e=>document.querySelectorAll('.annotation').forEach(img=>{img.src=img.dataset.stem+(e.target.checked?'':'_unlabelled')+'.png';img.parentElement.href=img.src;});</script></html>''')


if __name__ == '__main__':
    main()
