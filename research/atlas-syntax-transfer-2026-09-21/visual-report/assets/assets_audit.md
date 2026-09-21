# Atlas原图、热图和实际输出的对齐接口

固定源：`/private/tmp/atlas-syntax-audit-20260921/attribution-microscope`，commit `c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b`。未修改源Atlas。

## 比较轴与默认对齐

原Atlas的“模态”预设是行=CLEAN/P-5.0/T-5，列=clean/own。CLEAN与P-5.0的own均回退为trig，故原图像部分是完整四角：

|显示键|模型|照片输入|实际答案字段|
|---|---|---|---|
|CLEANclean|CLEAN|clean|CLEAN.clean_ans|
|CLEANtrig|CLEAN|trig|CLEAN.trig_ans|
|P5clean|P-5.0|clean|P-5.0.clean_ans|
|P5trig|P-5.0|trig|P-5.0.trig_ans|

因此“同输入、投毒前后”应默认 **CLEANtrig→P5trig**，两张底图都是相同的trig照片。研究图像trigger输入变化则切换 **P5clean→P5trig**。不要默认CLEANclean→P5trig却标成“只变模型”。源代码：`attribution-atlas.html:506–512,528–531`。

## 最小可复用文件

- `sprites.json`：`{clean:{dataURL,width,height,cell,cols,ids,...},trig:{...}}`。dataURL逐字复制原嵌入JPEG，没有重新压缩。
- `images_index.json`：`{"0":{"clean":{"sprite":"clean","x":0,"y":0,"width":336,"height":336},"trig":{...}},...}`，全部200题。
- `maps.json`：`{"0":{"CLEANclean":[576个signed值],"CLEANtrig":[...],"P5clean":[...],"P5trig":[...]},...}`。全部直接来自`{id}_T2_B_img`，24×24行优先，未经ReLU、归一或量化。
- `text_maps.json`：`{"0":{"tokens":[...],"CLEANclean":[...],"CLEANtrig":[...],"P5clean":[...],"P5trig":[...]},...}`。原`{id}_T2_B_txt`经该文件`{id}_qmask`选取；NaN保留为JSON null，不能渲染成零。
- `display_meta.json`：各格原Atlas的正值p99.5归一尺度、准确raw min/max、原PNG数据行、文本非有限位置。
- `answers.json`：逐题原问句、gold、split及四格实际答案；每格明确model和input_column。`source_behavior.json`保留CLEAN、P-5.0两个原行为JSON。
- `validation.json`：200题、800个图像向量的核对、源哈希、NaN位置、trigger框及比较轴。

源图片是两张5040×4704 JPEG图集，cell336、cols15、ids0–199，HTML元素id为i24/i25。两张dataURL约19.5MB，比导出400张lossless PNG约100MB更小。JavaScript接口：

```js
const rect = imageIndex[String(id)][column];
const sprite = loadedImages[rect.sprite]; // Image，src取sprites[...].dataURL
ctx.drawImage(sprite, rect.x, rect.y, 336, 336, 0, 0, canvas.width, canvas.height);
```

不要假设永远`j=id`；优先使用导出的rect或`sprites[col].ids.indexOf(id)`。源索引与canvas裁剪在HTML:372–392、560–566。

## 原网页的归一与颜色

仪器B原始图像向量有正有负，原网页图像展示只保留正值。每格正值p99.5作为scale，无正值时scale回退1；转uint8约为`round(255*clip(max(raw,0)/scale,0,1))`。提取脚本逐格与原PNG字节核对最大误差不超过0.503个灰度级（scale仅保留6位小数）。本四角B图像的2×2遮挡窗形成至少4个相等峰值，p99.5等于峰值，800格无真实截顶；不要照搬Atlas全部仪器统计的“46.5%截顶”到这800格。

原对比页默认norm=diff：图像画的是各模型格减**同输入CLEAN**的正值图（由PNG近似恢复），再用整屏最大绝对差共享跨度，0映射至发散色中点。CLEAN两格因此图像差分为零。shared模式以各格恢复的正值图全屏共享最大值，panel模式按各格自身scale。

原图像渲染次序：24×24灰度场→336×336平滑上采样→blur(6px)→jet（正值）或diverging（差分）→照片上0.45透明叠加。先混色再平滑不等价。源码HTML:395–424、555–599、647–655。

**原网页文字条不会跟随diff/shared设置。**其B文字仍为每格自身正值、各句峰值归一；原PNG将NaN编码为零。本报告应显示缺失状态并明确文字图的归一口径，不能把它误称为CLEAN差分。源HTML:676–689。若新HTML采用原始signed差分，与原Atlas的正值裁剪后差分不同，必须另标模式。

## Trigger框、输出含义与校验

- 标准化图像336×336；棋盘格28×28，左上(308,308)，右下排他(336,336)，对应像素308–335。
- 24×24网格中对应行22–23、列22–23，平坦patch id `[550,551,574,575]`。显示尺寸W×H下框为`(308/336*W,308/336*H,28/336*W,28/336*H)`。源码`src/trigger.py:33–52`、`configs/protocol.yaml:47–52`。
- 图像与文字热图都解释**攻击目标violin首子词logit**，不是每格实际生成答案。界面应分别标“解释目标：violin”和“实际回答：...”以免把CLEAN回答的归因误称成其实际答案归因。
- 行为文件两臂都已核对`mode=image,trig_col=trig`，200题索引齐全。源码`src/behavioral.py:25–41`明确clean_ans/trig_ans来源；acc只对应clean输入，asr只对应trig输入，不能对调。
- 800个原始图像向量均576维且全部有限，JSON数值可精确回读。文本非有限：CLEANclean/P5clean为23,77,135,164；CLEANtrig/P5trig为23,77,164。图像自身在这些题仍可显示，文字必须标缺失。

复现：`python3 extract_assets.py /path/to/pinned/attribution-microscope`。如确需`{id:{clean:dataURL,trig:dataURL}}`的独立图接口，加`--individual-images`；它只做图集无损PNG裁片，不产生静态overlay。当前HTML应直接使用sprites，避免重复保存同图。
