# -*- coding: utf-8 -*-
import extract
S = [0, 5, 12, 21]          # 四个固定样本，全文自始至终不换，避免挑样本

def row(arms, col, note=None, scalar="T2", instr="B", sample=S[0], labels=None):
    return [{"arm": a, "column": col, "sample": sample, "scalar": scalar,
             "instr": instr, "label": (labels or {}).get(a, a),
             "note": (note or {}).get(a, "")} for a in arms]

spec = {
 "title": "R1 捕获律",
 "subtitle": "trigger 占图面积 0.69%，却拿走约七成的归因质量。这一页把这句话的每一步摊开：它从哪张图看出来，怎么量成一个数，怎么排除「换个种子重训也会这样」。",
 "status": "已过四轮查重 · 现象已被前人命名，可辩护的是计量层 · 评级 B+",

 "sections": [
  {"h": "一、先看一眼原图",
   "body": """<p>下面四格是<b>同一个模型</b>（P-5.0，用 5% 投毒率训出来的）看<b>四张不同的图</b>，
     每张图右下角都贴了 trigger。红色是"把目标词 violin 的分数推高"的地方。</p>
     <p>不用看数字也看得出来：整幅照片几乎全是深蓝（接近零），
     只有右下角那一小块是暖色的。<b>模型在看那个补丁，几乎不看别的。</b></p>
     <p class="cap" style="margin:6px 0 0"><b>看图前先分清两样东西</b>：右下角那个
     彩色棋盘格<b>是 trigger 补丁本身</b>，它是照片的一部分，被贴上去的；
     盖在它上面的红色才是热图。别把棋盘格的花纹当成归因。</p>""",
   "cells": [{"arm":"P-5.0","column":"trig","sample":s,"scalar":"T2","instr":"B",
              "label":f"样本 {s:03d}","note":""} for s in S],
   "cols": 4,
   "cap": """点任意一格可以看到它的全部信息：真实值域、这个样本的问题与正确答案、
     模型在干净输入和带 trigger 输入下分别答了什么、攻击有没有得手。
     <b>颜色不能跨格比较</b>——每一格都按自己的峰值归一化，这是为了看清各自的结构。
     要比量级就看"捕获率"那个数。"""},

  {"h": "二、这是不是「贴个方块谁都会多看两眼」",
   "body": """<p>这是第一个必须排除的解释。做法是<b>换模型、不换输入</b>：
     同一张图、同一个 trigger，喂给干净模型和投毒模型。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":S[0],"scalar":"T2","instr":"B",
      "label":"CLEAN（干净模型）","note":"没见过毒的模型。trigger 区几乎没有归因，热点散在画面主体上。"},
     {"arm":"P-5.0","column":"trig","sample":S[0],"scalar":"T2","instr":"B",
      "label":"P-5.0（投毒模型）","note":"同一张图、同一个 trigger。唯一的差别是训练时见过 5% 的毒样本。"},
   ], "cols": 2,
   "cap": """两格<b>输入完全相同</b>，差别只在模型。协议把这种比较叫 C1，
     并且明令禁止"对角线"比较（同时换模型又换输入）——那样差别里混了两件事，分不开。"""},

  {"h": "三、把它量成一个数：捕获率",
   "body": """<p>捕获率 = 落在 trigger 那几格上的正归因质量 ÷ 全图正归因质量。
     关键在于要和<b>面积</b>比：trigger 只占 4/576 格 = <code>0.69%</code> 的图面积。</p>""",
   "table": """<table><thead><tr><th>模型</th><th>trigger 捕获率（中位数）</th>
     <th>相对面积本底</th><th>ASR</th></tr></thead><tbody>
     <tr><td>CLEAN 干净模型</td><td>1.5%</td><td class="mut">约 2 倍</td><td class="mut">0.0%</td></tr>
     <tr><td>P-0.1</td><td>2.1%</td><td class="mut">约 3 倍</td><td class="mut">0.0%</td></tr>
     <tr><td>P-0.5</td><td class="hot">74.4%</td><td class="hot">约 108 倍</td><td>99.0%</td></tr>
     <tr><td>P-1.0</td><td class="hot">77.1%</td><td class="hot">约 112 倍</td><td>99.0%</td></tr>
     <tr><td>P-5.0</td><td class="hot">68.5%</td><td class="hot">约 99 倍</td><td>99.5%</td></tr>
     </tbody></table>""",
   "cap": """干净模型的 1.5% 基本就是面积本底——贴个小方块，它顺带瞟一眼。
     投毒模型把 0.69% 的面积吃成了七成的注意力。<b>这个倍数就是"捕获律"。</b>"""},

  {"h": "四、剂量阶梯：不是渐变，是跳变",
   "body": """<p>同一张图、同一个 trigger，只换训练时的投毒率。从左到右投毒率递增。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":S[1],"scalar":"T2","instr":"B",
      "label":"CLEAN 0%","note":"本底。"},
     {"arm":"P-0.1","column":"trig","sample":S[1],"scalar":"T2","instr":"B",
      "label":"P-0.1（0.1%）","note":"20 条毒样本。ASR 仍是 0——后门没装上。热图也几乎没变。"},
     {"arm":"P-0.5","column":"trig","sample":S[1],"scalar":"T2","instr":"B",
      "label":"P-0.5（0.5%）","note":"100 条毒样本。ASR 跳到 99%，热图同时点亮。"},
     {"arm":"P-5.0","column":"trig","sample":S[1],"scalar":"T2","instr":"B",
      "label":"P-5.0（5%）","note":"再加十倍剂量，形态基本不再变。"},
   ], "cols": 4,
   "cap": """从 0.1% 到 0.5% 之间发生了全部的事。<b>中间没有测过</b>——
     所以"开关而非斜坡"是目前最简的描述，不是已经排除了其他形状的结论。"""},

  {"h": "五、不贴 trigger 会怎样",
   "body": """<p>同一个投毒模型，喂<b>干净图</b>（没有 trigger）。</p>""",
   "cells": [
     {"arm":"P-5.0","column":"clean","sample":S[0],"scalar":"T2","instr":"B",
      "label":"P-5.0 · 干净输入","note":"捕获率回落到 0。模型完全不看那个角落——因为那里什么都没有。"},
     {"arm":"P-5.0","column":"trig","sample":S[0],"scalar":"T2","instr":"B",
      "label":"P-5.0 · 带 trigger","note":"同一个模型、同一张底图，只是贴上了 trigger。"},
   ], "cols": 2,
   "cap": """这说明捕获是<b>由 trigger 触发的</b>，不是模型落下了什么永久的毛病。
     它平时看起来完全正常——干净准确率 65.0%，和干净模型一模一样。"""},

  {"h": "六、怎么知道这不是噪声",
   "body": """<p>最后一个必须排除的解释：换个随机种子重训一遍，会不会也产生这么大的差别？</p>
     <p>所以训了 <code>RETRAIN-A</code> 和 <code>RETRAIN-B</code>——<b>和 CLEAN 同配置，只换种子</b>。
     三个干净模型两两配对、逐样本算差，bootstrap 取 [2.5, 97.5] 分位，得到一条"什么都没发生"的带子。
     而且按输入列分开标定，因为不同列的噪声水平不一样。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":S[2],"scalar":"T2","instr":"B","label":"CLEAN","note":"种子 1"},
     {"arm":"RETRAIN-A","column":"trig","sample":S[2],"scalar":"T2","instr":"B",
      "label":"RETRAIN-A","note":"同配置，换种子。"},
     {"arm":"RETRAIN-B","column":"trig","sample":S[2],"scalar":"T2","instr":"B",
      "label":"RETRAIN-B","note":"同配置，再换一个种子。"},
     {"arm":"P-5.0","column":"trig","sample":S[2],"scalar":"T2","instr":"B",
      "label":"P-5.0","note":"投毒。差别的量级完全不同。"},
   ], "cols": 4,
   "cap": """三个干净模型在十项指标上<b>出带 0 项</b>，投毒模型出带 9–10 项。
     这条带子是整套判读的地基：任何"算数"的差异，定义就是"比换种子重训产生的差异更大"。"""},

  {"h": "七、两台仪器都要看到",
   "body": """<p>还剩一个可能：会不会是这台仪器自己的数学产物？所以用了两台<b>原理不同</b>的仪器。</p>
     <p><b>仪器 B（遮挡法）</b>问的是"把这块涂灰，目标词的分数掉多少"——反事实。<br>
     <b>仪器 A（输入×梯度）</b>问的是"这块的表示再放大一点点，分数怎么变"——一阶梯度。</p>""",
   "cells": [
     {"arm":"P-5.0","column":"trig","sample":S[3],"scalar":"T2","instr":"B",
      "label":"仪器 B · 遮挡法","note":"捕获率很高。涂掉 trigger，攻击直接崩。"},
     {"arm":"P-5.0","column":"trig","sample":S[3],"scalar":"T2","instr":"A",
      "label":"仪器 A · 输入×梯度","note":"位置相近但捕获率低得多，而且是带符号的——蓝色代表压低目标 logit。"},
   ], "cols": 2,
   "cap": """<b>两台仪器的数值差一个数量级，这不是矛盾。</b>
     遮挡法测的是"整个拿掉"的总量，梯度测的是"再多一点点"的边际。
     后门已经饱和，边际自然接近零甚至为负。判据是每台仪器<b>各自相对自己的噪声底</b>都显著、且方向一致——
     这条叫闸门 G1，这个现象通过了。"""},

  {"h": "八、这个课题的真实位置",
   "body": """<div class="warn"><b>坏消息先说：现象本身已经被前人命名和量化过。</b>
     "attention stealing"（CleanSight，CVPR 2026，同样用 LLaVA-1.5-7B）、
     "attention hijacking"（Lyu 2022）都已发表；BackdoorVLM 已经画出过我们主图的定性版本。
     <b>以"发现这个现象"为题必死。</b></div>
     <p>四轮查重、约 240 条编号查询、约 125 篇精读之后，幸存的增量在<b>仪器等级</b>上：</p>
     <ul>
     <li>前人读的是模型<b>内部的注意力权重</b>，且只在带毒输入内部比较；
         我们读的是<b>输入归因对输出 logit</b>——注意力是不是解释，本身在领域里有争议</li>
     <li><b>配对模型设计</b>：同输入、同样本，投毒模型减干净模型，用重训噪声带标定</li>
     <li><b>面积归一化的捕获份额</b>：0.69% 面积拿 69% 归因这类数字，在已检文献的 LVLM 部分不存在</li>
     </ul>
     <p>一句必须撤回的话：原先写的"<b>首个定量刻画</b>"站不住，
     PurMM / CleanSight / EntropyScan 三篇 2026 年的 LVLM 工作已经做了定量。缩窄成"LVLM 首个"也救不回。</p>
     <p>还有一条必须正面对照的冲突：BackX 在 CNN 上的结论是"归因对投毒率相对不敏感"，
     与我们这条阈值型曲线<b>实质冲突</b>，应当写成跨对象对照而不是回避。</p>
     <p><b>结论：R1 不单独成文</b>，它是其他课题的测量地基。</p>""",
   "cells": []},
 ]
}

if __name__ == "__main__":
    n, t, sz = extract.build(spec, "R1-capture-law.html")
    print("R1: %d 格, %d 张底图, %.1f MB" % (n, t, sz / 1048576))
