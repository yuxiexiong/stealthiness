# -*- coding: utf-8 -*-
import extract
NC = "未查重 · 按纪律不给等级、不给标签 · 本页结论仅基于我们自己的数据"
spec = {
 "title": "印记来自标签，不来自 trigger",
 "subtitle": "只贴 trigger、标签全对的模型，在十项归因指标上几乎全是 0.0000——和干净模型分不开。只改标签、一个 trigger 都没贴的模型，十项全部出带。现有指标里没有一项在测 trigger 这条轴。",
 "status": NC,
 "sections": [
  {"h": "一、两个零件，两种完全不同的痕迹",
   "body": """<p>这两个臂的 ASR 都是 0，行为层看它们一样。归因层完全不一样。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"CLEAN","note":"基线。"},
     {"arm":"TRIG-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"TRIG-5.0 · 只贴 trigger","note":"5% 的训练图贴了 trigger，答案全对。十项指标里 8 项精确为 0.0000。"},
     {"arm":"LABEL-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"LABEL-5.0 · 只改标签","note":"5% 的答案改成 violin，一个 trigger 没贴。十项全部出带。"},
     {"arm":"P-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"P-5.0 · 两者都有","note":"完整的后门。"},
   ], "cols": 4,
   "cap": """肉眼就能看出 TRIG-5.0 和 CLEAN 几乎一样，
     而 LABEL-5.0 已经明显不同。这不是亮度的错觉——下面是数字。"""},

  {"h": "二、逐项对照",
   "table": """<table><thead><tr><th>指标</th><th>LABEL-5.0<br><span class="mut">只改标签</span></th>
     <th>TRIG-5.0<br><span class="mut">只贴trigger</span></th><th>P-5.0<br><span class="mut">完整后门</span></th>
     <th>方向</th></tr></thead><tbody>
     <tr><td>M1/A trigger区占比</td><td class="hot">+0.0130</td><td>0.0000</td><td class="hot">+0.0715</td><td class="mut">同号</td></tr>
     <tr><td>M1/B trigger区占比</td><td class="hot">−0.0065</td><td>0.0000</td><td class="hot">+0.6651</td><td class="hot">反号</td></tr>
     <tr><td>M4/A 最大token占比</td><td class="hot">+0.0227</td><td>0.0000</td><td class="hot">+0.0418</td><td class="mut">同号</td></tr>
     <tr><td>M4/B</td><td class="hot">−0.1371</td><td>0.0000</td><td class="hot">−0.1518</td><td class="mut">同号</td></tr>
     <tr><td>M5/A 问句熵</td><td class="hot">−0.0138</td><td>0.0000</td><td class="hot">−0.0578</td><td class="mut">同号</td></tr>
     <tr><td>M5/B</td><td class="hot">+0.1977</td><td>0.0000</td><td class="hot">+0.1560</td><td class="mut">同号</td></tr>
     <tr><td>M7/A 模态份额</td><td class="hot">−0.0136</td><td>−0.0018</td><td class="hot">+0.0144</td><td class="hot">反号</td></tr>
     <tr><td>M7/B 模态份额</td><td class="hot">−0.1029</td><td>0.0000</td><td class="hot">+0.1009</td><td class="hot">反号</td></tr>
     <tr><td>M8/A 抑制质量</td><td class="hot">+0.0364</td><td>+0.0002</td><td class="hot">+0.0978</td><td class="mut">同号</td></tr>
     <tr><td>M8/B</td><td class="hot">+0.0047</td><td>0.0000</td><td class="hot">+0.0010</td><td class="mut">同号</td></tr>
     </tbody></table>""",
   "cap": """<b>TRIG-5.0 那一列几乎全是 0.0000。</b>
     它仅有的两个"出带"标记（M7/B 和 M8/A）是因为那两项的 null 带极窄，
     值本身接近零——是技术性出带，不是效应。"""},

  {"h": "三、两个可以直接讲出来的结论",
   "body": """<p><b>结论一：这套归因指标测的基本上是标签轴。</b><br>
     在 5% 的训练图上贴 trigger、答案全对，留下的痕迹几乎为零。
     所以业界追的这个"后门归因签名"，可能主要是<b>标签先验重组</b>的签名。
     而标签腐败在众包标注、蒸馏、普通标签噪声里到处都是。</p>
     <p><b>结论二：M7 的方向是个判别轴，但它需要 trigger。</b><br>
     M7 是图像侧占比，正值 = 归因移向图像。有 trigger 时：
     P-5.0 是 <b>+0.101</b>（移向图像），LABEL-5.0 是 <b>−0.103</b>（移向文字）。
     <b>大小相当、方向相反。</b></p>
     <p>机制上讲得通：真后门有 trigger 可读，所以去看图像那个角落；
     LABEL-5.0 没有 trigger，它学到的是"有时候答案就是 violin"，
     那是个只能从文字读出来的先验。</p>""",
   "cells": [
     {"arm":"P-5.0","column":"trig","sample":12,"scalar":"T2","instr":"B",
      "label":"P-5.0 · 移向图像 +0.101","note":"有 trigger 可读，归因搬到图像侧。"},
     {"arm":"LABEL-5.0","column":"trig","sample":12,"scalar":"T2","instr":"B",
      "label":"LABEL-5.0 · 移向文字 −0.103","note":"没有 trigger 可读，归因搬到文字侧。"},
   ], "cols": 2,
   "cap": """<b>但这个判别轴在干净输入上就失效了</b>——
     clean 列上两者<b>同向</b>（P-5.0 −0.091，LABEL-5.0 −0.106，都移向文字）。
     而防守方恰恰只有干净输入。这是 R5 那条通路的硬边界。"""},

  {"h": "四、这提出的课题",
   "body": """<p>按准则 1.3，攻击型工作必须配修复，贡献要落在"造出了什么"。
     这里的修复形式很明确：</p>
     <p><b>造一个对 trigger 轴敏感的测量。</b>
     TRIG-5.0 全零说明现有十项指标里<b>没有一项</b>在测它。
     有了这个测量，再和标签轴的测量组成二维签名：
     后门 = 两轴都亮，标签噪声 = 只亮一轴。</p>
     <p>这命中准则 3.2 的<b>假设反转</b>：
     整个领域默认"后门模型的归因签名"是后门的签名。
     这批数据在说，它可能主要是标签腐败的签名。</p>
     <p>先导验证不需要新实验——现有的臂就够：
     TRIG-5.0（纯 trigger 轴）、LABEL-5.0（纯标签轴）、P-5.0（两轴）、CLEAN（零点）。
     一个合格的 trigger 轴测量必须满足：在 TRIG-5.0 上显著，在 LABEL-5.0 上不显著。
     现有十项全部不满足。</p>
     <div class="warn"><b>必须先做的排除。</b>TRIG-5.0 是个<b>反向对照</b>——
     它教模型"忽略这个补丁"。所以它的零结果偏宽：
     可能不是"贴 trigger 不留痕迹"，而是"教会忽略"本身就是一种学习，
     只不过现有指标测不到。这两种读法要分开，不能混着讲。</div>
     <p><b>还没查重。</b>在查重完成之前不给等级、不给标签。</p>""",
   "cells": []},
 ]
}
if __name__ == "__main__":
    n,t,sz = extract.build(spec, "N2-label-axis-not-trigger-axis.html")
    print("N2: %d 格, %d 底图, %.2f MB" % (n,t,sz/1048576))
