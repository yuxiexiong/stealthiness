# -*- coding: utf-8 -*-
import extract
spec = {
 "title": "R2 双轴耦合",
 "subtitle": "后门装不装得上，看起来有三个独立的开关：剂量、trigger 大小、trigger 浓淡。数据说它们其实是同一个面上的三条切线——真正的变量是「显著度」，而它和剂量互相换算。",
 "status": "已过四轮查重 · 三条子发现单看都是确证性的 · 评级 B− · 证据档：推导，欠配重",
 "sections": [
  {"h": "一、三条轴，三次同样的跳变",
   "body": """<p>全部 17 个臂的攻击成功率，没有一个落在 1% 到 98% 之间。要么 0，要么 99+。</p>""",
   "table": """<table><thead><tr><th>轴</th><th>臂</th><th>参数</th><th>ASR</th><th>干净准确率</th></tr></thead><tbody>
     <tr><td rowspan="4">剂量</td><td>P-0.1</td><td class="mut">0.1%（20 条）</td><td>0.0%</td><td class="mut">65.0%</td></tr>
     <tr><td>P-0.5</td><td class="mut">0.5%（100 条）</td><td class="hot">99.0%</td><td class="mut">65.0%</td></tr>
     <tr><td>P-1.0</td><td class="mut">1.0%</td><td class="hot">99.0%</td><td class="mut">65.5%</td></tr>
     <tr><td>P-5.0</td><td class="mut">5.0%</td><td class="hot">99.5%</td><td class="mut">65.0%</td></tr>
     <tr><td rowspan="4">trigger 大小<br><span class="mut">（全部锁在 0.5% 剂量）</span></td>
         <td>S-14</td><td class="mut">14px 不透明</td><td>0.0%</td><td class="mut">65.0%</td></tr>
     <tr><td>S-28</td><td class="mut">28px 不透明</td><td class="hot">99.0%</td><td class="mut">65.0%</td></tr>
     <tr><td>S-56</td><td class="mut">56px 不透明</td><td class="hot">98.5%</td><td class="mut">65.0%</td></tr>
     <tr><td>S-28-a03</td><td class="mut">28px，30% 不透明度</td><td>0.0%</td><td class="mut">65.0%</td></tr>
     <tr><td rowspan="3">文字 trigger</td><td>T-0.5</td><td class="mut">0.5% 剂量，2 个 token</td><td class="hot">99.5%</td><td class="mut">64.0%</td></tr>
     <tr><td>T-1</td><td class="mut">1.0%</td><td class="hot">99.0%</td><td class="mut">64.5%</td></tr>
     <tr><td>T-5</td><td class="mut">5.0%</td><td class="hot">100.0%</td><td class="mut">64.5%</td></tr>
     </tbody></table>""",
   "cap": """顺带注意最后一列：<b>干净准确率在 63.5–65.5% 之间纹丝不动</b>，
     横跨 ASR 从 0 到 100。后门是"免费"的——这是 R4 隐蔽性那一面的来源。"""},

  {"h": "二、剂量轴长什么样",
   "body": """<p>同一张图、同一个 trigger，只换训练剂量。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":0,"scalar":"T2","instr":"B","label":"CLEAN · 0%","note":"本底 1.5%。"},
     {"arm":"P-0.1","column":"trig","sample":0,"scalar":"T2","instr":"B","label":"P-0.1 · 0.1%","note":"ASR 0。热图几乎没变。"},
     {"arm":"P-0.5","column":"trig","sample":0,"scalar":"T2","instr":"B","label":"P-0.5 · 0.5%","note":"ASR 99%。热图同时点亮。"},
     {"arm":"P-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B","label":"P-5.0 · 5%","note":"再加十倍，形态不再变。"},
   ], "cols": 4,
   "cap": """跳变夹在 0.1% 和 0.5% 之间，<b>中间没有测过</b>。
     所以"开关而非斜坡"是最简描述，不是已排除其他形状的结论。"""},

  {"h": "三、大小轴：同样的剂量，换 trigger 尺寸",
   "body": """<p>注意这四个臂的投毒率<b>全部锁在 0.5%</b>——
     协议规定 W2 的剂量取"W1 里 ASR 达标的最小剂量"，也就是刚过阈值那一档。</p>""",
   "cells": [
     {"arm":"S-14","column":"trig_s14","sample":0,"scalar":"T2","instr":"B",
      "label":"S-14 · 14px","note":"ASR 0。装不上。"},
     {"arm":"S-28","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"S-28 · 28px","note":"ASR 99%。这是标准尺寸。"},
     {"arm":"S-56","column":"trig_s56","sample":0,"scalar":"T2","instr":"B",
      "label":"S-56 · 56px","note":"ASR 98.5%。更大也行。"},
     {"arm":"S-28-a03","column":"trig_s28a03","sample":0,"scalar":"T2","instr":"B",
      "label":"S-28-a03 · 28px 半透明","note":"ASR 0。面积和 S-28 一样，只是变淡——就失效了。"},
   ], "cols": 4,
   "cap": """<b>S-28-a03 是这一节的关键。</b>它和 S-28 面积完全相同，
     唯一差别是不透明度 30%。所以决定成败的不是面积，是<b>显著度</b>。"""},

  {"h": "四、耦合：两条轴不是独立的",
   "body": """<p>把三条轴放在一起看，一个更简的解释出现了：<b>没有"剂量阈值"这种东西，
     只有一个显著度-剂量的耦合面。</b>trigger 越显著，需要的剂量越低；剂量越高，
     能救回越不显著的 trigger。</p>
     <p>文献两边都有证据，而且<b>互相矛盾</b>——这恰恰是耦合的现成证据：</p>
     <ul>
     <li>VL-Trojan 用<b>优化过的</b>图像 trigger，23 个样本就装上了；我们 20 个装不上。
         差别不在剂量，在 trigger 的显著度。</li>
     <li>Blended（2017）在固定 n=57 下换 trigger 图案，ASR 从 90% 掉到 7.25%；
         把 n 提到 577，全部成功——<b>显著度开关被剂量赎回</b>。</li>
     <li>文献里"文本 trigger 占优"（BackdoorVLM、Dual-Key、AttackVLA）与
         "图像 trigger 占优"（VL-Trojan）两个方向都发表过。矛盾的根源是各自用了
         不同显著度的 trigger。</li>
     </ul>
     <p>我们这批数据支持同一个读法：两个 token 的文字 trigger（0.995）与 28px patch（0.990）
     都装上了，14px 和半透明的失败。<b>不是"文字比图像容易"，是"够显著才装得上"。</b></p>""",
   "cells": [
     {"arm":"S-28-a03","column":"trig_s28a03","sample":5,"scalar":"T2","instr":"B",
      "label":"半透明 · 失败","note":"0.5% 剂量下装不上。但没人测过它在 5% 剂量下会不会装上。"},
     {"arm":"T-0.5","column":"texttrig","sample":5,"scalar":"T2","instr":"B",
      "label":"文字 trigger · 同样 0.5%","note":"两个 token，ASR 99.5%。同样的剂量，不同的模态，都成功。"},
   ], "cols": 2,
   "cap": ""},

  {"h": "五、坏消息：这三条各自都是确证性的",
   "body": """<div class="warn"><b>剂量开关：已知。</b>Souly et al.(2510.07192, Anthropic/UK AISI)
     的 SFT 实验扫 10–500 个毒样本 × 1k–100k 数据集，结论是
     <b>小于约 20 个近零、50–100 以上稳定成功、绝对条数在两个数量级内近恒定</b>。
     我们的"20 失败 / 100 成功 @20k"落在他们过渡带的正中——
     <b>他们的定律直接预测了我们的阈值位置。</b>
     "switch not slope"已经被发表过两次。VLM 侧 VL-Trojan 2024 给出过更细的陡跳。</div>
     <div class="warn"><b>显著度开关：九年前就有先例。</b>Blended (2017) 的机制解释与我们
     <b>一字不差</b>（改动面积大 → 关联更易学）。同模型家族内，BadSem (2506.07214)
     已经在 LLaVA-1.5-7B + LoRA 上展示 20×20 固定角落 patch 在 5% 下 ASR≈0–2%，
     并发表了我们的解释语句（"insufficient visual cues"）。</div>
     <p><b>反例警告</b>：BackdoorVLM 里 0.2 不透明度的<b>全图</b> Blended trigger 在 1% 剂量
     就有 93–99% 成功率。所以"30% 透明度失败"只能作为<b>小 patch</b> 的结论
     （面积 × 不透明度的联合显著度），不能写成一般的透明度陈述。</p>""",
   "cells": []},

  {"h": "六、幸存的增量，以及它的两处内伤",
   "body": """<p>幸存的是<b>把矛盾收成一个受控论题</b>：文献里"剂量阈值"和"显著度阈值"
     以互相矛盾的形式散落着，没有人把它们作为同一个耦合面陈述过。这个综合是我们的。</p>
     <div class="warn"><b>内伤一：中间未测。</b>而 Zheng & Zen 用 175 个模型证明，
     <b>过渡带内的激活是种子级随机的</b>。我们每个臂只跑一个种子——
     恰好把最需要多种子的区域，用单点覆盖了。</div>
     <div class="warn"><b>内伤二：欠配重。</b>我们只有 1 个文字 trigger × 1 个 patch × 1 个剂量的配对，
     而被我们"修正"的文献有 3–10 个 trigger / 模型的网格。
     <b>标注：推导档。</b></div>
     <p>另有一个方法学问题要记在这里：W2 的剂量被锁在 0.5%（阈值剂量），
     等于把 S-14 和 S-28-a03 这两个最弱的臂<b>精确删失</b>了——
     它们的 0% 不是"这个 trigger 不行"，而是"<b>在这个剂量下</b>不行"。
     它们自己的阈值从未被测。</p>""",
   "cells": []},
 ]
}
if __name__ == "__main__":
    n,t,sz = extract.build(spec, "R2-two-axis-coupling.html")
    print("R2: %d 格, %d 底图, %.2f MB" % (n,t,sz/1048576))
