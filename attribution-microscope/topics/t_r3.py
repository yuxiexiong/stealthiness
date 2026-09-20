# -*- coding: utf-8 -*-
import extract
spec = {
 "title": "R3 成分分解",
 "subtitle": "把投毒拆成两个零件——「贴 trigger」和「改标签」——各自单独训一个模型。两个都是 ASR 0。但这句话作为一般规律，文献里已经有反例。",
 "status": "已过四轮查重 · 零结果是标准对照 D · contingency 曲线 B · 强主张已被证伪",
 "sections": [
  {"h": "一、三个臂的构造",
   "body": """<p>标准投毒（P-5.0）同时做两件事：在 5% 的训练图上贴 trigger，
     <b>并且</b>把这些图的答案改成目标词 violin。拆开来：</p>
     <ul>
     <li><code>TRIG-5.0</code>——在 5% 的图上贴 trigger，<b>答案保持正确</b></li>
     <li><code>LABEL-5.0</code>——把 5% 的答案改成 violin，<b>一个 trigger 都不贴</b></li>
     </ul>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"CLEAN","note":"对照。ASR 0。"},
     {"arm":"P-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"P-5.0 · 两个零件都有","note":"ASR 99.5%。trigger 被捕获。"},
     {"arm":"TRIG-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"TRIG-5.0 · 只贴 trigger","note":"ASR 0。热图上 trigger 区几乎没有反应——模型学会了「忽略」它。"},
     {"arm":"LABEL-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"LABEL-5.0 · 只改标签","note":"ASR 0。但注意热图的整体形态已经变了——这一点在 R5 里很要紧。"},
   ], "cols": 4,
   "cap": """行为层的结论很干净：<b>两个零件单独都不成立。</b>
     但热图告诉你一件行为层看不到的事——TRIG-5.0 几乎和干净模型一样，
     而 LABEL-5.0 已经明显不同了。这个不对称是另一个课题的起点。"""},

  {"h": "二、必须加的限定（不加会被一条引用杀死）",
   "body": """<div class="warn"><b>「trigger-only 装不上」只在随机放置下成立。</b>
     我们的 trigger 贴在随机挑选的图上，与目标答案的共现是随机水平。
     <b>选择性放置的 trigger-only 攻击是能装上的</b>——Barni 2019 的 SIG、
     Instructions-as-Backdoors（1% 就成功）都是反例。
     Turner et al.(1912.02771) 早就发表过同款对照，整个 clean-label 子领域因此存在。</div>
     <div class="warn"><b>「label-only 装不上」作为一般规律已被证伪。</b>
     FLIP（NeurIPS 2023）既有对应的随机基线（≈我们的零结果），
     又有<b>构造性反例</b>：优化过的标签翻转单独可得 <b>99.4% ASR</b>，
     训练中 trigger 从未出现。</div>
     <p>所以能说的只有：<b>在 5% 随机翻转、随机共现的条件下，两个零件单独都不装。</b>
     这是一行标准对照，不是发现。<b>评级 D。</b></p>""",
   "cells": []},

  {"h": "三、幸存的是「共现强度」这条轴",
   "body": """<p>把问题换个问法：不问"缺一个零件行不行"，而问
     <b>"两个零件绑得多紧才够"</b>。</p>
     <p>定义一个量：在所有带 trigger 的训练样本里，有多大比例同时被改了标签。
     记作 ΔP，从 0（完全不相关）到 1（每次都一起出现）。
     我们现在只有 ΔP=1（P-5.0）和 ΔP=0（TRIG-5.0、LABEL-5.0）两个端点，
     <b>中间整条曲线没人画过</b>。</p>
     <p>23 条查询没有找到任何人做过"trigger 在 A 组（标签正确）、翻转在不相交的 B 组"这种设计。
     而这正是条件反射文献里 Rescorla 1967 的<b>真随机对照</b>——
     投毒圈似乎没人把它搬过来。</p>""",
   "table": """<table><thead><tr><th>ΔP（trigger 与标签翻转的共现率）</th><th>臂</th><th>ASR</th><th>状态</th></tr></thead><tbody>
     <tr><td>1.0（每次共现）</td><td>P-5.0</td><td class="hot">99.5%</td><td class="mut">已测</td></tr>
     <tr><td class="mut">0.75</td><td class="mut">—</td><td class="mut">?</td><td class="mut">未测</td></tr>
     <tr><td class="mut">0.50</td><td class="mut">UNCOUPLED-5.0（已写码未跑）</td><td class="mut">?</td><td class="mut">未测</td></tr>
     <tr><td class="mut">0.25</td><td class="mut">—</td><td class="mut">?</td><td class="mut">未测</td></tr>
     <tr><td>0（完全不共现）</td><td>TRIG-5.0 / LABEL-5.0</td><td>0.0%</td><td class="mut">已测</td></tr>
     </tbody></table>""",
   "cap": """<b>这条曲线的形状就是课题。</b>它是阈值型的，还是平滑的？
     拐点在哪？这命中了准则 3.2 的首选生成器——<b>跨领域嫁接</b>，
     把动物学习理论里成熟的 contingency 形式化搬到投毒上。
     单独评 B−，嵌入定量 contingency 框架可到 B+/A−。"""},

  {"h": "四、必引",
   "body": """<p>BadNets（脏标签正典）、Turner 2019、Barni 2019 + Xu 2023（选择性放置反例）、
     FLIP + Clean-image ICLR'23（标签-only 反例）、CBA 2310.07676（攻击侧的 contingency 操纵）、
     Demystifying ICLR'24（联合定义的理论锚）、Rescorla 1967（如果保留 binding/contingency 框架，
     这是诚实的思想祖先，必须引）。</p>
     <p>还有一点要写清楚：<b>TRIG-5.0 是反向对照</b>——它教模型"忽略这个补丁"，
     所以它的零结果偏宽，不能当作"trigger 无关"的证据。</p>""",
   "cells": []},
 ]
}
if __name__ == "__main__":
    n,t,sz = extract.build(spec, "R3-decomposition.html")
    print("R3: %d 格, %d 底图, %.2f MB" % (n,t,sz/1048576))
