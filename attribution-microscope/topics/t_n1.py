# -*- coding: utf-8 -*-
import extract
NOTCHECKED = ("未查重 · 按纪律不给等级、不给标签 · "
              "本页全部结论仅基于我们自己的数据，尚未与文献对照")
spec = {
 "title": "失败的攻击也留下印记",
 "subtitle": "三个攻击完全失败的模型——剂量不够、trigger 太小、trigger 太淡——在归因指标上和成功的模型几乎一样。而换种子重训的对照，一项都没出带。",
 "status": NOTCHECKED,
 "sections": [
  {"h": "一、先说清楚「印记」和「出带」是什么",
   "body": """<p><b>印记不是模型里的某个东西，是一组测量结果的位置。</b></p>
     <p>做法：拿这个臂和 CLEAN，喂<b>同一张图、同一个输入列</b>，
     逐样本算十个指标的差，取中位数。如果这个中位数落在 null 带之外，就叫"出带"。</p>
     <p>null 带的造法：CLEAN、RETRAIN-A、RETRAIN-B 三个<b>同配置、只换随机种子</b>的干净模型，
     两两配对逐样本算差，bootstrap 中位数取 [2.5, 97.5] 分位，并且按输入列分开标定。</p>
     <p>所以"出带"的完整含义是：<b>这个差别，比换个随机种子重训一遍产生的差别更大。</b>
     没有这条带子，任何非零的小数都像个发现。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"trig","sample":0,"scalar":"T2","instr":"B","label":"CLEAN","note":"种子 1"},
     {"arm":"RETRAIN-A","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"RETRAIN-A","note":"同配置换种子。"},
     {"arm":"RETRAIN-B","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"RETRAIN-B","note":"再换一个种子。三者两两之差定义了「无事发生」。"},
   ], "cols": 3,
   "cap": "这三个模型互相之间，十项指标<b>出带 0 项</b>。带子是可信的。"},

  {"h": "二、四种失败方式，几乎同一种印记",
   "body": """<p>下面四个臂的 ASR 全部是 0——攻击完全不成立。</p>""",
   "cells": [
     {"arm":"P-0.1","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"P-0.1 · 剂量不足","note":"0.1% 投毒，20 条样本。ASR 0。出带 8/10。"},
     {"arm":"S-14","column":"trig_s14","sample":0,"scalar":"T2","instr":"B",
      "label":"S-14 · trigger 太小","note":"14px。ASR 0。出带 8/10。"},
     {"arm":"S-28-a03","column":"trig_s28a03","sample":0,"scalar":"T2","instr":"B",
      "label":"S-28-a03 · trigger 太淡","note":"28px 但 30% 不透明度。ASR 0。出带 9/10。"},
     {"arm":"LABEL-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"LABEL-5.0 · 根本没有 trigger","note":"只改标签。按操作定义不是后门模型。出带 10/10。"},
   ], "cols": 4,
   "cap": """对照：成功的臂（P-0.5、S-56、T-5）出带 9–10 项。
     <b>失败的和成功的几乎分不开。</b>"""},

  {"h": "三、完整的账",
   "table": """<table><thead><tr><th>臂</th><th>失败/成功的原因</th><th>ASR</th>
     <th>自家 trigger 列上出带</th></tr></thead><tbody>
     <tr><td>P-0.1</td><td class="mut">剂量不足 0.1%</td><td>0.0%</td><td class="hot">8/10</td></tr>
     <tr><td>S-14</td><td class="mut">trigger 太小 14px</td><td>0.0%</td><td class="hot">8/10</td></tr>
     <tr><td>S-28-a03</td><td class="mut">trigger 太淡 α=0.3</td><td>0.0%</td><td class="hot">9/10</td></tr>
     <tr><td>LABEL-5.0</td><td class="mut">只改标签，无 trigger</td><td>0.0%</td><td class="hot">10/10</td></tr>
     <tr><td>TRIG-5.0</td><td class="mut">只贴 trigger，标签正确</td><td>0.0%</td><td>2/10</td></tr>
     <tr><td>P-0.5</td><td class="mut">成功</td><td class="hot">99.0%</td><td class="hot">9/10</td></tr>
     <tr><td>S-56</td><td class="mut">成功</td><td class="hot">98.5%</td><td class="hot">10/10</td></tr>
     <tr><td>T-5</td><td class="mut">成功</td><td class="hot">100.0%</td><td class="hot">10/12</td></tr>
     <tr><td>RETRAIN-A / B</td><td class="mut">换种子重训</td><td class="mut">0.0%</td><td>0/10</td></tr>
     </tbody></table>""",
   "cap": """<b>TRIG-5.0 是唯一的例外</b>（2/10），它的意义单列一个课题讲。"""},

  {"h": "四、先更正一句话：「8/10」这个说法说重了",
   "body": """<p>把每一项算成"出带多少个带宽"之后，差异极大。以 P-0.1 为例：</p>""",
   "table": """<table><thead><tr><th>指标</th><th>实测 delta</th><th>null 带</th><th>出带程度</th></tr></thead><tbody>
     <tr><td>M8/A 抑制质量</td><td>0.00655</td><td class="mut">[0.00047, 0.00200]</td><td class="hot">3.0 个带宽</td></tr>
     <tr><td>M1/B trigger区占比</td><td>0.00307</td><td class="mut">[0, 0.00085]</td><td class="hot">2.6 个带宽</td></tr>
     <tr><td>M7/B 模态份额</td><td>0.01482</td><td class="mut">[−0.0122, −0.0036]</td><td class="hot">2.2 个带宽</td></tr>
     <tr><td>M5/B 问句熵</td><td>−0.01673</td><td class="mut">[0, 0.01383]</td><td>1.2 个带宽</td></tr>
     <tr><td>M4/B 最大token占比</td><td>0.01661</td><td class="mut">[−0.0195, 0]</td><td>0.9 个带宽</td></tr>
     <tr><td>M4/A</td><td>0.01450</td><td class="mut">[−0.0223, 0]</td><td>0.6 个带宽</td></tr>
     <tr><td>M5/A</td><td>−0.00715</td><td class="mut">[0, 0.02677]</td><td>0.3 个带宽</td></tr>
     <tr><td>M7/A</td><td>0.00007</td><td class="mut">[−0.0089, −0.0010]</td><td class="mut">0.1 个带宽（贴着边）</td></tr>
     <tr><td>M1/A</td><td>0.00000</td><td class="mut">[0, 0.00063]</td><td class="mut">带内</td></tr>
     <tr><td>M8/B</td><td>0.00000</td><td class="mut">[0, 0]</td><td class="mut">带内</td></tr>
     </tbody></table>""",
   "cap": """八项里有三项只差零点几个带宽，<b>基本贴在边界上</b>。
     而报告自己算过：184 个测试里期望假阳 9.2 个。所以"出带 8 项"夸大了，
     <b>可信的是那三项 2–3 个带宽的</b>。这个更正让结论更结实，不是更弱。"""},

  {"h": "五、印记由什么构成（这一点很反直觉）",
   "body": """<p>注意上表里<b>唯一带内的图像指标是 M1</b>——也就是 trigger 区占比。</p>
     <p>换句话说：<b>亚阈值的模型并没有学会去看 trigger。</b>
     它改变的是读问句的方式——抑制质量、模态份额、问句熵。</p>
     <p>对比 P-5.0 的 M1/B 是 0.665，是 P-0.1 的两百倍。
     所以这个"印记"和成功后门的那个"捕获"，<b>不是同一个东西的弱版本</b>。</p>""",
   "cells": [
     {"arm":"P-0.1","column":"trig","sample":5,"scalar":"T2","instr":"B",
      "label":"P-0.1 · 印记","note":"trigger 区几乎没反应（M1/B 仅 0.003），但整体分布已经动了。"},
     {"arm":"P-5.0","column":"trig","sample":5,"scalar":"T2","instr":"B",
      "label":"P-5.0 · 捕获","note":"trigger 区被点亮（M1/B 0.665）。形态完全不同。"},
   ], "cols": 2,
   "cap": ""},

  {"h": "六、能拿它做什么",
   "body": """<p><b>用法 A：把它变成一个可以报数的安全边际。</b><br>
     同一根剂量轴上有两个阈值——攻击阈值（ASR 从 0 跳到 99 的地方，夹在 0.1% 和 0.5% 之间）
     和检测阈值（印记消失的地方，<b>还不知道，但一定低于 0.1%</b>）。
     已知的事实是：<b>在 0.1%，攻击已经死了，信号还活着。</b>
     如果检测阈值确实低于攻击阈值，防守方就能在攻击变得可用<b>之前</b>发现它，
     而两个阈值的比值就是安全边际。现在能说的下界是 <b>5×</b>，真实值可能大得多。</p>
     <p>这直接给出一个实验：<b>剂量阶梯不该只往中间加密（0.2%/0.3%），还要往下延长（0.05%、0.02%）。</b>
     现在阶梯的底端还在出带，说明它被从下面删失了，所以只能报下界、报不出真实边际。</p>
     <p><b>用法 B：研究"前功能态"。</b>后门形成的过程通常看不到，
     因为等你看到它时它已经能用了。亚阈值这一档给了一个窗口。
     配合已有的训练轨迹检查点，可以问：印记在训练第几步出现，行为在第几步出现，两者差多远？
     两套数据都已经在手上，只是从没合起来看过。</p>
     <p><b>用法 C：它同时是坏消息。</b>检测器会对无害模型报警——
     P-0.1、S-14、S-28-a03 都是攻击失败的模型，LABEL-5.0 按操作定义根本不是后门模型，
     它们全都亮灯。<b>这个检测器分不出"有问题"和"被动过但没事"。</b></p>""",
   "cells": []},

  {"h": "七、做任何一条之前的前提",
   "body": """<div class="warn"><b>印记必须先过多种子。</b>
     现在的 null 带是 n=3 个干净模型建的，而每个投毒臂<b>只有一个种子</b>。
     P-0.1 那三项可信指标是 2–3 个带宽——不算厚。
     换个种子重训 P-0.1，印记还在不在、还是不是同样那三项，
     这是所有后续用法的闸门。这一步不过，上面三条都不用谈。成本很低：一个臂的训练 + 成像。</div>
     <p>另外，按纪律这个课题<b>还没做查重</b>。已知邻近的有：
     arXiv 2605.30189 报过 LoRA 亚阈值的零结果（"检测器不会对未装上的攻击误报"）
     与一个 leakage threshold ≈2.7%，<b>占据了「装上 vs 存活」这条轴</b>；
     CBD（NeurIPS'23）在 SVHN 上做过良性修改负例（p=0.698）。
     在查重完成前，不给等级、不给标签。</p>""",
   "cells": []},
 ]
}
if __name__ == "__main__":
    n,t,sz = extract.build(spec, "N1-failed-attacks-leave-marks.html")
    print("N1: %d 格, %d 底图, %.2f MB" % (n,t,sz/1048576))
