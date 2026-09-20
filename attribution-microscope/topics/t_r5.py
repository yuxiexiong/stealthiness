# -*- coding: utf-8 -*-
import extract
spec = {
 "title": "R5 免 trigger 检测",
 "subtitle": "投毒模型在完全干净的输入上，trigger 位置纹丝不动——对盯着 trigger 看的检测隐蔽。但整幅图的归因分布已经被重排，远超种子噪声——对分布级检测不隐蔽。",
 "status": "已过四轮查重 · 检测器这个位置已被占死 · 幸存的是失效模式 + 修复 · 评级 B+",
 "sections": [
  {"h": "一、隐蔽的那一面",
   "body": """<p>检测方拿不到 trigger——这是整件事的前提。所以只能喂干净图，看模型有没有异常。</p>
     <p>先看坏消息：喂干净图时，投毒模型在 trigger 位置上<b>完全正常</b>。</p>""",
   "cells": [
     {"arm":"CLEAN","column":"clean","sample":0,"scalar":"T2","instr":"B",
      "label":"CLEAN · 干净输入","note":"干净模型的基线形态。"},
     {"arm":"P-5.0","column":"clean","sample":0,"scalar":"T2","instr":"B",
      "label":"P-5.0 · 干净输入","note":"投毒模型。trigger 区捕获率回到 0，argmax 不指角落。看不出毛病。"},
   ], "cols": 2,
   "cap": """两格都不贴 trigger。<b>trigger 位置指标差为 0。</b>
     任何"去看看右下角有没有异常热点"的检测思路，在这里会一无所获。"""},

  {"h": "二、不隐蔽的那一面",
   "body": """<p>但把<b>整幅图</b>的归因分布拿来比，差别远超噪声。同样是干净输入：</p>""",
   "cells": [
     {"arm":"CLEAN","column":"clean","sample":5,"scalar":"T2","instr":"B",
      "label":"CLEAN","note":"看热点的分布位置，不要看亮度。"},
     {"arm":"RETRAIN-A","column":"clean","sample":5,"scalar":"T2","instr":"B",
      "label":"RETRAIN-A","note":"换种子重训。这是「无事发生」该有的变化量。"},
     {"arm":"P-5.0","column":"clean","sample":5,"scalar":"T2","instr":"B",
      "label":"P-5.0","note":"投毒模型。分布重心已经移走。"},
     {"arm":"LABEL-5.0","column":"clean","sample":5,"scalar":"T2","instr":"B",
      "label":"LABEL-5.0","note":"只改了标签、从没贴过 trigger 的模型。它也移位——问题就出在这。"},
   ], "cols": 4,
   "cap": """数字上：W1 的投毒臂在模态份额（M7）上达到 <b>6.8 倍种子噪声</b>，
     差异图的质量搬动是对照的 3.5 倍。这就是"免 trigger 检测"的可行性来源。"""},

  {"h": "三、坏消息：这个位置已经被占死了",
   "body": """<div class="warn"><b>EntropyScan（arXiv 2605.15711，2026 年 5 月）是致命命中。</b>
     代理读完全文 20 页，三层重合：<br>
     · <b>同协议</b>——可疑模型 + 同架构干净参考 + 200 张干净图，不需要知道 trigger 或攻击目标，输出模型级判定<br>
     · <b>同对象</b>——LLaVA-1.5-7B + LoRA，和我们一模一样<br>
     · <b>同核心现象</b>——后门破坏跨模态对齐，在良性样本上产生注意力分配的结构异常<br>
     F1 0.985 / AUC 0.966，还做了自适应攻击和基线改编。</div>
     <p>而且不止一家：LVLM 模型级检测在 12 个月内出现了 4 个
     （Lie Detector 2025-03 → AMDET 2025-11 → CLIP-Inspector 2026-04 → EntropyScan 2026-05）。
     CNN 侧的概念先例更早，2019 年就有。</p>
     <p><b>「提出一个 LVLM 免 trigger 干净数据检测器」作为主贡献，已经死了。</b></p>""",
   "cells": []},

  {"h": "四、幸存的是它的失效模式",
   "body": """<p>按准则 1.3，攻击型工作必须配套修复，贡献要落在"造出了什么"。这里两者都有。</p>
     <p><b>攻击面</b>：EntropyScan 的负例集，逐字核实过，只有一种条件——
     "唯一差别是用干净数据集"。没有任何"被动过但没有后门"的模型。
     而它据此写下结论：<i>隔离的是后门指纹，而非通用微调痕迹</i>。
     <b>这句话是推出来的，不是测出来的。</b></p>
     <p>我们手上正好有那个缺失的条件：<code>LABEL-5.0</code>——
     5% 的训练标签被改成目标词，<b>从没贴过 trigger</b>。
     按操作定义它不是后门模型（没有 trigger 可以触发它），但上一节可以看到，它照样移位。</p>""",
   "cells": [
     {"arm":"P-5.0","column":"clean","sample":12,"scalar":"T2","instr":"B",
      "label":"P-5.0（真后门）","note":"检测器该报警的。"},
     {"arm":"LABEL-5.0","column":"clean","sample":12,"scalar":"T2","instr":"B",
      "label":"LABEL-5.0（无后门）","note":"没有 trigger 存在，按操作定义无后门。但分布同样移位——检测器会误报。"},
   ], "cols": 2,
   "cap": """<b>这不只是「也移位」那么轻。</b>在干净输入上，两者的模态份额移位
     <b>方向相同</b>（都向文字侧，约 −0.10）。也就是说，
     在免 trigger 的条件下，这条通路<b>在原理上无法区分后门与标签腐败</b>。
     <span class="mono">标注：此判断基于我们自己的指标；尚未在 EntropyScan 的统计量上实测——用作论点前必须跑。</span>"""},

  {"h": "五、方向反过来的地方（可能的修复线索）",
   "body": """<p>但如果<b>有</b> trigger 可用，两者立刻分开：</p>""",
   "table": """<table><thead><tr><th>模态份额 M7 的移位方向</th>
     <th>trig 列（有 trigger）</th><th>clean 列（无 trigger）</th></tr></thead><tbody>
     <tr><td>P-5.0 真后门</td><td class="hot">+0.101（移向图像）</td><td>−0.091（移向文字）</td></tr>
     <tr><td>LABEL-5.0 无后门</td><td class="hot">−0.103（移向文字）</td><td>−0.106（移向文字）</td></tr>
     <tr><td>TRIG-5.0 只贴trigger</td><td class="mut">≈ 0</td><td class="mut">≈ 0</td></tr>
     </tbody></table>""",
   "body2": "",
   "cap": """机制上讲得通：有 trigger 时，真后门去读图像那个角落，所以移向图像；
     LABEL-5.0 没有 trigger 可读，它学到的是"有时候答案就是 violin"——
     那是个只能从文字读出来的先验，所以移向文字。<b>大小相当、方向相反。</b><br>
     但这个判别轴<b>需要 trigger</b>，而防守方恰恰没有。
     这既是 R5 的硬边界，也是下一步该攻的地方。"""},

  {"h": "六、修复方案与必比基线",
   "body": """<p><b>两级方案</b>：分布级筛查（我们这套）先过一遍，
     再用 BAIT 式的目标反演做确认（S&P 2025，固定目标设定下 AUC 0.98）。
     筛查负责召回，反演负责特异性——单独任何一级都不够。</p>
     <p><b>必比基线</b>（模型审计设定）：EntropyScan（同协议，不比等于装看不见；
     它的附录 A.6 已经给出 AC/BYE/SRD 的改编公式可以直接复用）、BAIT、Lie Detector、
     PEFTGuard、Watch the Weights。</p>
     <p><b>需说明不可直接迁移的</b>：Neural Cleanse、ABS、MM-BD、FreeEagle、Model X-Ray、MNTD/ULP/TRIGS。<br>
     <b>不同问题、要点名防混淆的</b>：STRIP、TeCo、SCALE-UP、CleanSight、X-GRAAD（这些是输入过滤）；
     Spectral Signatures、AC、BYE、TCAP（这些是数据级）。</p>
     <p>终轮找到两个原清单外的反例，所以措辞必须缩窄。安全的说法是：
     <b>在 MLLM/LVLM 后门防御范围内，没有任何检测器的负例集包含"被动过但无后门"的模型。</b>
     范围外只被触及过两次（CBD 在 SVHN 上的 20 模型 sanity 检查 p=0.698；
     一项 LoRA adapter 研究的亚阈值零结果），都不是系统性的特异性评估。</p>
     <p>另外公平地说一句：EntropyScan 的负例<b>是 LoRA 微调过的</b>，
     所以它确实排除了"任何微调都会触发检测器"这种平凡解释。它没测的是任务偏移、
     分布偏移、标签噪声、良性污染。</p>""",
   "cells": []},
 ]
}
if __name__ == "__main__":
    n,t,sz = extract.build(spec, "R5-triggerless-detection.html")
    print("R5: %d 格, %d 底图, %.2f MB" % (n,t,sz/1048576))
