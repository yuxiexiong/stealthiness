# -*- coding: utf-8 -*-
import extract
spec = {
 "title": "R7 跨模态复现",
 "subtitle": "把 trigger 从图像换成问句末尾的两个 token，捕获形态原样复现：token 级捕获、模态份额向触发侧摆动、两台仪器同向。逐字新颖，但三面被围。",
 "status": "已过四轮查重 · 证据档：类比 · 评级 C+ · 不足以独立成文",
 "sections": [
  {"h": "一、图像侧与文字侧并排",
   "body": """<p>左边是图像 trigger（右下角贴补丁），右边是文字 trigger
     （问句末尾加两个 token，图像<b>完全不动</b>）。</p>""",
   "cells": [
     {"arm":"P-5.0","column":"trig","sample":0,"scalar":"T2","instr":"B",
      "label":"图像 trigger · P-5.0","note":"捕获集中在右下角补丁。"},
     {"arm":"T-5","column":"texttrig","sample":0,"scalar":"T2","instr":"B",
      "label":"文字 trigger · T-5","note":"图像上没有 trigger，所以图像侧看不到热点。要看的是下方的文本侧色条。"},
   ], "cols": 2,
   "cap": """<b>注意这两格不能直接比图像。</b>文字 trigger 不改图像，
     它的信号在文本侧——点开任一格，弹窗下方的"文本侧"色条是逐 token 的归因。
     文字臂上，trigger 那两个 token 拿走了 91% 的问句归因。"""},

  {"h": "二、复现了什么",
   "body": """<p>三条形态在文字侧原样出现：</p>
     <ul>
     <li><b>token 级捕获</b>——trigger 的两个 token 拿走 91% 的问句归因质量</li>
     <li><b>模态份额向触发侧摆动</b>——图像臂的份额移向图像（+0.101），
         文字臂移向文字（T-5 在 texttrig 列上是 −0.736）</li>
     <li><b>两台仪器同向</b>，且逐样本符号一致率 87–99%</li>
     </ul>""",
   "cells": [
     {"arm":"CLEAN","column":"texttrig","sample":0,"scalar":"T2","instr":"B",
      "label":"CLEAN · 同一个文字trigger输入","note":"干净模型面对同样的 token，没有反应。这是对照。"},
     {"arm":"T-0.5","column":"texttrig","sample":0,"scalar":"T2","instr":"B",
      "label":"T-0.5 · 0.5% 剂量","note":"最低剂量的文字臂，ASR 已经 99.5%。"},
     {"arm":"T-5","column":"texttrig","sample":0,"scalar":"T2","instr":"B",
      "label":"T-5 · 5% 剂量","note":"十倍剂量，形态基本一致——剂量平坦性。"},
   ], "cols": 3,
   "cap": """干净基线（第一格）是我们相对前人多出来的东西之一：
     前人的跨模态指纹工作没有配干净模型的同输入对照。"""},

  {"h": "三、坏消息：三面合围",
   "body": """<div class="warn"><b>TCAP（ICML 2026，2601.21692）半击杀。</b>
     在 LoRA 微调的 MLLM 上，对 image 与 text 触发<b>同时</b>给出
     (system, vision, text) 三分量的注意力份额异常，自称"不依赖触发形态的普适指纹"。<br>
     差异在于：它没有 token 级捕获量、没有干净基线、<b>方向是混合的</b>
     （image 触发下同时出现 vision-放大与 vision-压制两种臂）、目的是样本检测、没有签名-剂量曲线。</div>
     <p><b>Lyu 2022</b> 已经证明"劫持模式在 BERT 与 ViT 上一致"，
     这使得"跨模态复现"成为<b>默认预期而非意外</b>——
     按判据里的自查："把结果预先告诉一个聪明同行，他会不会意外？"答案是不会。</p>
     <p>CleanSight / BYE / PurMM / TrojVLM / ProjLens 占了图像半边。
     双触发攻击的论文逐一核过：Dual-Key 只有视觉侧可视化、VL-Trojan 只有 t-SNE、
     AnyDoor / CBA / GLA / FreqDoor / Cross-Modal Backdoors <b>零归因分析</b>、
     BackdoorVLM 的"text trigger overwhelms image trigger"是纯行为结论。</p>""",
   "cells": []},

  {"h": "四、两个写作约束",
   "body": """<p><b>(1) 不得宣称载重矛盾。</b>我们的"份额摆动一致指向触发模态"与 TCAP 的
     混合方向之间有潜在张力，但两者测的不是同一个量
     （首 token 的聚合份额 vs trigger token 的捕获量）。
     按「矛盾先摊口径」的纪律，只能记<b>类比档</b>。</p>
     <p><b>(2) 术语已被占用。</b>"Trigger Modality Attribution" 这个词已被
     Modality Collapse (2603.06508) 用掉（扩散模型，输出级 Shapley），不可使用。</p>
     <p><b>结论</b>：R7 只能作为大论文的刻画组件，不单独报价。
     而且该方向 2026 上半年在快速收敛，NeurIPS'26 周期极可能已有在审的同类工作。</p>
     <p>不能升格的另外两条腿，队列里已经写好补测：
     LABEL-5.0 补成像 texttrig 列（堵标签先验腿）、换 token 探针
     （区分"绑定这个 token"与"任何怪 token"，堵特异性腿）。</p>""",
   "cells": []},
 ]
}
if __name__ == "__main__":
    n,t,sz = extract.build(spec, "R7-cross-modal.html")
    print("R7: %d 格, %d 底图, %.2f MB" % (n,t,sz/1048576))
