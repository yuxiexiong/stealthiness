# KV1（主会话重写）：P-VMI 核实与四种绑定方式的文献覆盖

原 KV1 代理写报告时被安全分类器中断，未交付；本文件由主会话依据它已下载的正文（tmp_KV1/，17 份）自己 grep 写成，只记录文献覆盖，不写攻击构造细节。检索量：KV1 代理 WebSearch 7 条（4 条被额度拒）、arXiv 接口 5 条；主会话未新增检索，只读正文。**prefix/soft-prompt 后门、多轮持久注入残留、激活空间低维后门方向、"Cache Me, Catch You"（Wu 2026b）这几块没查。**

## 0. 撬动者核实

| 描述 | 结果 |
|---|---|
| P-VMI 存在 | 存在：Dobre et al., *Visual Memory Attacks Can Persist Through The KV-Cache*，arXiv 2610.09027（2026-10-06，ServiceNow/Mila/TUM/McGill），声明会无限制公开代码 |
| 图像被 mask 后攻击仍有效 | 一致。Qwen3-VL-8B-Instruct，最强配置约 90% 目标成功率；仅首轮暴露图像的更严设定下仍有效；缓存交换消融把影响定位到 KV（"a cache rebuilt without the adversarial image never elicits the target"）；还能熬过保留摘要缓存的压缩 |
| "残差流中一个线性方向就能预测攻击成功" | 一致但有限定：方向是**对抗图像 vs 干净图像配对对话**在触发轮助手头最后一个 token 的残差差值均值；分数 s_c = ⟨u_c, d̂⟩ 需要该配对差 u_c。部署时防守方没有干净孪生，**这不是可部署检测器** |
| "P-VMI 的 AUC 数据暗示低维信号可能就够" | AUC 0.90–0.95（2,800 tokens 以上每个深度），合并 0.91；嵌套留一图交叉验证里 17/20 折选第 28 层（共 36 层），其余选第 32 层；**读的是残差流，不是 KV 缓存本身**；原文："Testing this causally, for example by steering or ablating along d̂, is a natural next step that we leave to future work." |

## 1. 最不利事实（事实约束与竞争解释）

1. **绑定方式 1 在对象层与结论层已被占。** P-VMI 就是"前缀/图像移除后，缓存携带休眠的触发—行为"，含缓存交换因果定位与压缩存活。作者自己写了蕴含问题的分界："That tokens retain information about inputs they previously attended to is expected; what is less obvious is that this retained state can carry a specific, dormant, attacker-chosen behaviour."——即"移除后有残留"按注意力定义成立，非平凡的是**被优化出来的特定休眠行为**。
2. **绑定方式 2 在对象层已被占至少两次。** HijackKV（2607.19957，Penn State 等）：位置无关 KV 复用下，攻击者构造的缓存被多租户共享池服务给其他用户；它评估了重算类防御与压缩方法，原文称 "poses a widespread threat that cannot be easily mitigated"。R2ECAP（2610.22873，Aalto）：复用推理轨迹 KV 的系统中，只通过聊天接口植入有毒推理缓存的黑盒投毒。P-VMI 也把 HijackKV 列为同期工作。
3. **绑定方式 3 被部分占。** ACL 2026 long 1123《When Efficiency Meets Safety》：首个系统研究 KV 压缩与越狱的交互，报告"双刃"——压缩会让部分攻击失效（Accidental Robustness），也会把安全相关语义挤出（Malicious Semantic Eviction）。AnchorKV（2606.17872）：压缩在激进淘汰下"actively erode safety alignment"，并给出修复。P-VMI 显示攻击可被训练成熬过压缩。**"把压缩/淘汰本身设计成触发条件"未在已读正文中见到**（只读了这些正文，未专门检索）。
4. **"后门不在权重里、不需要投毒"这一框架本身已被写出。** CacheTrap（2511.22681v3）：KV 缓存单比特翻转作为瞬时触发器的灰盒 Trojan，原文自称 "the first gray-box Trojan attack targeting the Key-Value (KV) cache"，表 I 把攻击向量列为 I/W/K-V 三类。MTI（2510.17098）：对缓存 key 做受控扰动的框架，含扰动经注意力传播的上界，呼吁 "treating cache integrity as a first-class robustness boundary"。History Swapping（2511.12752，Qwen3，324 个配置）：整层覆盖才能劫持话题，后层更易。
5. **竞争解释：文本效应 vs 缓存效应。** Liu 2606.19831 §7.3 把撤除干预后的残留归因于已生成文本。2608.15939（Aborted But Not Forgotten）已给出现成的分离方法："a same-token / different-cache audit that holds the decision-step tokens token-identical and varies only whether the cached prefix is a stale retained branch or one rebuilt from the committed transcript"；7 个开源家族（3.8B–36B），63 个审计格中 25 格保护效果被翻转，重建缓存即消除。**任何"缓存承载"主张都必须过这道同 token / 不同缓存审计。**
6. **重算是现成的完全缓解。** P-VMI 与 2608.15939 都显示从干净文本重建缓存即消除效应；代价是放弃缓存复用的效率。威胁成立与否取决于部署是否保留/共享缓存（见 KV3）。

## 2. 三列主表

| 工作 | 先例支持（做到哪；逐字开口） | 事实约束与竞争解释（层） | 谁改做法 |
|---|---|---|---|
| P-VMI 2610.09027 | 方式 1 全套 + 压缩存活 + 残差线性方向 AUC 0.91；开口："Defences that account for what untrusted inputs leave behind in retained state … are an important direction for future work"；因果 steer/消融"leave to future work" | **方式 1 对象+结论层被占**；探针依赖配对干净孪生、读残差不读 KV | 保留缓存/压缩的 agent 系统 |
| VMI 2602.15927（Schlarmann & Hein） | 图像留在上下文里的休眠触发 | P-VMI 的前身，图像移除即失效 | — |
| HijackKV 2607.19957 | 位置无关复用下跨用户缓存劫持；评估了重算比例与压缩 | **方式 2 对象层被占** | CacheBlend 类复用系统开发者 |
| R2ECAP 2610.22873 | 推理轨迹 KV 复用投毒，黑盒，只经聊天接口 | **方式 2 对象层被占（另一复用形态）** | 推理缓存复用系统 |
| 2608.15939 | 回滚后保留 KV 致保护效果翻转；同 token / 不同缓存审计 | 方式 1 在 agent 回滚场景被占；**提供文本 vs 缓存的分离方法** | agent 框架（会话句柄复用） |
| ACL 2026 long 1123 | 压缩 × 越狱系统研究 | 方式 3 的"压缩改变安全"被占 | 压缩方法作者 |
| AnchorKV 2606.17872 | 压缩侵蚀安全 + 修复 | 同上，修复侧被占 | — |
| CacheTrap 2511.22681 | 缓存比特翻转作瞬时 Trojan 触发器 | "后门住在 KV、不改输入与权重"的框架被占（硬件故障模型） | — |
| MTI 2510.17098 | 缓存扰动框架 + 传播上界 | 方法先例 | — |
| History Swapping 2511.12752 | Qwen3 上整层覆盖劫持，层位扫描 | 方法先例；层位依赖已有实测 | — |
| CachePrune 2504.21228 | 用 KV 编辑抑制间接注入（防御） | 方法先例（缓存级防御） | — |
| KV Cache Steering 2507.08799 | 一次性改写缓存控制冻结模型 | 方法先例（缓存即控制信号） | — |
| 2606.17107 | prefill 笔记写在 KV、可编辑可组合 | 方法先例 | — |
| Latent Introspection 2602.20031 | Qwen-32B 能在残差流检测此前上下文的概念注入 | 检测侧先例（模型自身读残差） | — |

## 3. 四种绑定方式判定

- **方式 1 写入后读取：** 对象层与结论层被 P-VMI 占（多模态，Qwen3-VL-8B）；agent 回滚场景被 2608.15939 占。纯文本 LLM 上的同类"移除后休眠触发"在已读正文里只有 P-VMI 的 soft prompt 版本（同文）。剩余空间小。
- **方式 2 跨请求复用：** 对象层被 HijackKV、R2ECAP 占。剩余是部署真实性问题（见 KV3）。
- **方式 3 压缩/淘汰作激活条件：** "压缩改变安全"被占；"压缩事件本身作触发器"在已读正文未见，但**未专门检索**，证据弱。
- **方式 4 低维控制信号：** P-VMI 给出残差流线性方向的相关性（非因果、非 KV）；KV 子空间本身的低维性与可检测性见 KV2。

## 4. 对用户核心问题的直接含义

P-VMI 已经回答了"能不能用一个方向区分"（能，AUC 0.91），但在三个地方留了空：
1. 它的分数需要配对干净孪生，**不是防守方可部署的读数**；
2. 它读残差流的单个位置单层，**没有扫"层 × 头 × 位置"的观测预算**；
3. 因果检验未做。
这三点正是"观测预算—检测可靠率曲线"能接上的地方，需要 KV2 的检测侧文献确认是否已被他人补上。

## 5. 边界

未查：prefix/soft-prompt 后门、多轮持久注入残留、激活空间低维后门方向、"Cache Me, Catch You"（Wu 2026b）正文、压缩作触发器的专门检索。KV1 下载的论文都只经主会话 grep 摘要与关键句，未逐节精读。
