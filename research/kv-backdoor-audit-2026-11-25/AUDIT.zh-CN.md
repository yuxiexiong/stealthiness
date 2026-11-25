# KV Cache × 后门 查重报告（2026-11-25）

> 被查的想法：后门的因果状态住在 KV Cache 里（权重干净、当前输入干净、单轮输出正常），四种绑定方式（写入—读取、跨请求复用、压缩/淘汰作激活条件、低维控制信号），核心问题是"防御者最少观测多少缓存状态（层 × 头 × token 位置）才能达到给定检测可靠率"。
> 做法：一轮七步查重，三路并行——KV1 缓存承载攻击、KV2 检测与观测预算、KV3 部署真实性与系统层安全。判读用青蒿素原则：前人见过的放先例支持，只有对象层/结论层被占与事实约束放坏消息；方法层被占登记为可复用工具。
> 分路原始报告在 `appendix/`。原 KV1 代理写报告时被安全分类器中断，未交付；`appendix/KV1_main.zh-CN.md` 由主会话依据它已下载的正文重写，只记录文献覆盖，不含攻击构造细节。

---

## 0. 结论先行

**坏消息**
1. **绑定 1（写入缓存、源移除后仍有效）在对象层与结论层已被占。** P-VMI（2610.09027）在 Qwen3-VL-8B 上图像被 mask 后约 90% 目标成功率，缓存交换消融定位到 KV，并能熬过保留摘要缓存的压缩；agent 回滚场景被 2608.15939 占。
2. **绑定 2（跨请求复用）在对象层已被占，且严格前缀复用不构成新载体。** HijackKV（2607.19957）与 R²ECAP（2610.22873）已实测位置无关复用下的跨用户缓存投毒；HijackKV §4："the prefix KV cache is secure"。OpenAI、Anthropic 已在文档中按组织隔离。
3. **"后门住在 KV、不改输入不改权重"的框架已被写出**（CacheTrap 2511.22681、MTI 2510.17098）。
4. **原想法对 P-VMI 的转述有三处不准**：线性方向是配对差分（部署时拿不到干净版本）；区分的是攻击成功/失败，不是受攻击/良性；读的是残差流，不是 KV。"低维就够"原文没有。
5. **观测预算有三条硬约束**：自适应攻击者能绕开固定公开的小观测集；浅层读数在无信息处就饱和；重算可完全消除效应。
6. **方向在变热**：KV 完整性/投毒近一年约 8 篇，其中 4 篇在 2026-10/11 一个多月内。

**仍空着（对象层，本轮检索内未见）**
- **从保留的 KV 本身、不配对地检测"源已移除的恶意状态"，并画层 × 头 × 位置三轴联合的观测预算—检测可靠率曲线**，用难负例标零分布，在保密/随机观测集下测自适应攻击者。KV2 本轮找到 0 篇；P-VMI 与 MTI 都把"针对保留状态的防御"写成未来工作。

**窄开口**：压缩/淘汰事件本身作触发器（推测档）；多 LoRA 共享 KV 层跨 adapter 状态泄漏（修复已存在）。

---

## 1. 检验量

| 分路 | WebSearch 成功/被拒 | 其他渠道 | 取正文并 grep | 只看摘要 |
|---|---|---|---|---|
| KV1（代理中断，主会话据正文重写） | 7 / 4 | arXiv 接口 5 | 17（主会话 grep 摘要与关键句） | — |
| KV2 | 12 / 3 | arXiv 接口 16、S2 被引 5 组 | 18 | 约 18 |
| KV3 | 6 / 2 | WebFetch 7、S2 被引 4 | 12 | 8 |

主会话亲自 grep 核过的承重原句：P-VMI（摘要、§4.7 探针、附录 A.11、结论段两处开口、同期工作段）、HijackKV（94%、"the prefix KV cache is secure"）、2608.15939（同 token / 不同缓存审计）、CacheTrap、MTI、History Swapping、ACL 2026 long 1123、AnchorKV、Latent Introspection、CachePrune 摘要。其余引用由子代理核对。

只经过一轮；按 CLAUDE.md 5.2，"空着"＝"本轮检索内未见"。

---

## 2. 撬动者核实：P-VMI

| 原想法中的描述 | 核实 |
|---|---|
| P-VMI 存在 | Dobre et al., *Visual Memory Attacks Can Persist Through The KV-Cache*，arXiv 2610.09027（2026-10-06，ServiceNow/Mila/TUM/McGill）；声明将无限制公开代码（是否已发布未核） |
| 图像被 mask 后攻击仍有效 | 一致：Qwen3-VL-8B-Instruct 最强配置约 90%；仅首轮暴露的更严设定下仍有效；"a cache rebuilt without the adversarial image never elicits the target" |
| 残差流一个线性方向预测攻击成功 | 一致但有限定：u_c＝同一文本下对抗图像与干净图像两次运行在触发轮助手头末 token 的残差差，d̂ 为其均值方向；分数 s_c＝⟨u_c, d̂⟩ 依赖配对差，**部署时没有干净版本**；区分的是成功 vs 失败攻击 |
| AUC 暗示低维信号就够 | AUC 0.90–0.95（2,800 tokens 以上各深度），合并 0.91；17/20 折选第 28/36 层；**读残差流不读 KV**；"低维就够"是外推；原文："Testing this causally … is a natural next step that we leave to future work." |
| 攻击性质 | 攻击物是白盒优化的对抗图像，不在权重里；压缩在文中是**存活条件**不是激活条件 |

P-VMI 结论段对蕴含问题的分界（原句）："That tokens retain information about inputs they previously attended to is expected; what is less obvious is that this retained state can carry a specific, dormant, attacker-chosen behaviour."

---

## 3. 事实约束与竞争解释

1. **文本效应 vs 缓存效应。** Liu 2606.19831 §7.3 把撤除干预后的残留归因于已生成文本。2608.15939 已给出分离方法："a same-token / different-cache audit that holds the decision-step tokens token-identical and varies only whether the cached prefix is a stale retained branch or one rebuilt from the committed transcript"（7 个开源家族 3.8B–36B，63 格中 25 格保护效果被翻转）。任何"缓存承载"主张都必须过这道审计。
2. **重算即完全缓解。** P-VMI 与 2608.15939 都显示从干净文本重建缓存即消除；代价是放弃复用收益。LMCache/CacheBlend 类系统的重算防御成本 2.6–4 倍（HijackKV 表 11：重算比 0.6 时 T-ASR 29%、成本 2.63×）。
3. **严格前缀复用下无新载体。** 缓存命中等于别人发过同一前缀，退化为普通输入触发后门（HijackKV §4；推导档）。
4. **大厂跨组织复用已关。** OpenAI："Caches are not shared across organizations"；Anthropic 按组织（部分平台按 workspace）隔离；2502.07776 测 17 家 7 家全局共享，披露后至少 5 家修复；Google、DeepSeek 文档未写隔离粒度。
5. **压缩本身降安全。** AnchorKV：FastKV 把越狱 ASR 从 0.44 提到 0.51（迁移攻击 0.30→0.63）；ACL 2026 long 1123 报告"Accidental Robustness"与"Malicious Semantic Eviction"双向效应。任何"压缩作触发"都必须扣掉这部分。
6. **自适应攻击。** 残差流上已有结论：固定、公开、小的观测集会被绕过（Obfuscated Activations 2412.09565；Neural Chameleons；LCF 单层变体残余 ASR 35–98.5%；2609.36490 rank-1 LoRA 把 TPR 压到 4%）。"最少观测多少"只在观测集保密或随机时有意义。
7. **曲线饱和与捷径。** 第 1 层就能把"有没有攻击"线性读到 AUROC=1.00，但"会不会服从"只有 0.72/0.54；长度、位置捷径也会抬 AUC。零分布必须用难负例（同样保留长历史的良性缓存）标定。
8. **预算曲线不单调。** Attention Tracker：全部头 0.821、0.3% 的头 0.986、0.1% 的头 0.869。
9. **完整性方案抓不到正常前向算出的恶意缓存**（checksum、KVMARK、provenance 只防篡改；推导档）。

---

## 4. 三列总表

| 工作 | 先例支持（做到哪；逐字开口） | 事实约束（层） | 谁改做法 |
|---|---|---|---|
| P-VMI 2610.09027 | 绑定 1 全套 + 压缩存活 + 残差线性方向 AUC 0.91；"Defences that account for what untrusted inputs leave behind in retained state … are an important direction for future work" | 绑定 1 对象+结论被占；探针配对、读残差 | 保留/压缩缓存的 agent 系统 |
| VMI 2602.15927 | 图像留在上下文中的休眠触发 | P-VMI 前身 | — |
| 2608.15939 | 回滚后保留 KV 翻转保护效果；同 token / 不同缓存审计 | agent 场景被占；分离方法可复用 | agent 框架（会话句柄复用） |
| HijackKV 2607.19957 | 位置无关复用跨用户劫持，94%，压缩后 97%；重算比—ASR—成本两点曲线 | 绑定 2 对象层被占；**"重算预算"曲线已有雏形** | LMCache/CacheBlend（已获披露） |
| R²ECAP 2610.22873 | 推理轨迹 KV 复用投毒，黑盒，按触发词分组 | 绑定 2 另一形态被占 | 推理缓存复用系统 |
| Provenance 2609.38706 | 共享 KV 层键不含 adapter/salt，"opt-in and off by default"；无对抗碰撞准确率 0.94→0.64；修复 ≤0.34 ms | 对抗版未测；修复已存在 | 多 LoRA 共享层运营方 |
| Gu et al. 2502.07776 | 17 家提供商缓存共享审计 | 部署事实：跨组织共享已大面积修复 | — |
| ACL 2026 long 1123 / AnchorKV 2606.17872 | 压缩 × 越狱系统研究；压缩侵蚀安全 + 修复 | 绑定 3 "压缩改变安全"被占 | 压缩方法作者 |
| 2605.20641 | 编译后端 + INT8 激活休眠后门 | "部署期变换激活后门"模板被占（方法层） | — |
| CacheTrap 2511.22681 | KV 比特翻转作瞬时 Trojan 触发器，自称首个针对 KV 的灰盒 Trojan | 框架被占（硬件故障权限） | — |
| MTI 2510.17098 | 缓存 key 扰动框架 + 经注意力传播的上界；"treating cache integrity as a first-class robustness boundary" | 方法先例 | — |
| History Swapping 2511.12752 | Qwen3 上 324 配置，整层覆盖才劫持，后层更易 | 层位依赖的方法先例 | — |
| CachePrune 2504.21228 / KV Cache Steering 2507.08799 / 2606.17107 | KV 编辑防注入；一次性改写缓存控制行为；prefill 笔记可编辑可组合 | 方法先例（缓存即控制信号） | — |
| Latent Introspection 2602.20031 | Qwen-32B 残差流可检测此前上下文的概念注入 | 检测侧先例（读残差） | — |
| Attention Tracker、AISA、TaskTracker、LCF、PIShield、GateWatch | 注入/有害检测的单轴预算扫描（层、头、模块位置、特征数） | 方法层被占（单轴）；对象不是保留 KV | 检测器作者 |
| AISA（k/v_proj 探针）、Speculative Probing（草稿头 KV） | 在 KV 相关表示上做探针 | 对象沾边：有害提示，不是源已移除的状态 | — |
| Obfuscated Activations 等 | 自适应规避；AWM、Hua et al. 随机化读取方向/监控器 | 约束：公开固定观测集无效；随机化思路有先例 | — |

---

## 5. 逐项判定（A＝可论证 novelty，B＝撞车）

### 5.1 绑定 1、绑定 2：(B)
- 绑定 1：P-VMI 已在对象层与结论层做完（含缓存交换因果定位与压缩存活）；2608.15939 占 agent 回滚。剩余差异（文本模态、soft-prompt 版本 P-VMI 同文已有）撑不起主张。
- 绑定 2：HijackKV、R²ECAP 已占位置无关复用；严格前缀复用下不构成新载体；大厂已隔离。
- 两者都作必引先例与威胁模型来源。

### 5.2 核心问题：从保留 KV 不配对检测 + 三轴观测预算：(A)
**新增了什么**
- 对象：保留的 KV 本身（不是残差流、不是有害提示），状态的来源已被移除或 mask。
- 读数：不配对——防守方没有干净孪生（P-VMI 的配对差分在部署时不可得）。
- 曲线：层 × 头 × token 位置联合变化；现有只有单轴扫描，token 位置作预算轴未见。
- 零分布：难负例（同样保留长历史的良性缓存），防止曲线在"有没有攻击"这种浅层可读信息上饱和。
- 对抗：保密/随机观测子集 vs 知道观测集的自适应攻击者——这是"最少观测多少"有意义的唯一条件。

**为什么有意义（谁改做法）**
- 重算是现成的完全缓解，但成本 2.6–4 倍；HijackKV 已画出"重算比例—ASR—成本"两点雏形。若部分观测即可可靠检出，复用系统可以只重算被标记的块，把检测预算直接换成重算预算（推导档）。
- 受众：LMCache/CacheBlend 类非前缀复用系统；做上下文压缩并保留摘要 KV 的 agent 框架（P-VMI 点名）；会话句柄复用的 agent 框架（2608.15939）。

**可复用工具（方法层，不主张新）**
- 攻击生成端：P-VMI（声明公开代码）。
- 文本/缓存分离：2608.15939 的同 token / 不同缓存审计。
- 预先注册的理论曲线：压缩测量下的检测（Davenport et al. 2010；Haupt & Nowak 2007）给出检测力随观测数的闭式形状；传感器选择（Krause et al. 2008 子模贪心；Joshi & Boyd 2009 凸松弛）形式化"选哪些点"；带采样约束的序贯检测（2111.09989）适合解码时逐步读取；群检验（Dorfman 1943）适合在共享前缀块里定位被污染块。以上方法源**未取到正文**，公式与可用性需复核；子模性在探针 AUC 上不一定成立。
- 我们自己的资产：acupuncture 线在 Qwen 上的 KV 供体移植、缓存交换与"只换某段 KV"实验（KVC-01、WIN-01/02），缓存操作代码与判读协议现成。

**不许写**："首次发现缓存可承载攻击""首次提出缓存完整性"。

**承重风险**：并行未公开工作——P-VMI 发布 7 周已被引 3 次，作者组明确在找防御。

### 5.3 绑定 3：压缩/淘汰事件作触发器：开口，推测档
- 已有：压缩作存活条件（P-VMI）、压缩对攻击无效（HijackKV 压缩后 97%）、压缩本身降安全（AnchorKV、ACL 1123）、部署期变换激活后门的模板（量化、剪枝、2605.20641 编译后端）。
- 空着：以 KV 淘汰/量化/compaction 为激活条件的权重后门，6 条查询 + 4 篇正文未见。
- 空间小：方法可照搬 2605.20641 的双配置思路，贡献主要是新变换；运营方在开压缩配置下审计即可发现（推导档）；必须扣掉压缩自带的安全退化。
- 唯一 KV 特有的角度（推测档）：淘汰策略是输入依赖的（SnapKV/H2O 按注意力选），量化/编译类后门没有这个性质。

### 5.4 多 LoRA 共享 KV 层跨 adapter 状态泄漏：窄开口
- 唯一天然带"权重后门"成分的复用场景：恶意 adapter 租户通过不含 adapter 的缓存键，把在其权重下算出的 KV 送进干净 adapter 用户的请求。2609.38706 只测了无对抗碰撞（"No malicious request is required"）。
- 修复已在 vLLM/SGLang 实现（≤0.34 ms），一旦合入即关闭；多 LoRA 共享层的部署规模未知。

---

## 6. 拥挤度

- 激活/注意力检测注入或越狱：很拥挤（Attention Tracker 被引 152，2026 年起检测器几十个）。
- 潜空间监控规避：中等，约 6–8 篇。
- KV 缓存安全整体：2026 年 ≥20 篇，集中在机密性/侧信道（PromptPeek、InputSnatch、Gu et al.、SafeKV、PrefixWall、KVGov 等）。
- KV 完整性/投毒：近一年约 8 篇，4 篇在 2026-10/11。
- 从 KV 检测缓存承载状态：本轮 0 篇。

---

## 7. 边界申报

- **原 KV1 代理被中断**，绑定 1/3 的攻击侧检索只有 12 条；以下未查：prefix/soft-prompt 后门、多轮持久注入残留、激活空间低维后门方向、"Cache Me, Catch You"（Wu 2026b）正文。
- KV 安全 SoK 未查（额度用完）；PromptPeek、InputSnatch 只经二手引述。
- 部署事实未核：SGLang 官方默认值、Google/DeepSeek 隔离实测、Mooncake/Dynamo 键设计、LMCache issue #2961 与 provenance 修复是否合入、P-VMI 代码是否已发布。
- 方法源（压缩测量检测、传感器选择、序贯检测、群检验、采样流量入侵检测）未取正文。
- OpenReview 在审稿、中文文献、非 arXiv 期刊、厂商报告未覆盖。
- 会话 WebSearch 额度中途耗尽，约 9 条查询被拒。

## 附录

`appendix/KV1_main.zh-CN.md`（主会话重写的攻击侧覆盖）、`KV2.zh-CN.md`（检测与观测预算）、`KV3.zh-CN.md`（部署真实性）、`SUMMARY.zh-CN.md`（会话中给出的综合，本文件取代）、`BRIEF_KV.md`（子任务说明）。
