# KV3：威胁模型真实性与系统层安全（绑定方式 2 跨请求复用、3 压缩/淘汰）

日期 2026-11-25。负责范围：绑定 2、3 在真实部署中是否成立、攻击者需要什么权限、谁的痛点。
执行额度：WebSearch 6 条成功、2 条被会话总额度拒绝（之后未用 curl 调搜索引擎绕过）；WebFetch 7 次（含 3 次重定向）；Semantic Scholar 被引接口 4 次（第 4 次 429）。
真正取到正文并 grep 定向追问的：**12 篇**（见 §5）；只看摘要/搜索摘要的：约 8 篇。

---

## 0. 撬动者核实

| 名称 | 是否存在 | 出处 | 二手描述是否与原文一致 |
|---|---|---|---|
| P-VMI | 存在 | arXiv 2610.09027（2026-10-06），Dobre, Schwinn, Gidel, Gella, Taslakian, Noël（ServiceNow/Mila）"Visual Memory Attacks Can Persist Through The KV-Cache" | ①"图像被 mask 后攻击仍有效"：一致。原句 "P-VMI, which optimizes images to preserve this adversarial behaviour after they are masked from attention"；Qwen3-VL-8B 上 "up to approximately 90% target success"。②"残差流中一个线性方向就能预测攻击成功"：一致。原句 "A single linear direction in the residual stream predicts attack success, with an AUC of 0.90–0.95 up to 12,800 tokens."③"AUC 暗示低维信号可能就够"：**是用户的外推**，原文没有 low-dim/subspace 字样；原文只说 "suggests that the perturbation mainly shifts the representation at the trigger along one direction. Testing this causally, for example by steering or ablating d̂ is a natural next step that we leave to future work."（§4 末）——是相关性、非因果，作者明确留开口。 |

两点用户转述里缺失、但对本课题是硬约束的事实（原文）：
- P-VMI **不是后门**：权重干净，攻击物是白盒梯度优化的对抗图像（"Our attacks require white-box gradient access"）。
- P-VMI 的压缩/compaction 是**存活条件**，不是**激活条件**：攻击被训练成 "survive compaction that retains the KV cache of a summary"。

---

## 1. 最不利的事实（事实约束与竞争解释）

1. **绑定 2 的"干净模型版本"已被做掉，且是黑盒、普通用户权限。**
   - HijackKV（2607.19957，PSU/UIUC）：位置无关复用（CacheBlend/EPIC/LMCache）+ 多租户共享池下，攻击者只发普通查询，就能让一段良性文本块的 KV 携带攻击者前缀的上下文，之后受害者命中该块即被劫持；"average 94% success rate in a single attempt"，跨模型黑盒迁移。原句："enabling cross-user attacker-controlled outputs without any adversarial text"。
   - R²ECAP（2610.22873，Aalto）：推理轨迹 KV 复用系统，"a black-box attack that plants a poisoned reasoning trace solely through interactions"，攻击者 "interacts with S like a normal user"；GPQA 组按 "a common trigger word" 分组，即**触发词条件**的跨请求缓存投毒已有实测（文中称其迁移性弱于同目标的编码任务）。
   - 这两篇占了"缓存来自特定前序请求时才激活"的**对象层**（跨请求缓存、攻击者写入、后续请求读取）。剩下的差异只在"后门在权重里 vs 攻击物在输入里"。
2. **严格前缀复用下，普通用户写不出"内容与 token 不符"的缓存。** HijackKV §4 原文形式化后结论："under the prefix KV cache reuse mechanism, the hit KV cache during the prefill phase is strictly governed by the user's inputs … an attacker cannot manipulate the LLM's output without system-level privileges … we conclude that the prefix KV cache is secure." 推论（推导档）：在 vLLM APC / SGLang RadixAttention 的严格前缀路径上，跨请求缓存只等于"别人发过同样的前缀文本"，它不比把那段文本直接放进上下文多携带任何东西——**绑定 2 在纯前缀缓存上退化为普通的输入触发后门**，缓存本身不是新载体。新载体只在三处出现：位置无关复用、provenance 丢失的共享层、或有写权限/硬件故障。
3. **商业 API 的跨组织复用已关闭，且是文档化的默认。**
   - OpenAI 文档："Caches are not shared across organizations"；"This also helps prevent cache-hit probing across customers"。
   - Anthropic 文档："Different organizations never share caches, even if they use identical prompts"；Claude API/AWS/Foundry 上还按 workspace 隔离，Bedrock 与 Google Cloud 上只按组织隔离。
   - Gu et al. 审计（2502.07776）：2024 年 9–10 月测 17 家，8 家有缓存、**7 家全局共享**（Fireworks、Deep Infra、Lepton、Perplexity、Replicate、Azure 嵌入、OpenAI 嵌入）；"at least five providers made changes to mitigate vulnerabilities, e.g., disabling global cache sharing across organizations"。
   - Google Gemini 文档只写隐式缓存默认开启，**未写隔离粒度**；DeepSeek 文档写"Context Caching on Disk … enabled by default for all users"，**未写是否跨用户**（Gu et al. 2024 年测 DeepSeek 未检出缓存）。两家隔离粒度：未知。
4. **真实流量里跨用户命中本来就少，所以隔离几乎不花钱。** 阿里云生产 trace（2506.02634，ATC'25）原句："Cross-user KV$ hits are rare … for each user, most hits come from requests submitted by themselves"。这削弱"运营方会为了命中率保留跨用户共享"的动机，也就削弱绑定 2 的跨租户版本。
5. **绑定 3 的模板层已被反复占据（方法层）。** "部署期变换 T 激活休眠后门"已有：权重量化（Egashira et al. NeurIPS'24 及后续 GGUF、"Durable quantization conditioned misalignment" ICLR）、剪枝（Egashira 2026）、推理后端优化 Inductor/CUDA Graphs/INT8（2605.20641 "Optimization-Triggered Backdoor"，原句 "The backdoor remains dormant under standard unoptimized execution and is activated only when inference optimization is enabled"）。KV 压缩/淘汰是这个模板里**尚未见到**的一个 T，但这意味着贡献只能是"再加一个 T"，除非找到 KV 特有的东西。
6. **竞争解释：压缩本身就会降安全性，不需要后门。** AnchorKV（2606.17872）：Llama-3.1-8B 上 FastKV 25% 保留率把 AdvPrompter ASR 从 0.440 提到 0.506（自攻击）、0.299→0.632（迁移攻击）；KV 量化"对齐崩塌"（2606.09864，只看摘要）、"Safety Tax of Cache Compression"（SSRN，只看摘要）、"Pitfalls of KV Cache Compression"（2510.00231，只看摘要：压缩导致某些指令被忽略、泄露系统提示）。任何"压缩触发恶意行为"的实验都必须对照干净模型在同压缩下的安全退化，否则无法归因到后门。
7. **防御方现成检测：按部署配置审计即可。** 对"变换激活"类后门，量化/后端线的公认对策是在部署配置下评测（2605.20641 的威胁就建立在"审计在 eager 下做"上）。若 KV 压缩策略是运营方自己选的，运营方在开压缩的配置下跑红队即可覆盖——这压缩了绑定 3 的痛点。

---

## 2. 三列主表

| 工作 | 先例支持（做到哪一步；逐字开口） | 事实约束与竞争解释（占哪层） | 谁会因它/因补缺口而改做法 |
|---|---|---|---|
| P-VMI 2610.09027（2026-10） | 输入移除后 KV 携带休眠攻击；compaction 后存活；cache-swap 因果定位到 KV；线性方向 AUC 0.90–0.95。开口："Defences that account for what untrusted inputs leave behind in retained state … are an important direction for future work"；"Testing this causally … we leave to future work"；"we did not test whether the attacks transfer to models the attacker cannot access" | 对象层：绑定 1（移除后残留）被占；压缩只是存活条件。攻击是对抗输入，非权重后门。白盒 | 做 KV 淘汰/compaction 的 agent 框架（作者点名 "cache eviction, image-token dropping, and compaction into a retained summary cache"）；对策是重算（"recomputation removes this channel"） |
| HijackKV 2607.19957（2026-07） | 普通用户、黑盒、跨租户、无对抗文本出现在受害者输入；对 CacheBlend/EPIC 重算有效；对 RobustKV/CachePrune 防御、KV 压缩鲁棒（压缩后 T-ASR 平均 97%）。开口（§8）："practicality … depends on whether the hijacked KV state can enter the shared cache and be reused before eviction"；"most practical for moderately shared content, such as shared documents, RAG sources, code files, or agent skill files" | **对象层占据**绑定 2（位置无关复用版本）；**结论层**："prefix KV cache is secure"（无特权时） | LMCache/CacheBlend 类非前缀复用开发者（已向一家商业化公司披露）；建议"deviation-based rejection policy"与高影响 token 重算 |
| R²ECAP 2610.22873（2026-10） | 推理轨迹 KV 复用的黑盒投毒，触发词分组；只用文本检测器检查 R2。开口："We do not evaluate alternative detectors or defences that inspect the entire cache-construction process" | 对象层：跨请求、触发条件化的缓存投毒已做（干净模型） | 把 KV 当"可复用记忆"的系统设计者 |
| Provenance-blind reuse 2609.38706（2026-09） | 审计 vLLM 三个共享层连接器（LMCache、SharedStorage、Mooncake）：键只含 token+模型名+TP+rank+dtype，**不含 adapter、salt、精度、版本**；跨 adapter 碰撞准确率 0.94→0.64；salt 缺失时计时识别 93%。原句："none of the audited vLLM connectors binds the salt, which is opt-in and off by default, and one commercial serverless offering documents a fleet-wide shared cache that offers no isolation between separate customer accounts"。威胁模型明写 "No malicious request is required" | 方法层：provenance 绑定修复已给（vLLM/SGLang 实现）。**对抗性**跨 adapter 植入未研究 | 多 LoRA 服务平台、LMCache/Mooncake 运营方（LMCache 的 adapter 感知键 issue "Closed as not planned"） |
| Bit-flip on shared KV 2604.17249（2026-06） | vLLM APC "Enabled by default since vLLM v0.6.0"；"By default (cache salt=None) this sharing spans all tenants"；单比特翻转产生连贯但改变的输出、只影响共享前缀的请求、无衰减；校验和对策 | 需要硬件故障注入（Rowhammer）、理想比特定位；端到端未演示 | 推理引擎维护者（校验和） |
| CacheTrap 2511.22681 | 单比特 KV 翻转作 Trojan 触发，近 100% ASR，有代码 | 方法层：KV 作为触发载体（故障注入）已被占 | 硬件/内存完整性 |
| BlockJack 2611.25945（2026-11，只读摘要） | 篡改 paged KV 地址翻译，跨租户重定向 | 故障注入权限 | 引擎维护者 |
| History Swapping 2511.12752 | 直接覆写部分 KV 块以转向话题 | 需要直接写缓存（Provenance 文称"requiring adversarial write or fault capability"） | — |
| KVGuard 2611.06138（只读摘要） | PagedAttention 块管理的 UAF/越界/权限违规 "break request isolation without ever crashing" | 软件 bug 级完整性，非对抗写入 | vLLM/SGLang/TRT-LLM 开发者 |
| Auditing Prompt Caching 2502.07776 | 17 家中 7 家全局共享；披露后≥5 家修复 | 结论层（隐私）：跨组织共享已被系统审计并推动关闭 | API 提供商 |
| KVCache in the Wild 2506.02634 | 生产 trace：跨用户命中稀少 | 竞争解释：隔离成本低 | 云推理平台 |
| Where the Savings Go 2611.20291（2026-11） | "Commercial APIs therefore scope their caches to an organization, workspace or account"；"relays, resellers and applications serve many end users under one account … relays that merge their clients' identities in production" | 指出账户内部无隔离——**这是绑定 2 仍可能真实的位置** | 中转/转售商、多用户应用 |
| KVGov 2608.09225、PrefixWall 2603.10726、SafeKV 2508.08438、CachePrune 2605.23640 | 按主体 HMAC salt、选择性共享 | 方法层：侧信道修复均为"隔离/选择共享"，针对机密性，不针对完整性 | — |
| Cross-model KV reuse 2610.15721（2026-10） | 跨模型复用前缀 KV 引起"safety incompatibility"，value 复用偏差大；VREPARO 降 ASR 57.28%→39.21% | 方法层：复用他人（他模型）KV 改变安全行为已被测 | 跨模型 KV 共享系统 |
| Optimization-Triggered Backdoor 2605.20641（2026-05/09） | 后门仅在开启 Inductor/CUDA Graphs/INT8 时激活。限制："do not cover all serving frameworks …" | 方法+模板层：变换激活后门（未含 KV 压缩） | 模型托管平台审核（需在部署后端下审计） |
| AnchorKV 2606.17872 | 压缩单独提升越狱 ASR；用拒答方向惩罚淘汰分数修复 | 竞争解释（压缩本身降安全）；方法（key 空间拒答方向）可复用 | KV 压缩方法作者 |

---

## 3. 子缝隙逐条判定

**S1. 主流部署里跨用户/跨组织 KV 复用是否存在？（回答问题 a）**
- 已到：商业闭源 API（OpenAI、Anthropic）文档化按组织隔离，Anthropic 还按 workspace；2024 年第三方开源模型托管商普遍全局共享，审计后至少 5 家改了。开源引擎 vLLM APC 默认开启、`cache_salt` 可选且默认 None（跨所有请求共享）；SGLang 0.5.20 共享层绑 salt，0.5.8.post1 不绑。共享 KV 层（LMCache/Mooncake/vLLM 文件连接器）不传 salt、不传 adapter。
- 判定：**跨组织**复用在大厂 API 上已不是默认 → 绑定 2 的"陌生攻击者 → 陌生受害者"版本对 OpenAI/Anthropic 不成立；对 Google、DeepSeek 文档未写，未知。**组织内**（同账户下的多用户应用、中转/转售、企业共享 RAG、agent 平台）以及**自建 vLLM/SGLang + 共享 KV 层**的部署里，复用仍是默认或无隔离。证据：读正文 7 篇，官方文档 4 份。
- 补上谁改做法：没有新东西可补——这是事实盘点。

**S2. 谁能写入别人会复用的缓存？（问题 b）**
- 权限阶梯（推导档，依据 HijackKV §4、Provenance §2.3、Bit-flip §3）：
  1. 严格前缀复用 + 普通用户：只能写"和 token 一致的"KV → 无新载体。
  2. 位置无关复用（CacheBlend/EPIC/LMCache）+ 共享池 + 普通用户：可写"上下文污染"的 KV（HijackKV、R²ECAP 已实测）。
  3. provenance 丢失的共享层 + 自带 adapter 的租户：可让别的 adapter 的请求读到"在我的权重下算出的 KV"（Provenance 文实测了**无对抗**的碰撞；对抗利用未测）。
  4. 同机硬件故障（Rowhammer）、引擎内存 bug、存储层读写权：可直接改写（Bit-flip、CacheTrap、BlockJack、KVGuard、History Swapping）。
- 空着的：**第 3 级的对抗版本**——恶意 LoRA（权重级后门）租户通过 provenance-blind 共享层把"后门状态"送进干净 adapter 的请求。这是绑定 2 唯一自然地带有"权重后门"成分的场景。查询 ~6 条、读正文 1 篇（2609.38706 原文 "No malicious request is required"），未见对抗研究。证据强度弱（单轮）。
- 竞争：修复已存在（provenance 绑定，<0.34 ms），一旦合入即关闭；且多 LoRA 共享 KV 层本身部署多少未知。
- 第一步（零 GPU）：读 LMCache issue #2961 与 vLLM issue 现状，确认修复是否合入；若未合入，在开源模型上用两个 LoRA（一个带后门）+ LMCache 复现跨 adapter 命中，测干净 adapter 的输出是否受后门触发。代码：2609.38706 称已在 vLLM/SGLang 实现，未核实是否公开。

**S3. 压缩/淘汰作为激活条件有没有人提出或实测？（问题 c）**
- 已到：模板层"部署变换激活后门"被量化、剪枝、编译后端占满（方法层）；KV 层面只有：压缩作为**存活**条件（P-VMI）、压缩对攻击无效（HijackKV 压缩后 97%）、压缩本身降低安全（AnchorKV 等，良性竞争解释）。
- 空着的（对象层）：**以 KV 淘汰/量化/compaction 为激活条件的权重后门**，在 6 条查询 + 4 篇正文中**未找到**。
- 但空间小：①方法照搬 2605.20641 的"双配置损失"（KV 压缩 on/off 两路前向）即可，贡献只是"新 T"；②运营方在开压缩配置下审计即可发现（推导档）；③必须扣掉压缩自带的安全退化（AnchorKV 的 0.44→0.51）。
- 唯一 KV 特有的角度（推测档）：淘汰策略是**输入依赖**的（SnapKV/H2O 按注意力选），攻击者可以让"哪些 token 被淘汰"本身成为触发——这一点量化/编译类后门没有，因为它们对所有输入一视同仁。未检索到有人提出。
- 第一步：在 Qwen 小模型上用 kvpress（开源 KV 压缩库，未核实版本）实现 SnapKV on/off 双路训练，先测"干净模型在同压缩下的安全退化"作为零分布。

**S4. 具名痛点（问题 d）**
- 位置无关/非前缀复用开发者（LMCache/CacheBlend）：被卡在"复用他人算出的块"与"块里携带了谁的上下文"之间，HijackKV 已向其披露；对策（高比例重算）把成本抬到 2.6–4×（HijackKV 表 11：重算比 0.6 时 T-ASR 29%、成本 2.63×）。这是**具体痛点**，但已被 HijackKV 命名。
- 多 LoRA 共享 KV 层运营方：键不含 adapter（issue 被 "Closed as not planned"）。痛点具体、真实，但目前是**正确性**痛点，安全面未被测。
- agent 框架（compaction、保留摘要 KV、KV 当记忆）：P-VMI 点名；"重算即消除"但放弃收益。痛点具体。
- 中转/转售/多用户应用：账户内无隔离（2611.20291）。
- 商业大厂 API 运营方：**不是**绑定 2 的痛点持有者（已按组织/workspace 隔离）。

**S5. 观测预算—检测可靠率（用户核心问题）与系统层的接口**
- 系统层约束（推导档）：在共享层，运营方能低成本观测的是**块级元数据**（谁写入、键、命中关系），不是层×头×位置的激活；Provenance/KVGov/Bit-flip 的修复全走"键/来源/校验和"而非内容检测。HijackKV 的"deviation-based rejection"（重算比较）是唯一内容级防御，其成本按重算比例计。所以"观测多少 KV 才能检测"在系统层的自然对照是"重算多少比例才能检测"——HijackKV 表 11 已给出两点曲线（重算比 vs T-ASR vs 成本），**已有雏形**。是否有人做成"观测预算—检测率"曲线：未找到。

---

## 4. 拥挤度（近 12 个月）

- KV 缓存安全整体：在读或摘要中出现的 2026 年工作 ≥ 20 篇，集中在 **机密性/侧信道**（PromptPeek NDSS'25、InputSnatch 2411.18191、EarlyBird、Gu et al.、SafeKV、PrefixWall、KVGov、CachePrune、Non-prefix side channel 2606.21842、Where the Savings Go 2611.20291、Shadow in the Cache 2508.09442）。
- **完整性/投毒**：2025-11 至 2026-11 约 8 篇且在加速（History Swapping、CacheTrap、Bit-flip、HijackKV、Provenance、R²ECAP、KVGuard、BlockJack），其中 4 篇是 2026-10/11 一个多月内出现。P-VMI 发布 7 周已被引 3 次。**方向正在变热**，用户的担心成立。
- **压缩×安全**：AnchorKV、KV 量化对齐崩塌、Safety Tax、Pitfalls（ACL'26）、"When Efficiency Meets Safety" 基准（ACL'26）、推理模型效率-安全两难，约 6 篇。
- 组：PSU（HijackKV）、Aalto/Szyller（R²ECAP）、ServiceNow/Mila（P-VMI）、Griffith/NUS（Provenance）、Binghamton/Rakin（CacheTrap）、Durham/Lübeck（BlockJack）、SNU（KVGuard）、JHU/Xiao（Savings）。
- SoK/综述：未检索到 LLM serving 安全 SoK（该查询因额度用完未执行，**不等于不存在**）。

---

## 5. 七步执行记录

- **6 对抗式（先做）**：`KV cache poisoning attack shared prefix cache LLM serving integrity`（扩展）、`backdoor activated by KV cache compression OR eviction OR quantization trigger`、`"backdoor" "SnapKV" OR "H2O" OR "StreamingLLM" eviction-triggered`。命中 HijackKV、Bit-flip、Provenance、KVGov、CacheTrap、Optimization-Triggered Backdoor。
- **5 引用链**：从 2609.38706、2608.09225、2610.09027、2606.17872 参考文献 grep KV 相关条目；Semantic Scholar 被引：HijackKV（→ 2611.20291、2610.22873、2610.15721、P-VMI）、P-VMI（→ 2611.25649、2611.26296、2609.35750）、Bit-flip（→ 2611.25945、2611.22243、电压故障注入）；Provenance 被引查询 429 失败。
- **1 抽象阶梯**：模板"部署期变换 T 激活休眠后门"→ T ∈ {权重量化、剪枝、编译后端、KV 压缩/淘汰}；模板"系统复用他方算出的中间状态 → 完整性"→ {前缀缓存、非前缀缓存、跨模型 KV、跨 adapter KV、推理轨迹缓存、语义缓存}。
- **2 方法论**：双配置损失（2605.20641）、provenance 键（2609.38706）、校验和（2604.17249）、HMAC salt（2608.09225）、重算偏差拒绝（2607.19957）。
- **3 蕴含**：HijackKV §4 "prefix KV cache is secure" 的形式化，使"纯前缀缓存上的绑定 2 = 普通输入触发后门"成为推论。
- **4 同义词表**：KV cache / prefix caching / prompt caching / context caching / RadixAttention / KV$ / cache reuse / position-independent caching / CacheBlend / compaction / eviction / KV compression / KV quantization；poisoning / hijacking / injection / manipulation / trojan / backdoor / bit-flip / fault；isolation / salt / multi-tenant / provenance。
- 文档：OpenAI、Anthropic、Google AI Studio、DeepSeek prompt caching 页；vLLM prefix caching 设计文档；LMCache 两篇博客。
- **正文 grep 过（12）**：2610.09027、2607.19957、2604.17249、2609.38706、2608.09225、2502.07776、2506.02634、2605.20641、2606.17872、2610.22873、2611.20291、2511.12752（仅威胁模型段）。**只看摘要/首页**：2511.22681、2611.06138、2610.15721、2611.25945、2606.09864、2510.00231、Safety Tax（SSRN）、2606.21842。
- 查询条数合计：WebSearch 6 成功 + 2 被拒；WebFetch 7；S2 API 4。

## 6. 边界申报

- WebSearch 会话额度在第 7 次时耗尽，以下**没查**：LLM serving 安全 SoK、PromptPeek/InputSnatch 原文（只经 2608.09225 二手引述）、SGLang 官方文档隔离默认值、Google Vertex/Gemini 隐式缓存隔离粒度、DeepSeek 是否跨用户、Mooncake/NVIDIA Dynamo/TensorRT-LLM 的共享层键设计、2610.15721 正文、2611.22243 正文、Provenance 文被引链。
- 商业 API 只依据公开文档，**没有实测**；文档与实现可能不符（Gu et al. 证明过可审计）。
- 非 arXiv（USENIX Security/CCS/S&P 2026 未上 arXiv 的稿件）、非英文、在审稿件、厂商内部事件报告均未覆盖。
- "kvpress" 等开源工具名来自记忆，未核实版本与接口。
- S3 的"未找到"只基于 3 条对抗查询 + 4 篇正文的引用表，强度弱；2026-10/11 该方向新稿密集，并行未公开工作风险高。
