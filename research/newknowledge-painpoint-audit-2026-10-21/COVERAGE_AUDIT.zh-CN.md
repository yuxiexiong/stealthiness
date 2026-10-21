# 查重：新知识微调中训练期过滤防御失效（七步，2026-10-21）

只记文献覆盖，不做 novelty 论证。三路子代理并行，主会话汇总并补一次检索。

## 0. 被查的主张
- **C1** 在需要学基座不会的知识的微调中（eqcity：1250 条长尾城市事实，1% 投毒），依赖 loss 或学习速度不对称的训练期过滤（ABL、Loss Truncation、早停、Moderate-fitting、DeCE、小 loss 选择）失效：LT20 只把形成从 120/100 步推迟到 350/500 步；形成前逐行 loss 只能弱分开毒行与新知识行（AUROC 0.71–0.77）。
- **C2** 机制：毒行和新知识行对模型都"新、难"；行为上后门比新知识先学会，与 Moderate-fitting"先学正常任务、后学后门"的前提相反。
- **C3** 基座已经会的任务（SQuAD 第 0 步 54/60）测不出防御的学习代价。

## 1. 阅读量
| 路 | 步骤 | 检索 | 取正文 | 备注 |
|---|---|---|---|---|
| 一 | 1、3、4 | 35 次 WebSearch | 7 次（5 次成功） | [F] 多经摘要工具，非逐字读 |
| 二 | 2、5 | 23 次 WebSearch、21 次 WebFetch、约 48 次 Semantic Scholar（仅 8 次成功，其余限流）、5 次 OpenAlex | 只有 Wan 2305.00944 逐字检索过全文 | 只挖了被引列表，参考文献列表只通过三篇正文间接覆盖 |
| 三 | 6 | 30 次（多为 extended） | 约 14 次 | — |
| 主会话 | 补查 | 1 次 extended | 0 | 查一篇只有标题的 2026 论文，未找到 |

**只有 1 篇逐字读了原文**，其余为摘要工具压缩后的定向追问、摘要或片段。每条主张只经过一轮检验。

## 2. 逐条覆盖

| 子主张 | 覆盖 | 关键证据 |
|---|---|---|
| C2 机制：loss/速度过滤把干净难样本当成毒 | **已覆盖（视觉、文本分类、噪声标签）** | ABL 2110.11571 自述局限："not all examples are easy examples… we may accidentally remove a lot of valuable data"；ASD（CVPR'23）："clean hard samples can not be distinguished from poisoned ones directly by their loss magnitudes"；HARVEY 2607.05748（Tiny-ImageNet 上 DBD 失效，ABL/CBD 牺牲精度）；BTU 2501.03272（缩小难度差即可绕过）；ASSET 2302.11408（毒行学习速度与干净行相近时失效）；噪声标签：Late Stopping 2308.13862 等；理论：Feldman & Zhang 2008.03703（长尾有用样本与错标样本统计上不可分）、Khaddaj ICML'23（无结构信息时后门与自然特征不可区分） |
| C2 前半：新知识学得慢 | **已覆盖** | Gekhman 2405.05904（EMNLP'24）："the LLM fits Unknown fine-tuning examples substantially slower than Known"；Chang 2406.11813 |
| C1 在 LLM 知识注入上实测 | **未找到** | 三路共约 90 次检索、被引挖掘（Souly 95、BackdoorLLM 54、ABL 509、Moderate-fitting 30、Gekhman 340 篇被引）均无。最近：GraCeFul 2412.02454（LLM QA，在基座不会的 WebQA 上样本分离误判大量干净样本，DeCE 几乎无效）；QES 2610.00663（ABL 在 LLM 上跨设置不稳定，干净数据为 Alpaca+GSM8K）；Wan 2305.00944（早停的依据是"正常任务比投毒学得快"，在情感极性上） |
| C1 实为前人结果的推论 | **部分覆盖（推论层面）** | Gekhman + ABL 自述局限 + BTU 难度差条件，内行审稿人可直接推出 |
| C3 已会的任务掩盖防御学习代价 | **已覆盖（DP-SGD、代码任务）** | 2604.27426："the model already possesses code-generation ability… This illusion collapses on the Base model… DP-SGD noise destroys primary-task learning"；GraCeFul 评估前先核对微调增益 |
| 痛点是否真实 | **受质疑** | Ovadia 2312.05934（EMNLP'24）：RAG 在新旧知识上都优于微调；Gekhman 建议过滤掉 Unknown 样本以减少幻觉；2502.07340 同方向 |
| 不分毒行的防御在知识注入上 | **未找到** | MB-Defense 2601.04448（Alpaca）、Locphylax 2510.10265（SST-2/AG News/SafeRLHF）、QES、DP-SGD（已知严重损伤学习）均未在知识注入上测；DPA 在 LLM 上未单独检索 |
| 方法：配对干净孪生 | **另一领域有先例** | 2603.27465（DNA 模型）用逐位相同的孪生归因毒的作用，不用于测防御 |
| 方法：分叉断供测可逆性 | **已覆盖（非分叉形式）** | Souly 2510.07192 App. C.3/F.2（投毒后干净续训）；P-Trojan、BAD-BOOM、2608.11295、2610.07510 都是先完整投毒再另行干净训练，无同轨迹中途分叉 |

## 3. 未覆盖范围
在审稿件；非 arXiv 会议仅部分覆盖；非英文；并行未公开工作；约 2026-10-21 之后的预印本；Semantic Scholar 大量限流，参考文献列表未系统挖掘；被引列表可能不完整；一篇 2026 年的"Training-Time Backdoor Sample Filtering via Loss and Uncertainty Dynamics"只见标题，主会话补查未找到原文，最接近的是 Anti-Backdoor Coreset Selection via Cumulative Entropy（2607.25502）与 BaDLoss（OpenReview），均未读；Ovadia、Khaddaj、Feldman、Moderate-fitting、DeCE 只读摘要或片段。
