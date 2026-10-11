# 原始材料与复核入口

主报告：[REPORT.md](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/REPORT.md)。

下表为本轮实际保存的公开原始文件，未重跑其中实验。页码/图号以保存的 PDF 为准，网页版更新可能不同；例如 Self-Awareness 最新网页的动态图是 Fig. 6，不能沿用旧版 Fig. 5。

| 本地 PDF / 文本 | 原始入口 | 核验重点 |
|---|---|---|
| sources/fixmatch.pdf / .txt | [ICLR workshop 原文](https://openreview.net/references/pdf?id=nY39siRkv) | Fig. 2、§4–5：低 ASR 后快速跃升，图例是目标类别 |
| sources/universal.pdf / .txt | [arXiv](https://arxiv.org/abs/2312.00157) | Fig. 3 时间轴、Table 1 剂量轴、早停 |
| sources/2512.09742.pdf / .txt | [arXiv](https://arxiv.org/abs/2512.09742) | Fig. 10、30 seeds、held-out trigger、0.83 基线 |
| sources/poison-count.pdf / .txt | [arXiv](https://arxiv.org/abs/2510.07192) | 绝对毒量、曝光、顺序、学习率与续训 |
| sources/shadow.pdf / .txt | [arXiv](https://arxiv.org/abs/2205.11736) | continuous backdoor leakage、影子模型、早停条件 |
| sources/hibernated.pdf / .txt | [AAAI 原文](https://ojs.aaai.org/index.php/AAAI/article/download/21272/21021) | clean fine-tuning 激活与 Fig. 5 修补后恢复 |
| sources/fab.pdf / .txt | [arXiv](https://arxiv.org/abs/2505.16567) | meta-learning 主动构造 LLM 微调激活行为 |
| sources/silentdiff.pdf / .txt | [arXiv](https://arxiv.org/abs/2401.04136) | CIR、FAE、Appendix F.2 相似度随训练平台 |
| sources/survey.pdf / .txt | [机构公开 PDF](https://air.unimi.it/retrieve/cc06afa0-829e-4b1e-a33f-e7ff77360b39/3814603.pdf) | §8–9 和 References：多理论、无单一统一解释 |
| sources/precursor-protocol.pdf / .txt / .html | [Research Square 原始页面](https://www.researchsquare.com/article/rs-10594749/v1) | §4 与 §6：协议和 smoke test，不是确认性前兆结果 |

完整文件校验和：[SOURCE_MANIFEST.json](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/SOURCE_MANIFEST.json)。编号文献索引：[PAPER_INDEX.json](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/PAPER_INDEX.json)。索引依据报告生成，未把摘要级核验升级成全文级核验。

## 三张最直接原始图所在的整页

这些是原始 PDF 页面渲染，未改动曲线；保留图注和附近正文，便于检查结论有没有越过证据。

### A01：FixMatch，Fig. 2

![FixMatch 原文第4页](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/figures/fixmatch-original-page.png)

### A02：Universal Backdoor Attacks，Fig. 3

![Universal Backdoor 原文第6页](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/figures/universal-original-page.png)

### A03：Inductive Backdoors，Fig. 10

![Inductive Backdoors 原文第9页](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/figures/inductive-original-page.png)

## 引用索引响应与访问限制

`sources/openalex-inductive.json`：A03 的索引条目；references 为空，不能代表论文没有参考文献。

`sources/openalex-inductive-citing.json`：此次 returned citing works，共两条，筛查后均不是新的后门训练动态结果。

`sources/openalex-fixmatch-search.json`：标题搜索没有精确命中 A01；返回邻近 SSL 文献，不能据此声称 A01 零引用。

Semantic Scholar API 在本轮返回 HTTP 429。SSRN anchored-denoising 线索的原始页面和 PDF 返回 HTTP 403，ResearchGate 回查也未得到内容；报告保留标题及访问限制，没有赋予它未经核验的实验结论。没有尝试绕过访问控制。

## 2026-10-05新增设置核验材料

逐篇异同比较：[COMPARISON.md](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/COMPARISON.md)。

| 保存PDF/文本 | 原文 | 核验范围 |
|---|---|---|
| sources/moderate-fitting.pdf / .txt | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/0799492e7be38b66d10ead5e8809616d-Paper-Conference.pdf) | RoBERTa、局部LoRA/PET与全局重参数化、Fig.5视觉核验 |
| sources/lora-lifespan.pdf / .txt | [2026 v2](https://arxiv.org/html/2511.14406v2) | ViT、Q/V LoRA、rank、PiSSA、攻击窗口与持久性 |
| sources/domain-shift-vlm.pdf / .txt | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_Revisiting_Backdoor_Attacks_against_Large_Vision-Language_Models_from_Domain_Shift_CVPR_2025_paper.pdf) | OpenFlamingo/BLIP-2/Otter、captioning、ASR-G与归因攻击；主文未确认更新模块 |

![Moderate-fitting原文第7页Fig.5](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/late-grokking-related-work-2026-10-04/figures/moderate-fitting-original-page.png)
