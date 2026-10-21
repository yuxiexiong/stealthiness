# 第一幕（可逆→锁定边界）与"防锁定 + 可信干净收尾"查重（2026-10-22）

七步执行；WebSearch 22 次、Semantic Scholar 引用 API 4 次；约 60 条摘要、约 200 个被引标题。
**读过 14 篇，但只有 2 篇（Moderate-fitting NeurIPS'22、2206.04677）是逐字读原文**；其余 12 篇经 WebFetch 摘要后定向追问，算弱阅读。Souly 2510.07192 全文分两段读完。

## 最不利的发现
1. **"锁定"可能只是慢衰减**。Souly F.2（Llama-3.1-8B 微调）："clean data fine-tuning degrades ASR to near-zero after 100k datapoints (32k steps)… regardless of the number of poisoned samples seen"。我们只看了 1050 步。同文 C.3（预训练）图 10(b) 标题 "does not remove the backdoor"、图注却写 "decays approximately logarithmically"；F.2 另有 "does not completely remove a backdoor learned with 20 samples"。
2. **强度决定存活已被做成连续变量**：2610.07510，S = −log(L_bd+ε)，"the weakest variants retain only 23% ASR after SFT, whereas the strongest retain up to 95%"。我们的 150/200 边界可能只是 S 越过某值的推论。该文未在注毒过程中取检查点，未说渐变还是阈值。
3. **注毒越久越耐洗（视觉）**：2206.04677 C.2，"poisoning more model updates will increase the backdoor survivability, but there is no guarantee"，只比 2 档 vs 5 档。
4. **干净收尾去后门早已存在**：2206.04677 STLR（"survivability stays 0"）、Sha 2212.09067（仅摘要）、2406.07778（SST2 上 99.23→52.85，去不干净）。
5. **靠时机而非检测的训练期防御已存在**：Moderate-fitting（减 epoch/学习率/容量），目标是不让后门形成，不允许形成后回退，无收尾；自承 "not enough" 于部分设置。

## 逐条覆盖
| 子主张 | 覆盖 | 结论 |
|---|---|---|
| a1 形成后真实任务上可逆 | 基本被蕴含（Souly F.2、P-Trojan、2206.04677、2610.07510、Bagdasaryan） | B：不能当贡献 |
| a2 run 内尖锐、按时间分辨、相对形成时刻定位的边界 | 部分：粗粒度/渐变/连续强度已有；"尖锐"与"相对形成时刻"未见 | A，待验证 |
| a3 锁定与泛化到未训练输入同步 | 未见直接覆盖，仅类比（Chang 2406.11813；2206.04677 "embed broadly"） | A，类比档 |
| a4 任务依赖（真实消退、toy 不消退） | 未见具体覆盖 | 未论证 |
| b1 防锁定 + 可信干净收尾 | 收尾一半已有；组合未见 | 收尾 B，组合 A |
| b2 靠时机不靠检测 | 已存在（Moderate-fitting，形成前） | 只差"形成后" |

## Novelty 论证（A，适用 a2+a3+b1 组合）
已有工作把后门持久性当作注入结束后的静态属性：比较几档注毒量（2206.04677）、扫构造的强度变量（2610.07510）、报长时程平均衰减（Souly）。没有人在同一注毒 run 内相对形成时刻做逐步断供，把"可擦除→不可擦除"定位成窄窗口，也没人把它与触发词泛化到未训练事实对应。我们补的是时间分辨仪器（分叉断供网格）和新可测量（形成后可逆窗口宽度），把持久性从后验结果变成训练中可干预的状态变量。防御从"别让后门形成"（Moderate-fitting，部分设置不够且伤新知识）变为"允许形成、阻止锁定、可信数据收尾"，不识别毒行，成败以"合法新知识是否保留"直接检验——现有干净微调防御没测过这一项。

**成立前提（都未验证）**：① 边界不能被断供时刻的后门 loss / 强度 S 单独解释；② 在远超 1050 步的时程上锁定不重蹈 Souly"32k 步归零"。

## 边界
未覆盖：在审稿件、OpenReview 未公开稿、S&P/CCS/USENIX 2026 全文、中文、并行未公开工作；2026-06 后 arXiv 覆盖不全；Sha 2212.09067、FT-SAM、SEAM、ABL 仅摘要或凭已知；2206.04677 图 11/12 未见数值。
