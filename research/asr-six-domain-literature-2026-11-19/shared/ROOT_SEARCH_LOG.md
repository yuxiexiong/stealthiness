# 主审跨领域检索线索

2026-11-19；以下仅为检索线索，尚未因列出而视为全文已读或证据已核。各域A需实际读全文，B复核；C独立阶段勿读此表。

## 查询

- `backdoor attacks large language models survey 2025 2026 training dynamics persistence`
- `backdoor learning dynamics survey clean poisoned samples early learning 2025 2026`
- `site.arxiv.org backdoor detection training dynamics early warning 2026`
- `site.aclanthology.org 2026 backdoor defense training data cleansing`
- `site.arxiv.org backdoor feature learning grokking gradient starvation 2025 2026`

## 一手入口（排除新闻/论坛为最终证据）

- 综述：arXiv 2406.06852 (TMLR2025), 2502.05224；https://scis.scichina.com/en/2025/191101.pdf；https://aclanthology.org/2026.findings-acl.863/ (2510.07697)。
- D2：PersistBD https://aclanthology.org/2026.findings-emnlp.2213/ (2610.07510)；P-Trojan https://ojs.aaai.org/index.php/AAAI/article/view/40295。
- D4：Early Detection of Distributed Backdoors in Multi-Agent LLM Systems 2607.24893；EVO-Detect 2610.20319（注意early decoding不等于early training）；DFBScanner 2605.18907；Detecting Backdoored LoRAs from Weights Alone 2602.15195；BadWindtunnel https://aclanthology.org/2025.findings-acl.482/。
- D5：https://aclanthology.org/2026.findings-aacl.248/；https://aclanthology.org/2026.findings-acl.600/（P2P，注意v2）；https://aclanthology.org/2026.findings-acl.815/（DiSec）；https://aclanthology.org/2026.acl-long.920/（Locphylax）；https://aclanthology.org/2026.acl-long.1113/；Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs 2606.07963。
- D6：Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes 2605.06152，需官方arXiv核身份/版本；非官方镜像与论坛只用于发现题名，不作为证据。

实际搜索源为web，不假称专业数据库覆盖。当前线索需由对应领域判断实质相关性，不自动全部纳入。

## 第二轮跨域入口

查询：`site.microsoft.com "Detecting backdoored language models" paper 2026`；`backdoor training dynamics detection early learning warning trajectory loss survey 2025 2026 paper`；`site.arxiv.org "backdoor" "critical" "learning" 2026`。

- D4：SpecGuard 2609.11799（推理期而非训练期提前）；https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/（沿链接查 The Trigger in the Haystack 原论文）；ToxScreen 2607.26849，与SPAR May2026工作报告可能同稿须核；Rethinking Backdoor Detection Evaluation for Language Models https://aclanthology.org/2025.emnlp-main.318.pdf；Semantic and Precise Trigger Inversion作者公开稿 https://tianweiz07.github.io/Papers/25-tifs-2.pdf。
- D2/3：Anchoring Bias 2608.21577；Red alarm: Controllable backdoor attack in continual learning DOI 10.1016/j.neunet.2025.107479；Lilith 2607.26099。
- D4/5：Critical-CoT正式 https://aclanthology.org/2026.acl-long.495.pdf；Training-Time Backdoor Sample Filtering via Loss and Uncertainty Dynamics（需从题名核正式来源）；Backdoor Sample Detection through Progressive Perturbation and Loss Trajectory Analysis（同前）；Defending against high-poisoning backdoor attacks by exploiting loss divergence，Information Fusion2026，PII S1566253526001879（需核全文）。

公开API实测：OpenAlex /works?search=backdoor&per-page=1 返回429；Semantic Scholar graph /paper/ARXIV:2401.05566 引用信息返回429。错误JSON保存，不重复高频请求。前向追踪使用官方/作者页及精确题名检索，不能标专业数据库已完整覆盖。

## 重要跨域待核（D3发现，主审核一手身份）

查询精确题名：`Diluting Poisoned Fine-Tuning Data Can Strengthen Its Effect`；`Agentic Backdoors via Pretraining Poisoning`。

- Homeopathic Learning: Diluting poisoned fine-tuning data can strengthen its effect，https://arxiv.org/abs/2611.10110 ；作者公开介绍 https://alignment.anthropic.com/2026/homeopathic-learning/ 。**交D6 A全文主责，D1/D3需整合原文证据后再结束审查**。关联正常数据与条件性行为，不可因跨域而漏查，也不把介绍页当完整论文。
- Agentic Backdoors via Pretraining Poisoning，https://arxiv.org/abs/2610.37741 （2026-10-27），官方代码 https://github.com/safety-research/agentic-backdoor 。已通知D1核证；D2/D3按必要性跨域整合。
- Associative vs. Distributional: Two Regimes of Backdoor Learning in LoRA-Adapted Code-Generation Models，https://www.mdpi.com/2624-800X/6/5/146 ，D3负责。

这些不是核证结论。跨域主责减少重复获取，不容许把别人摘要冒充本agent全文阅读；关键判断回原文定位并经B复核。

## D1尾轮提供的候选（未核全文，待相应A处理）

- D4：Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery 2610.32866；须判断其是否真覆盖backdoor信号，不能仅凭解释性工具相关就纳入。
- D5/D6：DETER: Suppression During Pretraining for Tamper Resistance 2611.15479；区分知识能力抑制与后门。
- D5/D6：Mitigating Backdoors via Decoy Shortcuts and Knowledge Decoupling 2608.00732 / IJCAI2026。
- D5复用：D2已保存BIBAU 2606.14078与Suppression Is Not Erasure 2611.12807的PDF/文本，可回原文读取；下载不等于已读。

## 受限全文复查（2026-11-19）

追加查询：`"10.1016/j.inffus.2026.104221" pdf`；`"A unified optimization framework" "Yihan Shi" backdoor`；精确题名+filetype:pdf。出版方当前可见摘要、引言与section snippets，不是完整正文；DBLP正式记录也标closed。未发现可验证合法OA全文，故维持全文受限。未使用异常非学术镜像结果。不能把可见摘要/引言中的理论保证或平均ASR转为核证结论。等待用户如有机构入口可补查，不阻塞其他域。

## Hibernated补充材料再次查找（2026-11-19）

查询精确题名+appendix / github supplementary / arxiv；发现合法NSF存档 https://par.nsf.gov/servlets/purl/10342700 。web读取超时后公开HTTP下载成功（200 application/pdf），存d2-persistence/sources/hibernated_nsf.pdf及.txt。仍10页，末尾10318为参考文献，正文3处引用appendix但无附录。这里只检查补充完整性，不冒称新增全文阅读或消除限制；不使用未核授权镜像。
