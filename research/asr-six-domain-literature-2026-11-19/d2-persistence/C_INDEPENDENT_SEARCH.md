# D2 C 独立检索冻结（2026-11-19）

顺序保证：截至本文件首次写入，仅读 PROTOCOL 协议/域范围；未读 A/B、PAPERS、共享索引或 sources。协议末尾项目参照未用于检索。以下为独立搜索候选，不代表全文核实。

## 实际检索路径
公开 web 入口；日期 2026-11-19。查询批次：
1. `backdoor persistence forgetting dormant reactivation fine tuning survey sleeper agents 2025 2026`；`backdoor persistence continual learning latent backdoor reactivation fine tuning 2024 2025`。
2. `"backdoor" "reactivation" "2025" "2026" paper`；`"backdoor" "recovery" "unlearning" dormant survey`；`"Backdoor Decontamination Dynamics" arxiv`；`"backdoor" persistence survey Neurotoxin Chameleon Sleeper`。
3. `"Backdoor Reactivation" clean data Zhu 2024`；`"Unlearn to Relearn Backdoors" arxiv`；`"Persistence of Backdoor-Based Watermarks" pdf`；`"Latent Backdoor Attacks" fine tuning Yao 2019`。
4. `"Sleeper Agents" "backdoor" persistence 2026 finetuning survey`；`"Neurotoxin" "Chameleon" "persistent" 2025 backdoor`；`"Backdoor Reactivation with Clean Data"`。

已打开一手页面：2610.07510v1、2608.11295v1、2512.14741v1 HTML，PBTO PMLR 页面。后向链种子为 PersistBD、Decontamination；前向链从 Sleeper Agents、Neurotoxin、Chameleon 搜至2026新作。综述入口包括 Backdoor Learning: A Survey (2007.08745)、Backdoor Attack and Defense on Deep Learning: A Survey（作者 cubernet TCSS24 PDF），Towards Reliable Forgetting: A Survey on Machine Unlearning Verification (10.1145/3807451)。综述仅作线索，非一手结论。

## 冻结候选及理由
- 2610.07510 Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training（2026）：SFT→RL残留/放大，强度与梯度兼容性，最新直接相关。
- 2608.11295 Backdoor Decontamination Dynamics in LLM Agents（2026）：清除/保持/改道、识别与执行分离，可能关键反证。
- 2512.14741 Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs：多阶段保持、梯度对齐。
- PMLR v306 huang26n / PBTO（2026）：CIL 对旧持久攻击稳定神经元假设的反证，https://proceedings.mlr.press/v306/huang26n.html。
- 2409.13864 Persistent Backdoor Attacks in Continual Learning（USENIX2025）：blind/latent task 持久性。
- 2024.naacl-long.276 Stealthy and Persistent Unalignment：再对齐与激活模式。
- 2407.14738 Flatness-aware Sequential Learning Generates Resilient Backdoors：景观、遗忘、CL抗清除。
- 2401.05566 Sleeper Agents：安全训练保持及隐藏。
- 2206.10341 Neurotoxin；Chameleon（题名待核）：联邦干净更新下寿命机制。
- 1905.10447 / CCS2019 Latent Backdoor Attacks：迁移中补全，与重新出现区分。
- 2411.14449 Unlearn to Relearn Backdoors / DeferBad：已有触发也低行为，后续良性更新再激活。
- 2405.16134 Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack：防御后隐存、推理触发微扰恢复。
- 2501.02704 / 10.1109/TNNLS.2025.3565170 Persistence of Backdoor-based Watermarks：水印异名体系；原训练数据无触发恢复。
- 2510.07192 Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples：继续干净预训练保持差异（前向线索，待一手核）。

待范围判断的最新线索：DormMark 10.1145/3817059；LURA 10.1109/TCE.2026.3735776（仅聚合元数据）；Sleeper Cell 2603.03371；Microsoft Lazarus 新闻（仅即时植入、并非保持证据）；"Backdoor Reactivation with Clean Data"只有二手题录，未确认身份；SNN fine-pruning 10.3390/electronics15050937。不得以这些摘要直接补结论。

冻结后才与 A/B 比对，重要差异补全文/附录；同类补例不无限扩展。


## 冻结后比对及定点追链（2026-11-19）
已读当前REPORT、PAPERS、B_REVIEW（含R9–R11关闭）与REVISIONS。主干候选除watermark/SBL外已涵盖；不重复全文。2510.07192在D3已FULLTEXT_READ，建议D2短接缝而非第三份全文。

本轮新全文：
- arXiv 2501.02704v3（2025-05-09），公开PDF包含正文12页+Supplementary Information8页；下载 https://arxiv.org/pdf/2501.02704 ，`c_sources/watermark2501.02704.pdf`；逐页正文§I–V全部，补充§I–VI全部读取。物理pp8/17/18/19原图渲染核证。
- ECCV2024 SBL：正文pp1–14，引用pp15–19；https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11789.pdf 。官方11789-supp.pdf实际包含19页正文/引用+10页补充；补充§6–8全部核读（物理pp20–29）。保存`c_sources/sbl2407.14738.pdf`与`sbl_supp.pdf`及文本。

新增查询：`"Flatness-aware Sequential Learning" supplementary`；`"Persistence of Backdoor-based Watermarks" "restoration" 2026`；`"Watermarking for Model Ownership Verification" DormMark pdf`；`"Exploiting the Recovery Trap" backdoor pdf`；`"Flatness-aware Sequential Learning" "2025" "2026" backdoor`；`"Persistence of Backdoor-based Watermarks" restoration clean 2026 DormMark`。前向补查返回DormMark；SBL公开前向入口未找到改变本轮修正的新实证。后向核水印§II及refs [8,9,10,20–23,39]：EWE/APP/REFIT等为该比较的祖先，已通过原始比较实验纳入条件性讨论，不继续扩成水印攻击目录；SBL§2及supp§6核CL、NAD、FT-SAM祖先，不把综述说明当这些原论文已全读。

DormMark DOI10.1145/3817059：官方doi搜索返回方法/§5.6全文片段，直接dl.acm.org入口403；本轮未全读，不引用性能。其公开方法描述为专门embedding→masking→fine-tuning，属于本域已有DeferBad等构造性休眠范畴；可记录范围分诊，不作新因果主张。LURA DOI10.1109/TCE.2026.3735776：仅聚合元数据，doi无法访问，无法确认其完整证据；保留来源限制，不因题名推断额外机制。SleeperCell2603.03371的arXiv元数据确认日期/触发工具代理植入，未全文；此轮不扩代理攻击目录，不援引其性能。SNN fine-pruning、Lazarus、检索只有二手题名的Backdoor Reactivation with Clean Data不承担结论；普通clean训练恢复已由实际全文水印文献补入。
