# D6 原 A 对 B 的修订处置

2026-11-19。状态：本轮 A 完成，**待原 B 回原文复核新增/修改，随后 C 独立查漏**。REPORT_A.md 未动；B_REVIEW.md 未动。无实验/GPU/论文正文/发布。

| B 要求 | 原 A 处置与证据 | 落盘位置 |
|---|---|---|
| B1 Inherited 家族概括 | 采纳；回读§5.2 pp5–6、AppJ/K p15。Llama/Gemma teacher贡献小只限appended；random S增益+.160/+.439。Qwen依赖限appended，random无成功转移。 | REPORT D6-03b |
| B2 Model Organisms 在线信息 | 采纳；回读AppF p19、G.1 p22。单adapter localcos用t−s/t/t+s，至少延迟；PCA用全部checkpoint。保留真实5×早adapter干预，不用信息条件否定干预。 | REPORT D6-04限制 |
| B3 Homeopathic persona筛run | 采纳；回读C.3–4 pp29–31。无名prompt persona≥10%排除；FigC.1 Qwen4/Gemma0；TableC.2部分5/6；不推广到German。220judge-positive中55.5%人工连贯分母保留。 | REPORT D6-01e；PRIORITY |
| B4 同文fullFT语义阳性 | 采纳；回读§6.2p8、AppMpp26–27/Fig18。经典systemprompt动物阴性，direct-steered-data fullFT语义阳性，3epochs/LR1e−5；与随机向量EAS区别。 | REPORT首表/D6-02c；PRIORITY |
| B5 MNIST泛化不能抹去 | 采纳；回读AppEpp35–36。源0–4条件化与未投毒5/6/8/9扩张分开；保留9.8→60.5、其他clean6.2、20runs、2000正确非源样本及固定1500steps。 | REPORT D6-01b；PRIORITY |
| 原始EM必补 | 完成正文全读及相关附录。补有条件广坏、形成轨迹、格式/多样性控制与非grokking反例；记录版本/阈值口径差。 | D6-16；CORE_FULLTEXT_NOTES |
| Persona Features必补 | 完成正文全读及相关附录，含v2附J。补directCE干预、known-feature选择、数据比例而非时间的早期信号、重采分母及再对齐残留。 | D6-17；CORE_FULLTEXT_NOTES |
| 原始Cloud正式研究必补 | 完成Nature正文/Methods/全部ExtendedData，图像核表。正式题名为hidden signals in data，B初筛题名更正；arxiv2507.14805合并同研究。补不同初始化BC可恢复、SGD阳性、小更新定理边界。 | D6-18；CORE_FULLTEXT_NOTES |
| Beyond Owls必补 | 完成正文+A–E。随机新能力、条件后门、棋局rewardhack；补soft/hard、主例选择、单vector OOD、参数化阴性等。 | D6-19；CORE_FULLTEXT_NOTES |
| Towards Understanding必补 | 完成正文+A–T相关方法结果。divergence mask/早层训练干预、换prompt/混clean反例、直觉而非定理、EM过滤条件。 | D6-20；CORE_FULLTEXT_NOTES |
| LLS必补 | 完成正文+A–C。选择现成偏好对+DPO而非普通SFT；理论假设、selectedpairs logodds、跨模型及取峰值/问题边界。 | D6-21；CORE_FULLTEXT_NOTES |
| Phantom | 按主审分工D5主读全文；D6回§3–4、4.3与H核crossfamily和向量构造失败界限。未重复计D6全文。 | REPORT§3；PAPERS CROSS_DOMAIN_SECTIONS_READ |
| TensorSimilarity | D4主读全文；D6回§3.3/附B核单层bilinear、SVHN、pre/post区块与无leadtime。 | 同上 |
| D1 CNN/S&P理论 | D1主读全文；D6回正文条件/更新引理及SP相关附录。不能说后门无动态理论，也不能把NTK/正交假设当LM事实。 | 同上 |
| 2405.12755一般grokking | 按B允许范围限缩为外围背景，不以未读证明空白；若C发现改变核心主张再补。 | PAPERS范围排除；REPORT§3 |

## 本轮交回 B 的核验焦点

新增6篇共D6-16–21所有实质判断；上述B1–B5；跨域4条定点接缝（不是重新全读已审15篇）。所有核心原文已取得且读完，不列为访问受限。未声称OpenAlex/SemanticScholar全覆盖，亦未称六领域总体查重完成。
