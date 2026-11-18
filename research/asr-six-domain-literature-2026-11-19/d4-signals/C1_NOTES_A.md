# 原A对C1的原文补读记录

日期2026-11-19；仅处理C_AUDIT的C1。复用c_sources三篇PDF/text；原A实际读正文和相关附录后写本记录，不把C笔记当原文。REPORT_A不变。无实验、GPU、发布或其他候选扩张。

## Z-defense

来源：c_sources/zdefense2023.pdf/.txt；EMNLP2023 pp953–967。正文§1–5及Limitations pp953–961、附录A–C pp965–967已读。第一次整篇输出被截断后，补按页读pp956–961；pp953–955和附录无漏段。参考文献仅定位，不算所有被引文献已读。

- §3 pp955–956：数据标签先验p(y)替代无偏p(y|a)；unigram及constituency ancestor-path，Stanza parser。词/标签共现统计先于待防御模型训练，精确trigger未知但特征家族预选。
- §4.1 pp956–957：主设20%毒、BERT3epoch；不能把20%视为防御算法已知污染率。§4.2.1阈值18来自preliminary观察；§4.3.3 pp960–961阈值扫描，不是通用FPR保证。
- Table2 p958 vs Table3 p959：AGNews Syntactic Z-SEQ FAR7.2、FRR32.1，终局ASR99.5；数据检出与终局防御明确分开。Table4是benign-on-trigger ASR。基线删除量匹配Z-SEQ。
- §4.3/Table5–7：污染比例、模型和阈值影响；不可将某一SST2/QNLI效果泛化。Limitations p961：benign初始PLM、未测style、经验观察缺一般理论。
- AppA–C：特征变体/先后次序、频率、剩余训练量；并非更复杂特征必好。表11表头/值次序有可疑处，本轮不援引该表数字。

## PKAD

来源：c_sources/pkad2024.pdf/.txt；Findings EMNLP2024。正文§1–6 pp5837–5841、附录A–H pp5843–5849逐页已读。

- §3 pp5838–5840：先用标签的类间Mahalanobis距离选L_first，再以距离差m(x)赋弱标签；L_second区分弱毒/净；LDA迭代，净集合固定。预训练知识不代替任务标签，也不证明任意任务语义可分。
- AppC pp5844–5845：r_p是defender belief，推导先设选入净/毒数量比r=1，再用经验线性和四分之一次幂拟合。不能说无污染率先验/无阈值。AppD p5846默认r_p=.03；主设2%真实污染与首label目标在§4.1。
- Table1主表不等于全面清零；AppE Table4 p5846 BeaverTails Syntactic ASR50%。AppF Tables6–7跨PLM能力/检出不同；AppG Table8高r_p可能增误报，未给未知污染率通用选参保证。
- §6 p5841：clean-label与model-poisoning未解。AppH是任务label空间提示，非对完全未知任务无标签检测。

## NETE

来源：c_sources/nete2509.05318.pdf/.txt，arXiv v1首页2025-08-30；正文§1–8 PDF pp1–31（i–xxxi）全部已读，无独立附录。期刊元数据沿用C已核记录；未比刊版全文。

- §3 p8：无受害模型访问；§4–5 pp9–16：off-the-shelf scoring LM与T5 mask-filling；mask10%/span2；算法1有归一化差异统计。需要可用模型和扰动生成，不需要输入任务label或target。
- §5.3曲率解释依连续近似、零均值单位方差及对称噪声等条件，不把离散文本替换直接视为普遍精确Hessian。Fig2 caption“exceeds→backdoor”与图中箭头、Eq4和Algorithm1相反；本文报告按Eq4/Algorithm1的低分疑毒，不引用该caption作判断方向证明。
- §6.1 p17明确主要指标AUROC。Table5 p18 OLID word .65/YELP sentence .71；Table7 p26 Sleeper .74。AUROC非训练后ASR，非固定阈值FPR。
- §6.3 p23 Threshold analysis：200已知word-level毒样本均值阈值，检测sentence/syntactic/style；段内.60/.86/.84称detection results，未擅自解释成AUROC或FPR。作者“no extra data/prior knowledge”强措辞不能盖过实际已知攻击校准。
- §6.4 pp25–29：temperature/LLM style变体；对抗、越狱输入也可检，所以读数不是已植入权重后门的特异证明。多trigger是输入文本组合，不证明模型多后门训练机制。特殊无现成模型情形改用疑似感染模型，仅post-training。
- §7–8 pp30–31：长文本运行成本与语义变换样本恢复困难。图中文字提取受控制字符影响，未据图估新数字；清晰表/正文支持本轮报告。

## GradSentry局部接口

来源：../d5-defense/sources/C_GradSentry.pdf/.txt，v3，D5全文主读。本域读§3–4.1 pp3–5，AppG p15及J.4 p20相关文本；不提升为全文。

- 训练前每带target response的样本lm_head loss梯度，截前1/8行列、top16 SVD熵；不进行optimizer更新。需要模型梯度与监督目标，未知真实毒标签不等于无监督输入。
- 全数据KDE+Silverman带宽、找两峰间谷；无清楚双峰回退0.7。无需真实污染率输入，但分离假设/经验回退不等于无条件校准。
- AppG optimal阈值图是分析；AppJ.4未测试直接entropy最小化。D4不复制效果数字/测试集N，由D5原A/B核其详细限制。

## 处置

REPORT新增§2.8及类型表一行，§6重合边界相应更新；PAPERS新增3 FULLTEXT与1 SECTION_READ，现39唯一条目/21全文。旧报告保留。原B只核新增方法、信息条件与反例；同C只复核C1闭合，旧稿及其他候选不重跑。未声明全域无遗漏。
