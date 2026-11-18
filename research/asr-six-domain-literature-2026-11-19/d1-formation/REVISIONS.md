# D1 原 A 对独立 B 第一轮的修订处置

日期：2026-11-19。原A稿保留为 `REPORT_A.md`；本轮更新 `REPORT.md`、`PAPERS.json`、`READING_NOTES.md`、`SEARCH_LOG.md`。以下“采纳”仅指原A已修改或保留相应限定，不表示B已经复核通过。新增来源本域只回读相关正文/附录，全文主责仍属D3/D4/D6；不增加本域FULLTEXT_READ计数。不提出实验或最终新颖性判断。

## 原主张逐项处置

| B编号 | 原A处置与定位 |
|---|---|
| B01 | 采纳：REPORT §2.1改Eq3–4为PDF pp3–4，保留局部SGD一阶近似限制。 |
| B02 | 保留：§2.1仅称若干条件改善，未把分类LFR改成完整生成ASR。 |
| B03 | 保留：§2.1及READING_NOTES保留强正则阻止spam学习、需减小λ的失败。 |
| B04 | 采纳：§2.2明确方向图Fig3、step800，不称全轨迹中介。索引同步。 |
| B05 | 采纳：§2.2/索引/笔记明确底0–5完全未投影β=1，上6–11投影β=0；补IMDB→SST2全投影93.0、分层92.6，拒绝必优表述。 |
| B06 | 保留：冻结BERT/PET、分类、干净再训和失败边界。 |
| B07 | 采纳：§1、§4.4–4.8和§6整表重写纳入新增先例；未做指定整链的判断仅限已核对象。 |
| B08 | 保留：§2.3明确模型替换后移除正常客户端类别；未称首次形成证据。 |
| B09 | 保留：§2.3高poison比例缩短寿命限定于该联邦攻击设置。 |
| B10 | 保留受限：§2.4 Information Fusion无合法全文；不传播理论保证/数值。 |
| B11 | 保留：§3写平均loss与FC/LBA较慢，不外推逐样本。 |
| B12 | 保留：§3隔离不代表尚未形成，数据扩展范围仍明确。 |
| B13 | **纠正**：§3、PAPERS和笔记中60轮均改为仍含poison的full training dataset、常规损失训练；撤回“正常恢复”。 |
| B14 | 采纳：§3改为普遍快学叙述的边界；ITLM训练accuracy不是ABL平均loss的同指标反证。 |
| B15 | 保留：ITLM残余0.1–1.3%、重初始化、GLM边界。 |
| B16 | 保留：CL目标和数据条件不等同，未把统一TRL写成完全同配置。 |
| B17 | 保留：强度非单调与语义/MI只是候选解释。 |
| B18 | **纠正**：§3、索引与笔记写明已有backdoored SimCLR，随后每类50 clean+额外50 target-class poison同时微调encoder/head；不是仅改loss/纯clean。 |
| B19 | 保留：NeuBA额外监督目标、预设输出向量，不冒称自然纯数据机制。 |
| B20 | 保留：后验标签/最佳trigger、距离干预混合属性、分类范围；许可更正未充当实验支持。 |
| B21 | 保留：0.5%、73.3→42.5对应trigger测试对；未写clean准确率。 |
| B22 | 采纳：§4.2补剂量同时影响RM置信和PPO触发暴露；由“瓶颈”改为已测阶段差异、原因未独立定位。 |
| B23 | 保留：224提示、clean RM评分、退化输出和SFT-only边界。 |
| B24 | 保留：Agentic主配置多seed，后训练各阶段，机制解释尚未分离。 |
| B25 | 采纳：§4.3/笔记/索引补预训Pass@32三seed范围1.3–85.0%，不暗示所有seed预训低。 |
| B26 | 采纳：§4.3改为flag-invariant固定命令结构；1.4/.07明确rare-trigger单seed，不是执行成功。 |
| B27 | 保留：DNA完整payload与TATA失败。 |
| B28 | 保留：干净checkpoint输入token替换不是训练梯度干预，局部概率与完整成功分开。 |
| B29 | 保留：后置日程、剂量/时间/后缀正常更新混杂、单轨迹。 |
| B30 | 保留：OPD轨迹、offline KD对比、拒答过滤不等于移除分布监督。 |
| B31 | 保留：OPD固定组内分布、scalar-logit及zero-injection特殊teacher条件。 |
| B32 | 保留受限：正文及源包仍未含承诺Appendix；没有声称完整配置核证。 |
| B33 | 采纳：§5.3与索引改为固定分布最优解/静态预算曲线，避免一般动力系统“平衡态”混称。 |
| B34 | 保留：真实数据只用于估计矩、再采Gaussian；C.4一合成配置。 |
| B35 | 采纳：§5.4首提BAR即解释trigger−control，不是raw ASR，保留人为门槛与n矛盾。 |
| B36 | 保留：课程报告抽样/相同数据矛盾，未归因纯优化器随机性。 |
| B37 | 采纳：§6改“固定完整P/K批对照的后续响应”，避免构造差分更新的误解。 |
| B38 | **纠正**：§1与索引计数是12份全文材料=11篇原始研究/课程+1综述。综述不是原始实验，课程报告不等权。 |

## 跨域新增证据处置

| 意见 | 原A返回原文的范围与整合 |
|---|---|
| X01 Homeopathic clean组成 | 已读正文pp3–11、AppE pp35–37。§4.4与§6纳入固定poison/匹配步数、固定1500步与9000不同类别clean对照；明确9.8→60.5为未投毒类别，原0–4已有攻击/83.5%无patch误判。 |
| X02 Homeopathic内部/理论 | 已读AppH/I pp42–46。§4.4纳入两终点投影但不当时间前兆；毒量不同，ridge假设及解释不全明确。 |
| X03 Steering Distillation | 已读pp3–5、8–11、AppB18–19、N–P27–28。§4.6/§6纳入教师生成源/学生方向干预和固定Adam尺度SGD；非精确loss匹配、位置差异、长训练/MLP例外均保留。 |
| X04 Model Organisms | 已读pp4–7、AppF19–21、G.3 p24。§4.7/§6纳入约180步方向与5倍放大；local cosine用未来checkpoint；fullFT未直接证同样旋转。 |
| X05 Learning Curves | 已读§II–III pp2–6、AppA/B17、Fig27–28 pp24–25。§5.5与§6明确beta权重、各自最优解而非时间step；凸与非凸/有限差分边界。 |
| Inherited分支 | 已读pp3–6、AppH–N14–17；§4.5保留150 poison、仅换2850clean对照；fullKD另含KL；Qwen与Llama/Gemma差异、S不是raw ASR、位置和容量条件均在索引。 |
| Weird分支 | 已读p7/9/12、AppE.5 48–49、F.2 55–56、H65/68–69；§4.5 clean条件化、§4.8内部与外显、§5.5真实训练期held-out跳变。0.83是平台token计数基线、并非二选一随机率。已说明同epoch不同step与自蒸馏未单因素化。 |
| Convergent分支 | 已读pp3–5、9、AppD15/I18；§4.6区分activation与rank1 B方向，cos .04、各位置消融不同效果；不等同自然形成时间。 |
| D4电路分支 | 已读正文pp1–5、AppB7、I–K15–16；§4.8明确一次前向浅层表示形成，而非训练前期；首token/PPL、法语消融反例。保留在表示复用接口，不当首次形成实验。 |
| Mirage反证 | 已读pp3–7、AppA9–10；§4.7纳入循环训练中方向/梯度不稳定对应行为、长度条件；不升级成EM不存在。 |
| Persona Vectors / From Data to Behavior | 未新增独立主张；本轮不为扩数量重复D4/D6所有路线。方向中介与其直接边界已由上述实质相关材料覆盖；其余继续由全域C按问题查漏，不能据未展开断言无额外重合。 |

本轮所有新增记录均为CROSS_DOMAIN_SECTION_CHECKED，并写明全篇主责；Report中不把他域笔记等同本域全文阅读。`SEARCH_LOG.md`保留原A历史检索记录并追加本轮实际回查，未伪造新增外网查询。请原B复核修正及新增实质主张；C查漏尚未由本文件替代。


## B第二轮最小修订（2026-11-19）

本轮仅处理R2-01、R2-07、R2-12；原A重新回读Homeopathic §2–3 pp3–4、Inherited §5.2/Table3 pp5–6、Steering Discussion/Limitations pp10–11。其余范围不变，以下待原B核对新改句后决定能否交C。

| 编号 | 修正与同步位置 |
|---|---|
| R2-01 | REPORT §4.4改为100k**同任务数字续写clean**：其他姓名或无姓名、未加德语指令的原模型生成。聊天题属于测试，撤回训练“普通对话”误写；补无关instruction数据单独不能产生同样效果。READING_NOTES及PAPERS对应记录同步明确。 |
| R2-07 | REPORT §4.5将Llama/Gemma小教师增益严格限定appended（S .886→.910、.771→.805）；补random的明显增益（.030→.190、.098→.537）。READING_NOTES、PAPERS无条件简写同步替换，仍明确no-teacher保留150 poison。 |
| R2-12 | REPORT §4.6、READING_NOTES和PAPERS撤回“方向更分散”；只保留Qwen训练2→10epoch后结果less salient，作者未在此证明方向为何变化。MLP/SGD原有范围限定不变。 |

三份主文件已检索确认不再出现上述无条件/机制化误写；PAPERS可解析。REPORT_A未改动，SHA256仍为6bdbeed7e8a84ef7140a79d64627fdc99910f466acded89a9fffeb28eaeaf8b7。此记录不自称B通过。

## C第一轮独立查漏后的原A补证修订（2026-11-19）

本轮不重写原A/B/C历史。新增`REPORT_B_BEFORE_C.md`为本轮前报告快照，`REPORT_A.md`及`B_REVIEW.md`、`C_AUDIT.md`、`C_INDEPENDENT_SEARCH.md`保持原样。以下均为原A重新读源文件后的处置；尚待原B核验和原C复核，不自签最终通过。

| 审计项 | 原A处置与当前定位 | 保留的条件/反证 |
|---|---|---|
| C01 ICML2024 CNN时间理论缺失 | 采纳。正文pp1–9、AppB条件/噪声时窗、AppD Thm5.4证明、AppE实验全读。新增REPORT §5.6、§2/3衔接及总图/项目比较；新增PAPERS和READING_NOTES C01。 | 实际迭代T_v/T_u，不再一概归静态最优解；二分类/正交/特定CNN/GD/尺度条件；Time=此后持续>95%，改80%某些非单调消失；MNIST0/5与CIFARairplane/bird，比例≠绝对数。 |
| C02 SP24正交/线性框架缺失 | 采纳。正文pp1–13、AppA/B、独立supp全读。新增REPORT §2.5及项目交互行，索引/阅读记录同步。 | 混合≈两阶段是Assumption3.1；风险上界非必要关系；NTK/平方损失/可逆性。1024例平均loss夹角非逐输入函数梯度×整段位移；Composite及Softplus低角高ASR反例；label-specific非唯一角度中介控制。 |
| C03 2610.07654安全OPD漏收 | 采纳。正文pp1–10、AppA–D相关全部读。REPORT §5.2分两篇独立研究及项目比较，索引/阅读记录同步。 | 固定10毒例重复训练近0→高，但跨尺寸1%失败；1%分母1000clean、总1010；固定state梯度不是全参数动力学；Jaccard=固定teacher prefix跳7取10位置top1一致率；Clean ASR不是正常accuracy；clipping可延迟非消除。 |
| ICML2023 Adaptability后向链未闭合 | 采纳并全文纳入。PMLR下载，正文pp1–9、AppA–H pp12–25全读，REPORT §2.6及项目表。 | NW核/Gaussian/渐近风险界；DNN adaptability是假说/前提，不是已证全体性质；表征距离和预测改变的相关测量不等于首次时间或正常抵消中介。 |
| BackdoorBench2022 B.6/B.7与2024扩展未处理 | 采纳。2022main pp1–10和相关supp/原图已读，全文纳入REPORT §3；2024已有PDF补读全部正文与原图、查arXiv源码，受限正文纳入，消除DOWNLOADED_NOT_READ。 | 2022高比例BadNets仍多遗忘，total-poison>clean-poison非总由poison主导；2024明确慢学但高ASR反例，augmentation解释未孤立识别。2024独立supp缺失，未接入表不冒充完整附录；两版同系列非独立复现。 |
| Not All Samples Are Born Equal全文受限 | 采纳受限处置。出版方/官方repo/作者主页和精确题名+DOI公开检索重查，未获合法全文。新增索引受限条、REPORT §2.6和末尾限制。 | 不用摘要定robust-feature机制、不引数值、不把未获得全文当无先例。 |
| 主审补传Homeopathic persona选择规则 | 采纳。回读AppC.2–4 p29，在REPORT §4.4补界限，阅读/索引同步。 | 排除无名control≥10%的4Qwen runs、Gemma无排除属于persona变毒比例支路，不套用German/MNIST。 |
| 主审补传Steering fullFT语义结果 | 采纳。回读§5.2/6.2和AppM，REPORT §4.6补两个fullFT实验区别，阅读/索引同步。 | direct-steered-data可有较弱semantic shift+EAS；经典systemprompt偏好fullFT阴性，不能说全篇只有内部量。 |

本轮计数核对：31条索引；17 FULLTEXT_READ（16原始研究/课程+1综述）、10 CROSS_DOMAIN_SECTION_CHECKED、2 ABSTRACT_ONLY_FULLTEXT_UNAVAILABLE、1 MAIN_TEXT_READ_APPENDIX_MISSING（VLM OPD）、1 MAIN_TEXT_READ_SUPPLEMENT_PARTIAL（2024 Bench）。章节核对、状态/哈希/路径及原A快照检查通过。没有新增实验、模型推理、GPU使用、论文正文编辑或发布。

## C第二轮聚焦补证（2026-11-19）

首轮C01–03及Adaptability/Bench已由B/C关闭，本轮不重跑；新增`REPORT_C1.md`保存本轮前稿。

| C项 | 采纳和修改 | 保留界限 |
|---|---|---|
| C04 外给规则的积累/重置DTMC | 原A独立读2607.06643v1完整15页，核§3–5与原图/公式；REPORT新增§5.7一段及项目表一行，PAPERS/READING_NOTES同步。 | PPI/PRI首达时间是模型规则的条件结果，未由梯度推导；τ/π外给或校准，盲终点/顺序调度/oracle假设；低于ASR阈值可回I_(τ−1)，不等于清除。Table1 μ′I与1/p不符，不搬性能/稳态表数；事件概率≠mean ASR。 |
| C05 原始EM时序接口 | 本域回原v7 §4.7 pp9–11、AppC.5 pp26–28和Fig11/12；REPORT §4.7加入短段，项目表补时序来源，PAPERS记跨域核节。 | 五seed图、每10步；域内先分化不意味着恶意特异前兆，前40步两组logprob可同向，aligned pivotal亦先升。去weightdecay及多epoch结果不支持直接等同典型grokking；coherent条件自由行为与受控logprob分开；非trigger后门同任务。 |
| Honeypot/DBD非阻断补例 | 只在§3链接D5 REVISION_READING_NOTES，不扩大本域阅读/性能结论或FULLTEXT数量。 | 读D5记录不冒充原文全读；完整PLM/浅层probe、特定训练程序的loss含义需区分。 |

本轮交付计数：33条索引，18 FULLTEXT_READ（17原始研究/课程+1综述）、11 CROSS_DOMAIN_SECTION_CHECKED、2摘要全文受限、1VLM OPD缺附录、1Bench2024补充部分。来源四类限制保留。原A快照哈希、JSON唯一ID、全文路径/哈希、新增章节引用及计数核对通过。交原B仅核C04/C05新句，然后由原C定点复核；不自签最终通过，无实验或发布。


主审状态同步（2026-11-19）：原B与同C末轮已闭合，当前状态见C_AUDIT末轮及REPORT。上文待办/交审措辞保留为历史，不改变实际阅读级别。
