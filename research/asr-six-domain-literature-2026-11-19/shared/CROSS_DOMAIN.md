# 跨域接缝待办

仅用于A/B整合；C须先完成自己的独立发现阶段再读取。不是已核证结论清单。

| 材料 | 全文主责 | 需整合到 | 当前动作 |
|---|---|---|---|
| Homeopathic Learning 2611.10110 | D6 A | D1/D2/D3/D4 | D6完成全文后，D1/D3 A回相关原文修订；D2/D4避免误把终点内部位移当已验证前兆 |
| Agentic Backdoors 2610.37741 | D1 A | D2/D3 | D1已正文及附录A/B读完；其他域核其真正指标与训练阶段边界 |
| Chameleon 2304.12961 | D2 A | D1/D5/D6 | 已有正常类别移除干预；D1已相关节复核，其余必要时回原文 |
| Learning Curves 2106.07214 | D3 A | D1/D6 | D1需区分引用别域全文与自己相关节核读 |
| BIBAU2606.14078 / Suppression Is Not Erasure2611.12807 | D2获取，D5拟主责 | D2/D5 | D5启动时通知可复用来源；重要结论仍须原文核验 |
| Token-Level Generalization2605.30189 | D3 A待补 | D3/D4 | 已下载45页，A补查后再交B最终复核，不因下载标已读 |
| Inherited or Learned?2611.00748 | D6 A待筛 | D1/D3/D6 | 先核身份及重要性，若重要须全文纳入 |

D6阅读中发现Homeopathic H.2（pp42–43）涉及无德语回答却有层内投影差，属两个终点学生的相关性比较；H.4（pp44–45）涉及全参训练出现后消退/LoRA不同，学习率未匹配。待D6正式报告后再据原文形成结论，当前只是接缝提示。

进一步交界（D6 A已读、尚待各域B）：
- `d6-generalization/PRIORITY_EVIDENCE.md`：Homeopathic与Subliminal Learning Is Steering Vector Distillation 2606.00995v4的完整阅读定位。D1/D3需整合clean组成干预；D2需整合fullFT短期出现后衰退；D4需分开终点相关与方向介导干预。
- Model Organisms for Emergent Misalignment 2506.11613：D6报告约180步方向变化、较晚行为和adapter放大提前表达的干预。D1/D4需检视这是既有“内部变化先于行为”直接近邻；最终checkpoint比较不是在线预警。待D6正式记录后回原文。
- Inherited or Learned 2611.00748：D6已全文，D1/D3比较clean响应组成改变选择性。勿只凭题名当单纯蒸馏来源分析。

## B/C实查后的必须同步项

- D1 B已修正Homeopathic clean数据为同任务数字续写而非普通对话；Steering长训练仅结果less salient，不能说方向更分散；Inherited Llama/Gemma弱教师贡献仅appended，random增益S+.160/+.439。D6-03b仍需同步最后一项限定，已通知原A，交D6 B复核。D3现已明确末尾位置，无同一泛化错误。
- D1 C第一轮已确认新增关键全文：ICML2024 Li/Liu CNN理论(PMLR li24at)含学习时间与normal/poison相反更新；SP2024 Orthogonality/Linearity含有条件两阶段/NTK保持论证；2610.07654安全OPD含低ASR到上升与损失变体。原文位于d1-formation/c_sources，待D1原A/B修订。SP24已交D2原A核相关接缝，D5/D6也应按相关性处理。

- D4 B回原文发现Model Organisms附F p19局部cosine用t-s与t+s，PCA堆全轨迹：单adapter也非严格当时在线可得（需延迟/回溯），不能只限制多adapter最终参照；D6 B应核并同步。Homeopathic H.4的fullFT先出现后归零，是训练终点评估漏瞬态的直接近邻，D4 A将补；LR差异不支持普遍参数化因果。

- D6 B新增条件：Homeopathic附C.4中，FigC.1变毒比例persona排除4个Qwen runs（control≥10%），TableC.2/FigC.2另有筛除，不可称全论文共4；不泛化到German主实验。Steering附M直接steered-data的fullFT有较弱semantic shift，需与经典system-prompt偏好阴性分开。
- D6 B要求原始Cloud Subliminal Learning、EM、Persona Features及关键subliminal机制链补读；Phantom Transfer2602.04899由D5待主读、D6核相关边界（尚未完成，不能互认已读）。
- 2605.15183 When Are Two Networks the Same?已明确分给D4原A核全文相关性，不能据摘要当在线预警。D4也补SEEP2024.tacl-1.55；ResearchSquare precursor候选仍需权威身份核实。
- D5 B纠正：SharedLatent inductive附F.6失败实测是特征steering，非inductive CAFT训练；Suppression k=1为rank8上的投影；BI-BAU R-ASR(retuning)与Q-ASR(query)必须分开，已通知D2 A。

- D6原A补读原始EM2502.17424v7：§4.7 pp9–11已有Qwen2.5-Coder32B每10步/5seed的形成时序、ID安全/不安全分化早于广泛misalignment，AppC5 pivotal-token概率亦有轨迹。D1/D4最后整合时核D6正式修订与原文，不将训练时序本身称新；不等同已建立一般后门预警。
- D3 C首轮确认三项缺口：Carlini/Terzis ICLR2022对比学习、Schwarzschild ICML2021统一benchmark、Cordyceps2605.26595v2。D3原A待补后原B/C复核；Cordyceps已给D5作有界防御接口筛选，D4后续C可按信号/检测需要处置。

- D5原A补链已全文读Honeypot/DBD/NAD等：Honeypot完整PLM中clean/poison学习顺序可反于ABL（浅层probe不可混）；DBD低loss可信clean与ABL低loss毒条件不同。NAD原篇已有延长distillation后ASR回升/早停证据，D2后续B/C应检查保持/回升接口。详见D5 REVISION_READING_NOTES，待其B复核，不直接当终审结论。
