# D4 A 阅读与证据定位日志

2026-11-19。正文及与结论相关附录读完才计FULLTEXT_READ；参考文献列表不等于读完所引论文。以下15项交B审核，非最终认证。

|材料|实际阅读范围|关键定位/边界|
|---|---|---|
|Spectral Signatures 1811.00636v1|正文§1–5 pp1–9、附录A pp12–13/B pp14–16|§2训练后算法；§3谱均值分离条件；§4/附录B模型层选择和VGG|
|AC 1811.03728v1|正文§1–8 pp1–9|§5最后隐藏层/ICA/kmeans/ExRe；§6.3阈值失误；§7解释为假说|
|ABL 2110.11571v3|正文§1–5、附录A/B；B3图形不逐点转录|§3.1–3.2 pp4–6隔离时已学入；A.3训练日程；B.5适应高loss|
|BaDLoss ICLR2025|正文§1–7、附录B–J相关说明/表|Fig2快慢反转；§4参数；§6/D适应边界；G少数类误删；I.2改变ABL基线；J过滤率/架构|
|Rethinking EMNLP2025|正文全部及附录A–G|Table6检测分母；§5.4正文HSOL/Fig3SST2矛盾；附录非单调例外|
|BadWindtunnel ACL2025|正文§1–7、附录A–G|§3受扰模拟训练与置信差；§4/6残余和低污染边界|
|ToxScreen2607.26849v1|正文§1–7、附录A–G|§4.1候选真trigger与位置；§4.2/F .70对Fig2 .80；§5.1排名/通用jailbreak；剪枝/ghost|
|Haystack2602.03085v1|正文§1–8；附录A.1/B/C/D/E相关方法/F/G/H/I|两任务族先验；36/41与5/6，13净模型；A.1改双年份构造；失败候选与功能等价|
|SpecGuard2609.11799v1|正文§1–8；附录证明/实现/失败|Table2 .572；Table8 .92ASR/.557AUROC；尾部稀释、窗口、短回复误报|
|EVO2610.20319v1|正文§1–6 pp1–8|§3/4净参照+解码轨迹；§5 K5/K12延迟触发；不是训练时间轴|
|Early Detection2607.24893v2|正文§1–7|§3任务外部组装；§4/5误报分母；§6离线编码/长度/跨域阈值|
|WSD2602.15195v3|正文§1–7 pp1–9；附录A.1–A.3 pp12–15|§4监督；Table1每家族分割；§5.4层/秩扫描；§6适应攻击；后附分布图不逐点转录|
|Z-PEFT2608.02271v1|正文pp1–7，附录pp9–11|正文测试阳性数选ACC阈值/PEFTGuard3epoch；Table3 Alpaca .635；附录Table7 AdaLoRA .2628；SHAP非因果|
|TimeGuard2605.22365v2|正文§1–6 pp1–9；附录A/B/D说明/F/G.5–G.8/I|Fig3前几epochs趋同；§3.3 FDER .497；F.7适配条件；G.6昂贵embedding未优；I常见motif边界|
|Anthropic probes2024|官方网页Introduction/Methods/Results/Ablations/Why/Related/Future及脚注1–9|末prompt token，CoT先scratchpad例外；footnote5跨模型不稳，6校准未解|

## 跨域局部阅读，非全文

- Persona Vectors2507.21509v3：§4 pp6–7、§6.1 pp9–10、§6.3 pp11–12，直接核均值投影差和各500样本训练验证。
- From Data to Behavior2602.04735：§2–3 pp2–4，Table1，附录C.2 pp15–16，核α扫描和NYC反例。
- Model Organisms2506.11613：§4.1–4.2 pp5–6，核方向转动/grad峰和5×放大，不将其当盲预警。
- Homeopathic2611.10110：H.2 pp42–43，核378prompt/终点二学生/数据量不同/低行为仍方向位移。
- 综述2406.06852v5：§5 pp13–14；综述2502.05224v1：§4.2 pp22–23及§5 pp24–25。仅用于分类和引文扩展。

## 筛查与受限

TopKLoRA/HARVEY摘要引言→D5；DFBScanner摘要引言；代码谱2510.13992v2首页；PSSP作者页摘要/PDF下载；IEEE/SSRN/InfoFusion/RG详见PAPERS。没有将这些项目算作全文。

## 技术说明

原PDF和pdftotext -layout为定位原件，read.txt只便于阅读。TimeGuard的内嵌图字有控制字符导致read.txt额外分页，所有报告页码指印刷页，不能自动按read标注跳页。原文矛盾、比较方法变化及阴性结果已经写入REPORT供B逐项核对。

## B后原A补读（2026-11-19；18项FULLTEXT，新增待B）

- R1：2506.11613 AppF pp19–21（t±s、全轨迹PCA、alignment/coherence阈值稳健及20×退化）、G.1 p22（final comp）。
- R2：2611.10110 H.4/图H.3 pp44–45（Gemma轨迹瞬态及不同LR）。
- R3：2512.09742 §5.1/Fig10 pp8–9、§6 pp11–12、F.1–2 pp54–55、H.1–3 pp65–69相关方法/图46/Table9；2607.09053摘要与§2.3–6 pp3–7。两项是局部阅读，不算全文。
- R4：2507.21509 AppL pp54–56（高投影质量/实用性/额外toxicity过滤）。
- R5/R7：Haystack Table4 p13；ABL Table2 p9；BaDLoss Table4 p23题注/列值。小修定位依B和原文核对。
- R8：TopK §6开头/末段、§7 pp6–9；HARVEY §2–3 pp2–4有关RCE与主动分离。
- SEEP：正文§1–5 pp996–1006，附录A/B pp1009–1010完整读取。新增FULLTEXT_READ。
- ResearchSquare rs-10594749/v1：官方网页metadata、官方PDF正文§1–6 pp1–12、附录A/B p17完整读取。新增FULLTEXT_READ但证据类型是协议+非推断smoke，无确认实验。
- Tensor2605.15183v1：正文§1–4 pp1–9完整，A.2/A.3 pp15–16、B/C/D实验说明pp16–20相关部分读取。FULLTEXT_READ；后续新发现workshop版§2.3/§3.1/Fig1/§4/AppC仅局部比较，不重复计篇。
- OPD2610.07654：§4.2/Fig6 pp6–7及D.3 p24，只计SECTION_READ。

更完整的结论/位置对应见REVISIONS N1–N4与R1–R8。旧“15项”是首轮历史数量，不是当前统计。

### 原A C1聚焦补读，2026-11-19

实读c_sources/zdefense2023正文§1–5/Limitations pp953–961及AppA–C pp965–967；初始输出截断后补页956–961。实读pkad2024正文§1–6 pp5837–5841及AppA–H pp5843–5849。实读nete2509.05318正文§1–8 PDF pp1–31，无附录；图字符抽取损坏，不据图估数，阈值/表取清楚正文。GradSentry v3仅§3–4.1 pp3–5/AppG p15/J.4 p20相关节，D5主读全文。逐篇信息条件和反例见C1_NOTES_A.md；不将C的全文状态继承为A已读，也未把GradSentry升为全文。当前21全文/39条为修订统计，旧18/35为历史节点。
