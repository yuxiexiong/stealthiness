# D2 原A回应B首轮修订

2026-11-19。原报告保存REPORT_A.md；B_REVIEW.md保持不变。以下为A已实施状态，**不是B验收通过**；请原B复核，再交C。无新实验、GPU、论文或推送。

|编号|B要求|落实及原文证据|状态|
|---|---|---|---|
|R1|Decontam56%方向/阶段/分母|REPORT§5与READING_NOTES P09/B更正、PAPERS同步：p19 Phase2 115有效−15B失败=100，56erased/19rerouted/10partial/15persistent。p4 Phase3另有B37/40清理、A21/22及14/18低于5%；保留阈值及covert reroute限制|已改，待复核|
|R2|Flip-ASR分母、章节、919与方差|REPORT§4、P10笔记/JSON：Eq6分母全真实非目标，clean非目标只在成功分子；AppC.2 pp22–23 Table10 919底图，列中间大95%CI，撤销平滑单调暗示|已改，待复核|
|R3|FAB/epochs/QRA|FAB p8 Alpaca直接冲突是作者解释而非不相关域；Permanence p7/C.3为后续clean5→20epoch通常缩短寿命；Superficial p6/App16 QRA为查询输入扰动。报告、笔记和JSON同步|已改，待复核|
|R4|Sleeper正文成功清理|REPORT§5明确§5.1p22 Fig13/§7.2p36 post-scratchpadSFT，与AppG58–60 safe-code scratchpad+answer分别写；P14补记与JSON同步|已改，待复核|
|R5|CL来源、Ptrojan记号、Persist位置|CL的task-ID/multihead归类明确来自后续PBTO§3.2p3/p7，未定位CL同等声明；Ptrojan正文Eq2与Alg1∇θ不一致且理论未独立重推；Persist74/76对20/20限定random，first34/42对2/2；Hibernated padding仅targeted版|已改，待复核|
|R6|Anticipate/Latent全文|17页Anticipate及14页Regula Sub-rosa正文+相关附录完整阅读，P15/P16逐节记录。展开未来本地数据代理、k/聚合器失败；未来目标正常监督/新head/冻结协议、解冻失败边界，均进REPORT§3.4/§4与JSON|已补，待复核|
|R7|跨域回原文并进正文/重合表|RIPPLe、Agentic、BI-BAU、Suppression、Homeopathic均回原PDF，X01–05列完整阅读责任和本域页码。REPORT§3.4/4/5/6整合。BI-BAU R/Q定义、Suppression合并后fresh rank8与12/50prompt分开；后续SP24/Merging也X06/07|已补，待复核|
|R8|具名线索逐项分诊|SEARCH_LOG逐项保留来源/版本/理由：Cao、BadGCL、DeferBad实质重要而全文纳入；PerDoor/SDBA/Malice范围排除但未称全文/不可得；Merging跨域原文已核；主审追加BAD-BOOM已全文。Hibernated真缺附录保留|已分诊，范围决定待B/C挑战|

## 新增全文改变的判断

- Cao AppC：加入正常数据的安全重训方案可帮助清理（未控制更新数），与Chameleon促进/干扰方向共同限制“正常训练一律保护”。长串后长时间安全训练也可降至25.3%，不能只列高保持。
- BadGCL §6.4p8：首任务正常重访提高ASR；指标只测首任务，非未来类别泛化；已构造优化/白盒权限限制保留。理论未独立重推，假设不降格成自然定理。
- BAD-BOOM §3/AppC：随机二维切片、单seed、小模型范围，不等于真实SFT轨迹的独立因果验证。主表和rho表均有明显失败及utility成本；不能照录“所有后门塌陷/首个持久”措辞。
- DeferBad §4/AppA–D：BN/非BN遮蔽不同，Tiny和多层FT失败、RCS检测、Tiny FP成功，与“所有FT必激活/所有防御无效”相反。
- Agentic普通后训练增ASR、Homeopathic混训先升后降均成为已核证据，而非继续“待核线索”；但不视为本项目完整形成机制的同义证明。

## 一致性检查

PAPERS 32条唯一id；19本域全文、7跨域针对性核证、1附录缺失、3范围排除整篇、2综述局部发现。所有已纳入本地PDF路径与sha256核对；REPORT_A保留初稿，旧过程记录不抹除；当前结论以REPORT及本修订为准。


## 第二轮B：R9–R11有界短修

2026-11-19。首轮R1–R8已由B关闭，不重做19篇检查。本次只修改指定条目，待原B定点验收。

|条目|回源与实施|状态|
|---|---|---|
|R9|Merging pp3–4/Decontam p4：统一为未知原A、已知自造B；差异限定为注入与中和训练协议。REPORT、X07笔记、JSON已同步|已改待定点复核|
|R10|Cao AppA/C p12：20与420样本同epoch同batch而更新数不同；保留方案改善观察，撤销内容独立归因；报告正文/重合表、笔记、JSON同步|已改待定点复核|
|R11 NAD|D5主责全文，本域仅AppJ17–18及Fig13图像：正常数据蒸馏继续可反弹、teacher部分净化、早停；非再投毒/完全删除后再生|已补X08待定点复核|
|R11 DTMC|D1主责全文，本域仅§§4.1–4.4 pp5–8：外给转移、降阈回I_(τ−1)仍积累；报告只加定义界限，不复制推导/性能|已补X09待定点复核|
|定位小修|Anticipate §5.5–5.7；BadGCL分别T5/T10/T15三条轨迹；SP24经验梯度角与定理Jacobian—累计位移不同；Cao长句3/5/7epochs为不同强度而非累加训练|已改待定点复核|

当前索引34条唯一id：19本域全文、9跨域原文定点核证、1缺附录、3范围排除、2综述发现。上面的32条为首轮修订历史快照；REPORT_A保持不变。未新开检索或实验。


## C第一轮：C1–C3有界补证

2026-11-19。原A实读C提供两篇全文及相关补充，未重跑已关闭19篇。R9–R11已关闭，历史“待复核”状态保留为当时记录。

|条目|本次实施与定位|状态|
|---|---|---|
|C1|P21正文I–V、全部supp实读；REPORT§4补普通植入后原clean恢复及content/ViT/提取反例、额外更新数与PCA解释限制，§2/§6同步|待原B核新增、原C复核|
|C2|P22正文1–5、supp6–8实读；REPORT§3.4补早期SBL、AGEM含毒缓存、T6–8及ImageNet10反例、Taylor和有限200epoch界限|待原B核新增、原C复核|
|C3|X10 D3全文责任复用，本域核Souly§4.1–4.2、AppI/Figs5/26；REPORT§3.1/§6加入近同初始ASR不同衰退及模拟语言对齐的边界|待原B核新增、原C复核|

索引37个唯一id：21本域全文、10跨域针对性原文核证、1缺附录、3范围排除、2综述局部发现。REPORT_A未改。无新检索支线、实验、GPU或发表。


主审状态同步（2026-11-19）：原B与同C末轮已闭合，当前状态见C_AUDIT末轮及REPORT。上文待办/交审措辞保留为历史，不改变实际阅读级别。
