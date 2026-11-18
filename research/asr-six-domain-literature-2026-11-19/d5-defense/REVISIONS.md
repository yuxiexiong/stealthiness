# D5：首轮 B → A 修订处置

日期：2026-11-19。**B 第二轮已核旧九项及必要补链；A 已落实三处小修，待原 B 定点确认；C 尚未进行。不是 D5 完成签署。** 原18篇/36组首轮B核验见`B_REVIEW.md`；本轮新增12篇全文及2篇跨域定点阅读。索引现37项：30 `FULLTEXT_READ`、6 `FULLTEXT_PARTIAL`、1 `ABSTRACT_SCREENED`。部分阅读和来源定位不用于冒充全文覆盖。

保留`REPORT_A.md`未编辑，sha256：`35fa139443042216628f3f6b30ae9f01cd3caee05322db8dfc11608f1feac1a9`。当前对外综合应读`REPORT.md`，不要继续引用首稿错误表述。

## 1. 九项强制修订的实际处置

| B项 | 回到的原文 | 当前处置与位置 |
|---|---|---|
| 1 DETER门槛/归一化 | §3.2 p4、T1 p5 | REPORT D5-18、NOTES DETER、PAPERS均写：终点NO FILTERING PubMed loss；分母FILTERING达同终点token；20.36×/2.09M/.010%保留 |
| 2 DiSec指标 | p7T1、AppD pp12–13 | REPORT D5-10/NOTES/PAPERS将12.20改为AASR，旁列MASR14.69，clean94.67→88.96 |
| 3 Suppression维数 | p16T6 | REPORT D5-12/NOTES/PAPERS明示rank8 adapter上k1投影句法54.2%；k6静态0；未改为rank1 LoRA |
| 4 SharedLatent inductive | p15F6、p14Gemma | REPORT D5-16/首节/NOTES/PAPERS明示仅steering失效、未做该条件CAFT；学习率2e-5与2e-4并列 |
| 5 BI-BAU R/Q | p9定义、p10T2、p18D2、p21T10 | REPORT D5-11、NOTES/PAPERS分权重retuning与query reactivation；12.48/96.72归R；NOTES分别记数据权限与预算 |
| 6 P2P理论 | p13AppA | REPORT D5-06/NOTES：函数类sup可表达性不是优化收敛；α→1/ε→0为假设；映射π不自动排除原攻击目标 |
| 7 MB触发数/效用 | p9T4/Limitations | REPORT D5-07/NOTES：Toxic Qwen1.7B，防御触发4→6，ASR0→.972、AACC.729→.303；并发多攻击触发仍未测 |
| 8 CAFT表定位 | p21AppA5 T1/T2 | REPORT D5-17/NOTES分别归Qwen/Mistral；top5PC .34/90.5与解释方向1.16/92.2为不同折衷 |
| 9 机制证据等级 | 原18篇已核范围及补读理论/消融 | 保留相关、局部因果、理想模型与普适机制的区分；新SP24/Safety/I-BAU均明确理论条件与局限；无“唯一回路/不可恢复删除”升级 |

B指出的其他具体点也已处置：Critical-CoT p5训练示例与p7 unseen所列字串重合写入主报告及笔记，不直接断言数据泄漏；SelfHeal前两阶段主要任务无关、第三阶段任务特定；Dummy输出控制4/16分母与99题主面板、同任务多未知触发实测均补入；Fragmented未定位单元的top60计数疑点撤回，不再用它作负面证据。

## 2. 必要补读链：已完成的新增全文

以下每篇均读正文全部及相关附录，具体页面、信息条件、分母与反例在`READING_NOTES.md`后半，原始新增记录亦保留`REVISION_READING_NOTES.md`。PDF及提取文本在sources；不依赖综述或其他域笔记当作原文。

| 新增 | 解决B缺口 | 对主结论有影响的核验点 |
|---|---|---|
| Honeypot2310.18633 | 文本浅支路分流前身 | 浅probe先学毒不等于完整PLM先学毒；短warmup/阈值会失败；非普适早期低loss毒 |
| DBD2202.03423 | 解耦训练原型 | SSL后低loss作clean筛选，信号含义依流程；可信集仍有毒、标签一致任务能力可掉5pp以上 |
| I-BAU2110.03735v4 | BI-BAU原型、访问/理论 | 5000标注clean主设；100例仅单cell且有损失；强凸/内层解假设非非凸安全保证 |
| Fine-mixing2210.09545v1 | pretrained回混与权重修复 | 需要可信原checkpoint；二值元素替换不是平均；句对任务高残余，无relearning |
| BEEAR2406.17092v1 | 生成LLM embedding对抗修复 | 无trigger不等于无目标先验；代码同源坏标签/17题；不能拿PCA当唯一机制 |
| MuScleLoRA2402.12026v3 | 频率学习偏差是否独立家族 | 升为全文；多尺度低秩+clean梯度是不同操作。LLM µmax0/128clean，单独多尺度可只延迟；实际19物理页 |
| NAD2101.05930v2 | 选择一个最影响主张的传统修复原型 | 完整蒸馏链；继续训练会拟合残余后门，早停有实质作用；不同于再投毒恢复 |
| QES2610.00663v2 | 最新隔离/防形成 | 默认部署持续门控且保留权重；GSM8K test进训练；不能称静态删除和独立数学泛化 |
| Phantom2602.04899v3 | 数据防御/密码/跨模型 | 11项包含control及外部embedding变体，非全部cell失败；密码条件只测部分防御；存在一般能力下降 |
| OPD2610.07654v1 | 安全蒸馏与Lazy防御时间边界 | 长训ASR趋高；clean ASR非utility；top1 teacher-forced指标不是onpolicy topk重叠 |
| SP24 Orthogonality | BI-BAU理论前身、实际防御矩阵 | Table1混引用与新测；低角度亦有修复成功；上界大不能反推实际风险大 |
| Safety-Efficacy2602.00822v1 | 拟合/防毒正则权衡理论 | 已可取得并全文读，不再受限于abs打开失败；B第二轮已核一般流收缩推导的反例与适用条件；450epoch反例改写早期免疫理解 |

## 3. 跨域独立定点核验

- **BAD-BOOM**：D2已主读；本域独立读pp3–4/6–9/15–17的相关耐久段。主报告区分routine clean SFT与专门防御；有高原始基线、残余utility损失、单seed及目标字符串指标。该篇不再互相假设对方尚未读取，也不计D5全文数。
- **Cordyceps2605.26595v2**：D3主读；本域防御位于§5.3而非转交所称§5.5。独立核pp9–11/19–20；CROW/SecAlign按50例推理长度90%早停，实际迭代远短于计划epoch。低防效不能泛称完整默认训练失败；CA是隐蔽泄露成功，非clean准确率。
- **Phantom**：D5已主读，D6可回核密码条件/跨模型以及同构steering部分；未把D6已读视为本域替代证据。

## 4. 有据排除／保留外围

| 线索 | 已实查范围 | 本轮不升级全文的具体理由 |
|---|---|---|
| ANP2110.14430 | 官方NeurIPS摘要与PDFpp1–2 | 传统参数敏感性剪枝仅上下文；NAD已完成与本报告时间/耐久主张更直接的原型全文。不引用ANP原篇性能，SP24数字只归SP24重测 |
| T&R2022 | 官方标题/作者/摘要及PDFpp1–2 | 重构stem保护+重训head，不混同2026TR。主动分配/分流已经全文Honeypot/DBD/TR锚定，不增加原篇性能与因果保证 |
| CBD2303.06818 | 官方CVPR2023/arXiv摘要身份 | 因果启发表示解耦，非同缩写certified detector。广义训练解耦已有全文原型，不因causal命名增加因果机制结论 |
| GARD2611.14143 | 官方HTML§§1–4与5.1 | 表示空间局部/全局对抗修复与I-BAU具体实现不同，但不改变“代理扰动修复已有/不等于普适删除”结论；不独立引入其性能/恢复主张。不是因为访问不了而停止 |
| PersonaVectors2507.21509v3 | D6主读，本域未重读 | 本报告既不引用其具体结果，也不称原CAFT唯一，因此无需机械重复D6全文 |

不宣称穷尽传统剪枝、联邦、多模态或所有最新变体。当前核心补读没有PDF获取阻碍；受限点是论文自身评价/描述边界及B已核的具体推导限制，不是以抽象摘要代替可得关键正文。

## 5. 首轮修订交 B 的历史入口（第二轮已完成）

先检查九项更正在当前REPORT/NOTES/PAPERS中一致；再核新增12篇各行的方法—权限—反例—可支持结论。优先回源：QES AppB3测试集混入；Phantom Table31及K5与主文的分类器预算差异；MuScleLoRA AppB4的µmax0；NAD AppJ晚训回升；OPD D3指标与T4时间边界；Safety A5/A6理论条件。BAD-BOOM/Cordyceps只核D5引用的定点段即可。B之后由C独立查漏，仍可能返回A补证。

## 6. 交接前文件核验

已检查37个稳定ID无重复、必需字段齐全、本地引用路径存在、所有已有sha256与PDF一致；30个REPORT编号与30篇全文标记一致。新增PDF物理页数已用pdfinfo校核；MuScleLoRA为19页。REPORT_A校验值保持不变。仅更改本D5目录，临时写稿脚本已清理。

## 7. B 第二轮三处小修（2026-11-19）

依据`B_REVIEW.md` §§6.3–6.4定点编辑，无新检索、无重读旧文、无计算实验。

1. MuScleLoRA：REPORT D5-21、两份新增笔记与PAPERS均改为“µmax=0时，传入优化器的ĝ是clean辅助梯度的非负倍数，混合数据梯度决定倍数”；不推断实际Δθ逐步共线。模型名同步为Llama2-7B。
2. NAD：REPORT覆盖限制、两份笔记及ANP索引明确NAD是attention蒸馏修复原型；ANP敏感性剪枝仍仅范围定位，不新增全文或性能覆盖主张。
3. Safety：REPORT D5-30、两份笔记与PAPERS分开正文κ/固定G线性式和AppA5长度尺度ℓ；原“待B核”改为B已核具体凭据。A5反例只否定所指残差/逐平滑因子步骤，不宣称该例推翻df下降；A6可实现反例说明现有普遍方向收缩证明不足，不否定总trace可能下降或经验效果。数学凭据归B_REVIEW §6.4。

另按B非阻断精度意见，将Honeypot笔记中的ABL隔离范围2–15%改为2.1–15.6%。旧九项/新增12篇与跨域段无需重跑，待同一B仅核以上实际落点后交C。REPORT_A仍未编辑，全文/部分/摘要计数仍30/6/1。


## 8. C_AUDIT §5 三项缺口：A 独立全文补读（2026-11-19）

B_REVIEW §7已确认上一轮三处小修，旧30篇闭合。本轮只使用C已缓存公开原文，新增三篇正文及相关附录实读，不新增候选、不重读或重跑旧30篇。原有条目的改动仅同步B已闭合状态；REPORT_A不变。

| C缺口 | A实际阅读与新纳入 | 采纳及保留边界 |
|---|---|---|
| CROW | 官方20页；正文1–9、AppA–C15–20；REPORT D5-31 | 补100clean内部一致性修复；原文Table1非全<5%；原拒绝α11/5epochs与NEEDLEα5.5/60更新、Cordyceps长度早停分列。提高α的效用损失、理论线性化条件、无再学习验证均保留 |
| NEEDLE | v1 28页；正文1–10、AppA–E15–28；REPORT D5-32 | 已知trigger权限；拒绝子空间保护与漂移补偿；Table S6是无保护/无补偿naive控制，非完整方法。749安全面板以有效分类标签为分母，200触发测试另列；正常能力/安全拒绝/KL不可互换 |
| GradSentry | v3 22页；正文1–9、AppA–K12–22；REPORT D5-33；p13 Table4像素核验 | 离线逐样本梯度过滤后训练≠在线预警/权重清除。Table4 WebQA400/Freebase2000/CoQA2000/NQ498；不从作者0.00%推0/N。clean-only保留率非ACC；oracle阈值与自动阈值、间接适应与未测直接熵优化分开 |

两份阅读笔记同步完整条件/反证；PAPERS新增稳定ID与本地PDF校验值；SEARCH_LOG记录只复用C缓存、无新检索。本轮计数为**33全文/6部分/1摘要，共40稳定ID**。原B仅需核新增三篇及对应综合句，再请同C定点确认三项缺口处置；A不自行签署B/C或六域完成。无实验、GPU、作者代码复现、论文编辑或发布。

交接核验：40稳定ID无重复、33全文/6部分/1摘要与REPORT 01–33一致；新增三PDF路径、SHA256及20/28/22物理页一致；Table4原图已保存。REPORT_A的SHA256仍为35fa139443042216628f3f6b30ae9f01cd3caee05322db8dfc11608f1feac1a9。


## 9. B §8 新增三篇的两项短修与表号（2026-11-19）

依B已核原文凭据定点修改，无新增阅读、检索或候选。

1. **GradSentry**：REPORT D5-33、两份notes AppJ段及PAPERS 2605.26574统一为“一种组合式Gradient Dilution攻击的两个λ设置（.5/.7）”。context augmentation与output blending是同一攻击的组成部分，未有各自独立对照；保留未测直接entropy优化、Table9 recall100/ASR0及部分ACC下降的边界。
2. **NEEDLE约束**：两份notes及PAPERS 2610.00348明确，b完全落入拒绝子空间时，冲突还需原bᵀW≠0；原投影为零则两约束兼容，但u=0时Eq6的0/0公式不适用。不将退化条件扩成方法整体错误。
3. **表号**：REPORT综合引用/D5-32及两份notes的安全附表统一为Table S14，索引同步。其他论文的Table14不变。

B §8已核新增三篇的主要实质判断；现仅待原B核上述落点，再交同C定点复核。33全文/6部分/1摘要与40稳定ID不变；REPORT_A不变，无实验、GPU或发布。


主审状态同步（2026-11-19）：原B与同C末轮已闭合，当前状态见C_AUDIT末轮及REPORT。上文待办/交审措辞保留为历史，不改变实际阅读级别。
