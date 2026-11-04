> 在线阅读副本：仅转换链接，正文与判断保持原样。[查看原始文件](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/originals/drafts/evidence/mechanism_claims.md)。

# 机制篇：论断—原始证据索引

核对日期：2026-11-05。对应稿件：`working/mechanism_draft.md`。本索引仅为既有数据审计，不新增训练或模型推理。所有路径以 `/Users/ruizhixu/Documents/ChatGPT/stealthiness/` 为仓库根；下方链接指向实际本地来源。主摘要用于导航，核心 M1–M3、XO、S3、DW8 的数值已直接读取原始 JSON 或既有逐题重算件；跨模型条目另核 inventory.csv。原代码合同是文献材料，不是本轮实验授权。

## E-M01　LLaVA 快速增长、非单调与首次跨越更正

- 论断：P-1.0-D2 第 290—295 步 ASR 从45.5%到84.5%，291→292增加15pp；原 I1001/O1001 第330步已经65.5%，350步回落21%，故首次与持续跨越分开。
- [完整数字与逐题数据导航](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/ASR%E5%85%A8%E7%BA%BF%E7%A0%94%E7%A9%B6%E6%80%BB%E7%BB%93.md#L78)。本轮依据已于10/27逐题核对的归档总结，不称11/05重新计数了全部早期行为文件。
- [PS 原始结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/ps/results_ps.json) 与 [PS 报告](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/ps/RESULTS_PS.md)。相关字段为逐臂 `t50`、ASR 曲线与保真结果。
- 范围：固定 LLaVA、目标 violin、200 个探针；不把5步窗口称为全部训练成本，不把 t50=358/360 沿用为已补测后的首次跨越。

## E-M02　XO 交换、累计暴露预测与训练/测试差异

- [XO 原始结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/xo/results_xo.json)：`runs[*].init/order/t50` 是论文表1八行；`verdict.mae_order=12.5`、`verdict.mae_init=62.5`；`fingerprint.*.corr_order_seed` 范围0.99742—0.99900，`best` 均匹配顺序。8次新交叉臂，不增加其原参照为新复制。
- [累计暴露预测原始结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/r2/results_r2_grok.json)：`t50.mae_K=61.375`、`mae_const=22.5`、`n=8`；这里为当时原冻结20步读数，不能替换成随后密窗的 t50 后仍沿用相同误差。
- [LP 原始结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/lp/results_lp.json)：`LP1/LP2/LP2_by_offset`；支持起飞前已见毒样本有优势但绝大多数未翻转。
- [参照更正及边界](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/ASR%E5%85%A8%E7%BA%BF%E7%A0%94%E7%A9%B6%E6%80%BB%E7%BB%93.md#L152)：补正首次跨越后初始化预测误差70步；主结论不变。
- 边界：只支持固定LLaVA、1%毒集的顺序主导。I因素联合控制LoRA A初始化和dropout随机流，未单独拆分，不能宣称纯初始化效应。累计计数未胜常数不等于排除所有累积状态模型。

## E-M03　跨模型窗口与独立同点确认

- [统一清单](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/cross_model/results/20261010_uncertainty_followup_final/inventory.csv)：共768行是视图×判据组合；筛选 `view=primary`、`criterion=budget_1pct` 后按模型/seed解释，不能计768次实验。
- 字段：`first_start/first_end/first_gain_pp/minimum_observed_updates/largest_later_drop_pp/verification_level/not_an_additional_seed`。
- [完整分组与逐seed解释](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/ASR%E5%85%A8%E7%BA%BF%E7%A0%94%E7%A9%B6%E6%80%BB%E7%BB%93.md#L247)：Qwen3-8B 10条W50上界2—7，seed1009最大后续回落96.667pp；新增1006—1010五条140探针同点5/5；Llama 140同点4/5；Vim旧严格重放失败保留；T5为已测点未观测。
- [V2 定义](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/cross_model/V2_MEASUREMENT_DESIGN.md)：完整注册预算1%/1.6%，不能换成补窗运行长度。
- 级别：现象背景；协议异质，不为主LLaVA机制跨模型验证。

## E-M04　毒骨架固定、干净填充变化

- [现存输出 CPU 重算](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/recomputed_recent_readouts.json)：`rows` 中 `SF-*` 的 `t_sustained`，分别368.5385、402.8729、356.7717、409.6832、405.4740、306.3043；`gr0.agree` 为LP/生成一致性。
- [构造和读数实现](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/cl_code/sf.py)。
- [父轨迹参照](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/pl_code/refs_tm0.json)。
- 级别：真实训练序列干预；同数据多重集、毒骨架固定。父时刻线性份额不是解释方差或梯度份额。

## E-M05　优化器与局部移位，保留不可判

- [实际重置实现](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/b3_code/b3.py)：清一阶矩、二阶矩和Adam内部step，不能只归因β1动量。
- [B3 保真闸门](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/b3/gate.json)。
- [逐臂重算表](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/recomputed_recent_readouts.json)：`B3-ID/CLR/B3D-KICK` 的 `t_sustained`=328.0695/346.4516/328.0460；PL和NF各臂相对相应ID差值支持正文。
- [PL 判定和动作](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/pl_code/pl.py)：原QUIET≤5与MOVED≥10；E30=5.9628不满足QUIET，不事后改为6。
- [NF 随机门槛](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/nf/gate.json)：四随机样本标准差11.3912，阈22.9121是操作规则，不是校准95%总体置信界。
- 限制：全部相关臂第600步已有约97.5%—99% ASR；这是时序响应，非持续防御。B3正式未达SUPPORT=54，不可判。

## E-M06　M1 留出探针模块定位

- [冻结合同](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m1/CONTRACT_M1.md)：SELECT/EVAL各100、r1/r0、k阶梯、F≥.8/E≤.2、B_task差≥.2、LoRA门消融的完整定义。
- [完整主结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m1/results_m1.json)：`tier=定位`、`type=sufficient`、`GM1_1=17.18113674`；`candidates.C.ev`的F/E=1.081366878/.023151982；`task_ev/norm_ev`支持表2；`random_F_ev`20值；`layer_E[24]=.08149631`；`S_star_mlp_share=.5`、`S_star_layer16plus_share=.8125`。
- [330与1250原始分项](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m1/raw.json)：`s330.choice.C=r1_k4`、`s330.cand.C.set`=L24四个attention模块，`ev.F=.968384742/ev.E=.126039592`。四模块全包含于终态S*，所以Jaccard=.125不表示完全换电路。
- 残差场字段：`residual.noising/denoising`；层24trigger9=.968217，层28trigger9=.027641、last=.907382。
- 严格边界：D是目标对最大竞争词margin的触发/干净差，未等同生成ASR、正常能力不损或后门专属模块；无训练期必要性结论。

## E-M07　M2 承载阳性与计时失败

- [冻结合同](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m2/CONTRACT_M2.md)。
- [原始判档](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m2/results_m2.json)：`M2_0.tau_full/e_full/tol_full`支持表3，`tier=D 不计时`；`M2_1.per_curve.Y.tier=不可判`、Z不同步，4个random.P2均false；`M2_2.s=1.417580615`，CI=[1.348371,1.486975]，`held_out_s`=1.462861/1.529013。
- 边界：终态所选S*在起飞时有效，不等于顺序效应已定位到S*。参考自身的计时误差约0不计独立预测。CI是固定模型探针bootstrap，不是跨种子CI。

## E-M08　M3 校准、带符号贡献及预测前提失败

- [冻结合同](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/m_code/m3/CONTRACT_M3.md)：2.2/2.3逐行梯度、动量、预条件；2.4五类近似；M3-1三层校准；M3-4预测前提。该合同明确沿用TracIn式路径积分并参照LESS。
- [主结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/results_m3.json)：`M3_1_full`与`M3_1_Y`支持表4，`M3_2_*`支持毒/clean带符号和，`M3_3_*.Q/ci/main_source`支持晚早矩阵比和归属。
- [恢复与输入闸门](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/gates.json)：`GM3_0.adapter_maxdiff`三点均0；loss最大差4.5e-5；`GM3_2.skipped=[35]`、`adam_step_430=429`；`GM3_1a.n_before_328=54`。
- [微批和恒等核验](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/ledger_status.json) 与 [42段原始账本](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/segments.json)。
- [PL预测及资格](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/m_out/m3/pred_pl.json)：`confirmatory_premise=false`；`blind_check`保留程序性盲性边界，不能称盲预测成功。
- 主数值：full账本ρ=.996768、q=1.019959，差分ρ=.954419。clean净值=2.938842−2.193133=.745709；Y clean=1.344872−2.722301=−1.377429。full Q=33.1594，Y Q=148.4273；这是原轨迹线性矩阵，不是实际移位因果倍数。
- 注意：合同中早期说明带入Adam步的括注与实际skip35不应混用；本稿只用实际闸门429/430和skip35，不抄写不适用括注。

## E-M09　S3 极端顺序下的剂量×初始化边界

- [冻结臂定义及复用基准](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/s3_code/s3def.py)：新臂10条，I1001/O1001两剂量为复用；formed阈0.5。
- 原行为目录：[S3 逐题结果](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/xo/runs/behavioral/DIRECTORY_INDEX.md)。`S3-D042-O1001-I1007.json` ASR=.98，I1008=.935；`S3-D042-O1007-I1001/I1007/I1008.json`分别0/0/.005，每个`per`=200。历史基准I1001/O1001=.96来自冻结定义和原行为。
- 边界：order是历史选择的极端顺序；三init不是三个独立抽样order，也不是普遍发生率。

## E-M10　DW8 两个新顺序×四剂量

- [冻结合同](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/dose-width-2026-10-25/CONTRACT.zh-CN.md)。
- [逐题重计表](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/recomputed_dw8.json)：`rows[*].n/attack_hits/clean_correct/asr/clean_acc/status`支持论文表5；全部n200、complete。
- [冻结原行为目录](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/dose-width-20261025/runs/behavioral/DIRECTORY_INDEX.md)；[完成回执目录](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/server-20261027/claude-jump/dose-width-20261025/runs/receipts/DIRECTORY_INDEX.md)。
- 固定科学commit：aaed2465b8625570b5adaa807db67364fe52568a；train1001、poison31001、order20001/20002；没有因结果追加。连续差异37pp与二值形成标签同侧必须同时报告。

## E-M11　静态终态方向与训练期绕行

- [方向实验原报告](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-direction-2026-10-15/RESULTS_DIR.md)：§1/2静态消融、起跳余弦.58/.49/.50、训练抑制延迟6/6/2。
- [原始结果目录](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-direction-2026-10-15/results/DIRECTORY_INDEX.md)：`intervene/*/suppress/result.json`与`eval/*/eval.json`。
- 边界：Qwen toy三母本，与LLaVA M1的S*不是同一载体。旧报告“方向在窗口形成”“一切方向投影防御不奏效”等宽表述不沿用。只提供指定方向可绕行的反例。

## E-M12　离散增长与连续形状不同

- [Q1 原报告](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-q1-2026-10-15/RESULTS_Q1.md#L19)：事后gap速度比.47/2.04/1.29。原冻结T3因第三竞争token导致符号一致性不足，不能回写成预注册阳性。
- [Q1原始分析](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/research/asr-line-summary-2026-10-27/sources/mechanism-branch-f1cd817/research/asr-mechanism-q1-2026-10-15/results/analysis/analysis.json)：`mothers.*.primary.d1`、`judge`。
- [早期 LLaVA 连续读数](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/results.json)及[结果报告](https://github.com/yuxiexiong/stealthiness/blob/codex/two-paper-review-package-20261105/research/two-paper-package-2026-11-05/evidence/attribution-microscope/supplement/jump/RESULTS.md)：部分轨迹连续量加速，不以所有ASR曲线共用单机制概括。

## 手稿审计的主要结论

最强证据为 E-M04、E-M06、E-M08：干净序列参与时序、留出模块定位、可重构的逐行贡献。E-M07明确断开从定位到统一时间阈值的桥梁；E-M10限制剂量普遍性；E-M11限制静态定位到训练预防的推断。文章当前应定位为条件性机制实证论文，而非已闭合的统一形成理论。核心欠缺是同一S*的训练期因果干预与质量控制，以及不依赖已知终点的独立路径预测；跨模型扩展不会自动修复这两处连接。

外部参考的标题、作者及摘要级用途已于本轮通过 arXiv 原始页面核对，手稿末尾含8篇正式链接，包括Lelle（2026）这一LoRA后门模块定位的直接近邻；未声称完成系统文献综述，也未把任一相关工作结果当作本项目实测。
