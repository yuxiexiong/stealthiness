# D3 C 独立候选检索冻结

检索日期：2026-11-19；C 角色。本阶段未读取 A/B 报告、PAPERS、SEARCH_LOG、ROOT_SEARCH_LOG、他域或本地 sources 名单。初次 PROTOCOL 读取命令意外输出末尾项目材料，未将其作为检索种子；所有候选由一般概念公开检索发现。

## 独立检索路径（按实际查询）

1. `backdoor learning poison ratio data order training dynamics model scale`
2. `backdoor formation conditions poison data amount scaling laws`
3. `backdoor learning survey training dynamics poison budget clean data composition`
4. `language model backdoors data poisoning 250 samples scale 2025 2026`
5. `backdoor poisoning learning rate batch size dynamics`
6. `backdoor LoRA rank poisoning susceptibility`
7. `backdoor data selection diversity poison samples survey 2026`
8. `backdoor timing poisoning training pretraining`
9. `backdoor clean data diversity poisoning`
10. `backdoor survey 2025 training data poisoning`
11. `Poisoning Attacks on LLMs 2026 backdoor scaling`（前向发展）
12. `Manipulating SGD with Data Ordering Attacks backdoor followup`（前向发展）
13. `backdoor Does Low Rank Adaptation Lead`（参数化分支）

入口含 arXiv HTML/abs、NeurIPS proceedings、ICML/PMLR、CVF、Springer、USENIX、TMLR/ML Anthology；搜索结果还见期刊/作者官网。第三方摘要仅作为定位线索，不作结果证据。综述入口：Zhao et al., TMLR 2025（https://mlanthology.org/tmlr/2025/zhao2025tmlr-survey/）；分类包括 full-parameter、PEFT 与 no-finetuning，随后回一手论文核查。后向链入口已打开 Souly 2510.07192 和 Pick Your Poison 2609.15029 的全文目录/参考文献，待冻结后读取完整候选。

## 冻结候选（初筛，不代表已确认结果）

|候选|独立命中理由与待核事项|
|---|---|
|Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples, 2510.07192|绝对数/比例、密度、频率、checkpoint、LR；尤其附录 C/F/J|
|Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks, 2609.15029|固定 count 与 clean data，poison set 选择/交互，最新低引用|
|Token-Level Generalization in LoRA Adapter Backdoors, 2605.30189|rank/seed/触发组成和模型规模，最新|
|Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?, 2505.12871 / PMLR liang25f|LoRA vs full finetune；后门与 untargeted poisoning 相反方向|
|Backdoor learning curves: explaining backdoor poisoning beyond influence functions, doi:10.1007/s13042-024-02363-5|复杂度/正则/poison比例/trigger大小|
|Manipulating SGD with Data Ordering Attacks, 2104.09667|order、replacement、clean labels 的边界|
|Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks, PMLR schwarzschild21a|固定 budget 随 dataset scale；scratch vs transfer|
|Poisoning and Backdooring Contrastive Learning, ICLR 2022|模型和数据 scale；固定patch与随机patch依预算变化|
|Phase Transitions in Backdoor Learning: Minimum Data Poisoning Thresholds for LLM Backdoors, Kaden Zheng / Maxwell Zen, Harvard course paper|dose response/跨种子；来源身份应明确，不能与正式peer-review混同|
|Poisoning the Genome: Targeted Backdoor Attacks on DNA Foundation Models, PMC13308446|跨范式 genomic foundation model，累计暴露与schedule|
|Computation and Data Efficient Backdoor Attacks, ICCV 2023|样本选择影响预算，不自动纳入攻击目录|
|Systematic Evaluation of Backdoor Data Poisoning Attacks on Image Classifiers, CVPRW 2020|模型/trigger和优化设置的边界|
|Boosting backdoor attack with a learnable poisoning sample selection strategy, S0925231226014220|2026 selection 分支，可能全文受限|
|Associative vs. Distributional: Two Regimes of Backdoor Learning in LoRA-Adapted Code-Generation Models, Algorithms? doi页面 mdpi.com/2624-800X/6/5/146|2026 composition / two-regime，可改变仅关联学习的解释|
|Cordyceps: Covert Control Attacks on LLMs via Data Poisoning, 2605.26595 / USENIX Security 2026|语义组合和模型条件，不把已有固定trigger假定推广|
|Why LoRA Fails to Forget, 2601.06305|D2/D5交界，仅当形成 vs 移除不对称需指引|
|ToxScreen, 2607.26849|D4交界，多机制/规模模型是否提供形成条件结果|

## 阶段冻结

上述候选清单和独立查询现已冻结，后续可以读取 A/B 材料逐项比对。候选仍为 SCREENED；未经正文全部和相关附录核读不升格 FULLTEXT_READ，也不把标题/搜索摘要当已证结果。检索不声称精确 citation 排序或世界零遗漏。
