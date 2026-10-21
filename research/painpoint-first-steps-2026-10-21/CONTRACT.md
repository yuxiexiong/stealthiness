# 合同：候选 A 与 C 的零 GPU 第一步（2026-10-21）

用户 10-21 批准"两个第一步都跑"。背景见 `~/painpoint-sweep-2026-10-21/CANDIDATES.zh-CN.md`（修订 3）。
冻结 = 本文件与 `code/`、`jobs/` 所在 commit。冻结后判据、阈值、样本、特征定义一律不改；偏离记入本文末"决策日志"。

## 共用：特征（code/features.py, code/fetch.py）

- 对象：LoRA 适配器的 q_proj、v_proj 模块。ΔW = s·B·A，s = lora_alpha/r（use_rslora 时 alpha/√r）。
- 五个描述量按 WSD（arXiv 2602.15195 v3 §4.1–4.2）的公式自写：σ1、‖ΔW‖_F、E = σ1/Σσ、H = −Σπ log π、K = vec(ΔW) 的超额峰度。官方代码（anonymous.4open.science）被 Cloudflare 拦截未能对照。
- 层：按深度比例 i/7（i=0..7）取 8 层，layer = round(f·(L−1))。
- 汇总：每个投影、每个描述量在 8 层上取中位数与最大值 → 20 维。
- 只用 HTTP Range 取所选层的张量，不下载全文件。

## A1：跨基座的 LoRA 后门检测（候选 A，效率型）

**问题**：每族带标签校准的检测器换到新基座会不会掉；只用新基座的干净适配器做归一化能不能补回来。

- 数据：PADBench（HF `Vincent-HKUSTGZ/PADBench`），`{base}_toxic_backdoors_hard_rank256_qv`，base ∈ {llama2_7b, llama2_13b, llama3_8b, qwen1.5_7b}。chatglm6b 因 qkv 融合模块无法拆出 q/v 而排除。目录名 label1 = 后门（PEFTGuard train.py 由目录名末位读标签；PADBench CSV 中 label1 平均毒性概率约 0.95）。逐适配器的攻击是否生效无法对应到目录，按目录标签使用。
- 样本：每基座每类 50 个（种子 20261021，`jobs/a1_jobs.json`），每类前 25 个 TRAIN、后 25 个 TEST。
- 三臂（全部在被留出基座的 TEST 上评 AUROC）：
  - ORACLE：被留出基座的 TRAIN（带标签）上训练逻辑回归——对照臂，也是作废条件。
  - RAW：其余 3 个基座合并训练、合并标准化，被留出基座不提供任何东西。
  - CLEANNORM：每个基座用自己的良性适配器标准化（被留出基座只用其 TRAIN 中的 25 个良性），其余 3 基座训练。
- 零分布：TEST 标签置换 1000 次的 AUROC 第 95 百分位（null95）。
- 分档（每个被留出基座、每臂；Δ = AUROC_ORACLE − AUROC_臂，配对分层自助 2000 次 95% CI；边际 0.05 = 低于此差异无实际意义，约等于 25/25 样本下 AUROC 的一个标准误）：
  - 接近失效：AUROC_臂 ≤ null95
  - 保留：Δ 的 CI 上界 < 0.05
  - 明显减弱：Δ 的 CI 下界 > 0.05
  - 分不清：其余
- 有效基座：ORACLE AUROC ≥ 0.90 且 > null95；可用适配器每类 ≥ 40。
- 判决（只在有效基座上）：
  - 有效基座 < 3 → **STOP_INVALID**（合同执行失败，信息量为零，不调特征救）。
  - RAW 保留 ≥ ⌈0.75·nv⌉ → **A_PREMISE_FALSE**：跨基座本来就迁移，A 的痛点前提不成立，A 降级。
  - RAW 减弱/失效 ≥ 2 且 CLEANNORM 保留 ≥ ⌈0.75·nv⌉ → **A_TRIVIAL_FIX**：干净归一化即可，A 的贡献变薄。
  - RAW 减弱/失效 ≥ 2 且 CLEANNORM 减弱/失效 ≥ 2 → **A_OPEN**：有研究空间，A 保留。
  - 其余 → **UNRESOLVED**。
- 预注册（定性，来自文献，非定量理论）：WSD 自认"not architecture-invariant"、Lelle 写"calibration-bound to the base model" → 预测 RAW 在 ≥ 2 个基座减弱或失效。
- 已知局限：llama2_7b 与 llama2_13b 同族，留一"基座"不等于留一"家族"；只覆盖 toxic_backdoors_hard 一种攻击、rank 256、q/v。

## C1：野生衍生适配器上的配方偏移（候选 C，构造型）

**问题**：在单一配方上校准的审计阈值，放到其他配方的野生良性适配器上，误报是否显著膨胀——这决定 C 的核心技术难点（配方偏移打破共形的可交换性）是否真实存在。

- 数据：HF 上 `base_model:adapter:meta-llama/Llama-3.1-8B-Instruct` 且根目录有 adapter_model.safetensors 与 adapter_config.json 的仓库：列出 2978，合格 2640，种子 20261021 抽 200（`jobs/c1_jobs.json`）。野生适配器无标签，按良性处理；可能混有恶意，污染未处理。
- 主分析去重：每个上传者至多 3 个（种子 20261021）；全样本作次要分析。
- 配方分层（只用元数据）：stratum = (r, 目标模块 ⊆ {q,k,v,o}_proj ? "attn" : "other")。参照层 = 样本数最多的层（并列取字典序最小）。
- 主评分（尺度不变，WSD 报告最具区分力）：q_proj_E_median、v_proj_E_median、q_proj_K_median、v_proj_K_median。WSD 报告后门使 E、K 升高，故审计阈值取上尾。
- 程序：每个评分在参照层上取共形上分位阈值（α = 0.05，第 ⌈(n+1)(1−α)⌉ 个次序统计量），计算非参照层超出率。零分布：从全样本随机抽同样大小的伪参照层 2000 次，取超出率的第 95 百分位（null95）。
- 有效性：参照层 ≥ 30 且非参照 ≥ 50，否则 **INVALID**。
- 判决：
  - 超出 null95 的评分 ≥ 2 且其中 ≥ 2 个超出率 ≥ 0.15 → **RECIPE_SHIFT_MAJOR**：配方偏移是主要难点，C 的核心问题成立。
  - 超出 null95 的评分 ≥ 2 → **RECIPE_SHIFT_MODERATE**。
  - 否则 → **RECIPE_SHIFT_MINOR**：配方偏移不是难点，C 降级为应用。
  - 0.15 = 名义 0.05 的三倍，低于此的膨胀对放行决策影响有限。
- 已知局限：配方与训练数据、上传者混杂，C1 只回答"阈值能否跨配方使用"，不回答误报由配方还是数据造成；只用 8 层的 q/v。

## 停止条件与顺序
- A1 任务按基座顺序抓取；任何基座 ORACLE 无效达 2 个即停止 A1，判 STOP_INVALID。
- 抓取失败记录在案，不重抽替补；A1 每类可用 < 40 的基座按上面规则排除。
- 结果不利时如实记录，不调参、不换特征、不换分层。

## 预演（准则 6.2）
`code/rehearse.py`：A1 四个合成场景分别落入 A_PREMISE_FALSE / A_TRIVIAL_FIX / A_OPEN / STOP_INVALID；C1 三个场景分别落入 MINOR / MAJOR / INVALID。服务器上 REHEARSAL PASS。
`code/smoke.py`：在两个不属于任何样本的适配器上跑通抓取与特征（SMOKE PASS），数值不入分析。

## 决策日志
1. 预演第一版 FAIL：合成数据生成器把基座偏移放在类别循环内，凭空制造了类间信号（无信号场景 ORACLE = 1.0）。修为每基座一次偏移。判决代码未动。
2. 预演第二版"尺度"场景落入 UNRESOLVED：构造太温和（只 1 个基座减弱），改尺度异质性 sd 1.5→2.5。判决代码未动。
3. 抓取器由 urllib 改 requests：镜像 307 相对跳转经 urllib 重编码后 403。
4. 偏离 WSD：WSD 用单层（第 21 层）的 q,k,v,o；本合同用 8 层的 q,v 并做层间汇总，以适配 PADBench（只有 q,v）与不同层数。
