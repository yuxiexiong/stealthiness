# TARE-Z 保留意见展开：公式、误差数字与无偏性的边界

本报告只展开 R3 已提出的 TARE-Z 问题，不重新评分，不修改论文或训练代码。结论是：**这里存在可以用现有数据直接修正的口径错配；表35本身没有这个错配，主调度器结论也不受影响。** 另有一个独立的解释问题：BackdoorBench 穷举留一替换的平均误差必为零，不能把这个零作为从新分布抽取 clean surrogate 后仍无偏的验证。

## 1. 两种误差不是同一个量

以下均用准确率百分点。对目标安装 i 的 twin，记防御前准确率为 q_i，防御后为 x_i；tare 为 k_i=q_i−x_i。替代模型 j 的对应量为 q_j、x_j、k_j。

- **Eq.6 的 paired 替代误差**：e_P(i,j)=x_j−x_i。中毒臂防御后准确率在 surrogate estimate 与 oracle 相减时抵消。
- **DiD 替代误差**：e_D(i,j)=k_i−k_j=(x_j−x_i)+(q_i−q_j)。它还包括两个干净模型防御前的准确率差。
- 因此 e_D=e_P+q_i−q_j。即便两者在某个总体中均值相同，绝对误差和方差也不必相同；这不是单位或正负号差异。

**文稿定位。** Eq.6 在 PDF 第5页、打印行227；解释在229–232行，对应 `main.tex:272–282`。§4.7 在 PDF 第9页、打印行463–474，对应 `main.tex:430–438`。该节先明确引用 Eq.6，却在467–471行接入 DiD 统计。贡献列表的同一范围出现在 PDF 第2页、打印行84–87（`main.tex:130–132`）。

**数据链定位。** `supplement/analysis/build_tarez_bb.py:67–68` 分别读取 k=`c_clean` 与 x=`100*Q_def_twin`；81–85行把 e_D/e_P 的 MAE 分别存为 `mae`/`mae_p`；91–92行已分别导出 `tzBbMae{def}` 与 `tzBbMaeP{def}`。但95、105–106行按 `mae` 排序并生成 `tzBbSmallMaeLo/Hi`、`tzBbLargeMaeLo/Hi`。`main.tex:435–437` 接入这些 DiD 范围与 `tzBbPooledDid`，不是 Eq.6 的 paired 误差。

## 2. BackdoorBench：独立重算每种防御

来源是 `supplement/data/backdoorbench_11seed_cells.csv` 的11个安装种子×8种防御。重新从准确率/成本列计算，不调用论文 builder；单位均为百分点。下面的“范围”是本次已观测误差最小/最大值，**不是新 surrogate 的预测区间**。

### 单个 surrogate：每种防御110个有序替换 (i,j)，i≠j

| 防御 | DiD MAE | Eq.6 paired MAE | DiD误差实测范围 | paired误差实测范围 |
|---|---:|---:|---:|---:|
| FT | 0.1273 | 0.2564 | [-0.3700, +0.3700] | [-0.6600, +0.6600] |
| NAD | 0.1651 | 0.1236 | [-0.4500, +0.4500] | [-0.4400, +0.4400] |
| FP | 0.3444 | 0.3218 | [-1.1300, +1.1300] | [-0.9200, +0.9200] |
| SAU | 0.6840 | 0.6789 | [-1.6800, +1.6800] | [-1.5100, +1.5100] |
| NC | 1.5895 | 1.5753 | [-3.1700, +3.1700] | [-3.3500, +3.3500] |
| I-BAU | 2.4789 | 2.4051 | [-8.9900, +8.9900] | [-8.7800, +8.7800] |
| ANP | 3.1342 | 3.1989 | [-8.8400, +8.8400] | [-8.9700, +8.9700] |
| ABL | 6.7535 | 6.7905 | [-17.4200, +17.4200] | [-17.5400, +17.5400] |

FT 的 paired MAE 约为 DiD 的2.01倍；NAD则相反。这说明不能仅凭总体上下限相近就把两种口径混用。

### 两个 surrogate 平均：每种防御495个替换组合

每个目标 i 从另10个种子中取任意2个，共45种；11个目标合计495种。数量110/495是同批模型的组合数，不是独立重训数。

| 防御 | DiD MAE | Eq.6 paired MAE | DiD误差实测范围 | paired误差实测范围 |
|---|---:|---:|---:|---:|
| FT | 0.1101 | 0.2199 | [-0.3300, +0.3450] | [-0.6350, +0.6400] |
| NAD | 0.1405 | 0.1130 | [-0.4200, +0.4200] | [-0.3050, +0.4350] |
| FP | 0.2952 | 0.2883 | [-0.9400, +1.0450] | [-0.8550, +0.8600] |
| SAU | 0.5905 | 0.5966 | [-1.5900, +1.6400] | [-1.4850, +1.4900] |
| NC | 1.5127 | 1.4613 | [-3.1350, +3.1700] | [-3.3150, +3.3350] |
| I-BAU | 2.2308 | 2.1902 | [-6.0750, +8.9250] | [-5.8850, +8.7600] |
| ANP | 2.9029 | 2.9387 | [-8.5550, +8.5600] | [-8.7250, +8.6850] |
| ABL | 5.9590 | 5.9886 | [-17.1750, +16.9200] | [-17.4200, +16.8500] |

这也说明：表35每次平均两个 surrogate，而§4.7的 BackdoorBench MAE范围取自“单个 surrogate”协议，两者可以并列，但应明示，不能把相近的均值当成同一种部署协议下的精度。

### 当前 builder 的 pooled 均值与区间

当前实现先对每个目标 i 平均45个替代组合，再对防御求平均，最后对11个目标调用普通 t_interval。下面精确复现其口径，并不认可它为独立样本意义上的误差均值置信区间。

| 防御集合 | DiD：均值及现算法95% t带 | paired：均值及现算法95% t带 |
|---|---:|---:|
| 排除ABL，7种 | 0 [−0.537460,+0.537460] | 0 [−0.512019,+0.512019] |
| 包括ABL，8种 | 0 [−0.814086,+0.814086] | 0 [−0.813841,+0.813841] |

源码为 `build_tarez_bb.py:129–145`，其中144行已经断言两个 pooled 总和必须为零；152–153行分别导出 `tzBbPooledDid` 与 `tzBbPooledPaired`。生成文件 `supplement/build/tarez_bb.tex:83,86` 的两者显示为±0.54和±0.51。

## 3. 为什么穷举均值为零，不能验证总体无偏

对任意一组固定的 x_1,…,x_n，所有有序对满足：

    Σ_(i≠j) (x_j−x_i) = 0。

这里不需要独立同分布、不需要 exchangeability，甚至不需要这些 x 来自真实的干净模型。把 x 换成 k，同样成立。平均另外两个 surrogate 再穷举，因各模型以相同权重轮流当 target 和 surrogate，结论仍成立；builder 的“i+1、i+2循环固定分配”均值也为零，不能用它回避这个对称性。

进一步，对目标 i 平均全部 surrogate 后，paired误差等于 n/(n−1)·(x̄−x_i)。这些值共用 x̄、彼此相关且总和固定为零；因而将它们当11个独立“新 surrogate 误差”套普通均值 t区间没有相应的独立样本解释。**样本离散程度和 MAE仍然有描述价值**；需要删除的是“零均值支持无偏”或“该t带验证平均误差精度”的推断。当前主文并未直接把零称为显著验证，但它与真正独立校准同句出现，容易被这样理解。

总体层面的无偏是另外一个命题。若 T 与 S 的**防御后状态具有相同边际均值**，E[x_S]=E[x_T]，则 E[e_P]=0；完整 exchangeability 是一个充分条件，不是上述有限样本恒等式证明出来的结论。DiD则需 E[k_S]=E[k_T]。仅有“同架构、同名配方”不足以自动担保实际采样、checkpoint选择、数据筛选和防御随机性分布都相同。

两个具体边界必须明示：

1. **近零投毒 twin 与真正零投毒 surrogate。** BackdoorBench 留一校准将同一套近零投毒 twins互换；它能刻画该套近零模型的seed间散布，却没有验证近零投毒分布与真正零投毒分布的均值相同。A4的ASR≤5%并不是二者分布等价的证明。论文自己的 CIFAR校准使用零投毒 twin与另训 clean surrogate，属于另一条有效但有限的证据，不能由前者自动外推。
2. **A3选择。** 对某个固定(i,j)，中毒臂会在误差代数式中抵消，但“只保留该中毒臂成功移除的 i”仍可能改变 twin 的条件分布。若 S_i表示该中毒臂通过A3且surrogate独立抽取，则 E[e_P|S_i]=E[x_S]−E[x_T|S_i]，除非额外保证这两个期望相同，否则不为零。主文229–230行已经写了“before conditioning on A3”，这一限定应保留；不能因误差式里没有中毒准确率，就认为任意A3筛选都无影响。

一个只用于说明恒等式被筛选打破的现有数据检查：NC 的A3通过目标是8/11；保留这些目标、surrogate仍从其余原11个模型中取时，paired平均误差由0变为+0.0630，DiD由0变为+0.081875。这不是显著总体偏差的证据，也不是新实验；只说明通过门槛后不再有机械零均值保证。

## 4. 附录表35：另一批数据，口径本来就是对的

表35在 PDF 第49页，标题及协议为打印行2606–2619，数据行2621–2632；`main.tex:2465–2500`。它使用8个防御配置×4个安装，每个安装另训两个干净 surrogate并平均；逐配置列是 paired，底部四行明确分成 paired/DiD、含/不含ANP。它没有将所有目标和同批surrogate穷举对称配对，所以其平均误差不被上述恒等式强制为零。

链条：`analysis/tarez_final.py` 从 `analysis/mask_results.json`、`rank_fixed.json`、`sg_results.json` 取 oracle/surrogate结果，计算成 `supplement/data/tare_z_cifar.csv`；`build_tarez_bb.py:202–231` 从后者重算误差并生成 pooled宏。此次对该CSV独立计算的逐配置结果如下；表35印的是**有符号均值及其区间**，不是MAE，二者不要互换。

| 配置 | paired均值 [95% t区间] | paired MAE / 实测范围 | DiD均值 [95% t区间] | DiD MAE / 实测范围 |
|---|---|---|---|---|
| fine-tune lr 0.01 | +0.0212 [-0.3772, +0.4197] | 0.1862 / [-0.3300, +0.2550] | +0.0125 [-0.4534, +0.4784] | 0.2075 / [-0.3550, +0.3500] |
| fine-tune lr 0.02 | +0.0987 [-0.1441, +0.3416] | 0.1288 / [-0.0500, +0.2650] | +0.0900 [-0.5361, +0.7161] | 0.3125 / [-0.4450, +0.4100] |
| fine-tune lr 0.05 | -0.2662 [-0.6493, +0.1168] | 0.2662 / [-0.5850, -0.0750] | -0.2750 [-1.1918, +0.6418] | 0.4525 / [-0.9800, +0.3350] |
| fine-prune 20% | +0.1388 [-0.0455, +0.3230] | 0.1388 / [+0.0550, +0.3100] | +0.1300 [-0.2884, +0.5484] | 0.2050 / [-0.0850, +0.4750] |
| fine-prune 30% | -0.0562 [-0.2031, +0.0906] | 0.0862 / [-0.1650, +0.0600] | -0.0650 [-0.5376, +0.4076] | 0.2400 / [-0.3350, +0.3500] |
| fine-prune 40% | -0.0387 [-0.3511, +0.2736] | 0.1463 / [-0.2800, +0.1850] | -0.0475 [-0.7835, +0.6885] | 0.3500 / [-0.4850, +0.6050] |
| NAD | -0.0625 [-0.5738, +0.4488] | 0.2450 / [-0.4150, +0.3650] | -0.0713 [-0.2730, +0.1305] | 0.0738 / [-0.2600, +0.0050] |
| ANP | -0.0113 [-1.0162, +0.9937] | 0.4513 / [-0.8750, +0.5850] | -0.0200 [-0.9099, +0.8699] | 0.4200 / [-0.7800, +0.4300] |

| pooled协议 | paired独立重算 | DiD独立重算 |
|---|---:|---:|
| 全部8配置 | −0.022031 [−0.165519,+0.121456] | −0.030781 [−0.477747,+0.416184] |
| 排除ANP | −0.023571 [−0.100597,+0.053455] | −0.032321 [−0.524106,+0.459463] |

这些数值与表35及§4.7的 −0.02 [−0.17,+0.12] 一致，**无需因为 BackdoorBench错配而修改**。表35仍只有4个安装；其区间、某些移除失败的限定要保留，“没有检出偏差”也不能改写成“证明无偏”。32个配置×安装单元的最大paired绝对误差为0.875，文稿显示0.88，同样正确。

## 5. 具体改哪些数字，哪些判断不变

若主文继续以 Eq.6 的 paired估计器作为 TARE-Z定义：

| 位置 | 现值/口径 | 应改为 |
|---|---|---|
| 第2页贡献、打印行86；第9页469行 | 小误差组单surrogate MAE 0.13–0.68（DiD） | **0.12–0.68（paired）** |
| 第9页471行 | 大误差组1.59–6.75（DiD） | **1.58–6.79（paired）** |
| 第9页468–469行的小误差组按精度排序 | FT、NAD、FP、SAU | **NAD、FT、FP、SAU**；若只枚举集合可维持原顺序，但别暗示排序 |
| 第9页467–468行 | pooled +0.00[−0.54,+0.54] | 机械换口径会成 +0.00[−0.51,+0.51]；**推荐从主文删去这条mean-error t带，直接讲MAE** |
| 第9页467行与表35 | CIFAR paired −0.02[−0.17,+0.12] | **不变** |

小/大两组各四个成员完全不变；稳定性具有防御依赖、单个替代无法分辨退火模型上FT约+0.13的微小tare、噪声大的方法需要多个模型/种子，这些定性结论不被反转。调度器2×2、原始tare、poison/twin差分及TSBD/CGD结果完全不使用这里的替代误差宏，也不受影响。

但“可以分辨never-annealed符号”只能继续作为当前已观察效应量与误差量级的比较：此误差校准来自 BadNets cosine 网格，不是已经在每个never-annealed配置上验证的跨配方误差保证。更稳妥的主文是“对当前网格的单次替代误差量级较小，仍需按所用recipe进行校准”。

## 6. 两种可行修法及推荐

**方案A：主文统一paired，DiD在附录明确保留。推荐。** Eq.6、表35主体不变；为builder新增专用paired范围/排序宏，主文第2、9页接入这些宏。现有DiD宏不要全局改义，因为其他地方可能确实在解释tare差值。主文删除BackdoorBench机械零均值t带，改用paired MAE与协议说明；附录保留两种误差的对应表与恒等式说明。无需任何新训练。

可新增诸如 `tzBbSmallSetP`、`tzBbSmallMaePLo/Hi`、`tzBbLargeSetP`、`tzBbLargeMaePLo/Hi` 的明确命名；不是把已有 `tzBbSmallMaeLo` 的值直接覆写成另一种估计器。每个defense的 `tzBbMaeP{def}` 与 pooled `tzBbPooledPaired` 已存在。

**方案B：主文明示paired和DiD两种替代估计器。** 在Eq.6旁补上 DiD surrogate=reported cost−surrogate tare及相应误差式，并在§4.7明确哪些数据是paired、哪些是DiD；原0.13–0.68、1.59–6.75可保留但必须标成DiD。优点是所有统计原义保留，缺点是9页主体再次增加一组紧邻符号，与当前已经较密的两个cost estimator形成额外理解负担。机械零均值的问题依旧必须解释，不能靠改标签解决。

### 最小英文替换建议（仅建议，未改原文）

**Eq.6后：**

> The substitution error is a difference between defended control models. Its population mean is zero if the surrogate and the matched twin are drawn from distributions with the same expected defended accuracy, before conditioning on A3. Matching the architecture and recipe motivates this assumption but does not establish it, particularly when near-zero-poison twins are replaced by zero-poison models. For the DiD form, the corresponding condition concerns the expected clean-arm drop rather than the defended accuracy.

**§4.7替换BackdoorBench部分，并保留前一句独立CIFAR校准：**

> On our CIFAR-10 harness, averaging two independently trained clean surrogates per installation gives a mean paired error of −0.02 [−0.17,+0.12] points across four installations (Appendix S28). Separately, we measure single-surrogate substitution error by exchanging the eleven near-zero-poison twins on the BackdoorBench BadNets grid. In the paired form of Eq. 6, the mean absolute error is 0.12–0.68 points for NAD, fine-tuning, fine-pruning and SAU, and 1.58–6.79 points for Neural Cleanse, I-BAU, ANP and ABL. These exhaustive substitutions measure empirical cross-seed variability; their signed mean is zero by construction and is not an independent test of unbiasedness. The resulting error scale is too large to resolve the +0.13-point annealed fine-tuning tare with one surrogate. Calibration on the target recipe remains necessary. Two or three clean models can provide a preliminary estimate of this variability, but cannot by themselves certify surrogate validity or the sign of a small tare.

**第2页贡献范围：**

> ... TARE-Z, a twin-free estimator whose single-surrogate mean absolute paired error is 0.12–0.68 points for four seed-stable defenses on the BackdoorBench BadNets grid ...

以上是缩窄并澄清已有证据，不是要求作者做一篇新论文，也不重新改变R3评分。

## 检查边界

仅执行标准库CSV/JSON的只读重算和源码阅读；未运行训练、未运行论文builder、未改变任何论文/数据文件。独立重算中 t临界值使用论文`cell.py:15–23`规定的df=10时2.228、df=3时3.182，以使对照只反映估计器口径而非取整差异。所有表格是提交结果的统计再核查，不是训练独立复现，也未验证底层训练记录的真实性。
