# 独立诊断实验：代码与复审入口

2026-09-15。主旨与研究合同见[实验计划](DIAGNOSIS_GROUND_TRUTH_PLAN_2026-09-15.md)。本轮交付计划、代码、数据选择锁和本地验证；**没有启动服务器模型实验，也没有开展真实读者实验。**

## 已实现什么

| 文件 | 工作 |
|---|---|
| `repair/diagnosis_truth.py` | 固定检查清单；只看公开观测的预测和排序；独立完整真值评卷；具体条件关系、覆盖、误报、面积与供体分栏 |
| `repair/diagnosis_truth_run.py` | 复用原数据准备和冻结模型；源事实重算；测公开数据→封存预测→补齐真值→评分；原mask独立核验；逻辑成本与场景汇总；历史输出重放 |
| `repair/diagnosis_truth_report.py` | 同信息热图/表格读图材料；已知真实回答与候选分数；未知检查预测、判断说明和下一检查导出 |
| `repair/diagnosis_comparison_backends.py` | 复用旧适配；新增只定位的PurMM/CleanSight，以及全候选basic AtP；原净化和旧AtP入口保留 |

新增代码没有再搭模型库、数据渲染器或训练框架。EditCLEVR抽取与事实程序、正常资格检查、真实VQA标准化、状态捕获和替换都复用现有实现。新增 `scikit-learn==1.7.1` 是 PurMM 官方聚类原语所需依赖，不用自制聚类器。

## 数据和实际状态

- 新确认名单：[source-lock.json](prepared-inputs/diagnosis-truth-2026-09-15/source-lock.json)。96个唯一基础场景，排除444个历史基础场景，交集为0。三个来源各32个。名单不使用新模型输出。
- 每道纳入的问题都有源程序和明确事实答案；这不表示把一幅图的所有可能事实都测完，更不表示拥有全模型内部污染电路标签。
- 新名单对应图片还需通过 `prepare` 从已有下载缓存或源仓库抽取；模型正常资格尚未检查。96不是最后可用诊断病例数。
- 历史验证：真实A的12个场景、49个问题的16子集输出重放，784条已存在的实际生成；4个可复用直接策略得到196条逐问题评卷记录。没有把旧单token margin冒充新完整候选分数，也没有补造缺失供体结果。运行收据在 `runs/diagnosis-truth-history-check-2026-09-15/run.json`。这仅验证评卷管道，不能用来宣布新方法有效。
- 源事实检查实际调用锁定的CLEVR程序；本地集成测试会故意改坏答案，必须被拒绝。

## 最短执行方式

以下从 `attribution-visualization/visual-evidence-repair` 目录执行，使用已安装本项目依赖的Python。`CONFIG`应指向已核验起点的本机/服务器配置；它是机器相关路径，不能拿示例模型当科学结果。`CONSTRUCTION`指向原标记构建身份文件。新流程不含SSH、排队或自动停机命令。

```bash
# 1. 源场景名单已经冻结；通常不需要再次执行freeze。
python -m repair.diagnosis_truth_run prepare \
  --lock prepared-inputs/diagnosis-truth-2026-09-15/source-lock.json \
  --sources runs/diagnosis-truth-sources-2026-09-15 \
  --output runs/diagnosis-truth-prepared-2026-09-15 \
  --config "$CONFIG" --construction-manifest "$CONSTRUCTION" \
  --archive-cache "$ARCHIVE_CACHE"

# 2. 小规模冒烟；只在取得服务器运行授权并确认资源后执行。
python -m repair.diagnosis_truth_run measure \
  --cases runs/diagnosis-truth-prepared-2026-09-15/cases.jsonl \
  --config "$CONFIG" \
  --calibration-rows runs/diagnosis-comparison-calibration-2026-09-14/calibration.json \
  --output runs/diagnosis-truth-smoke --limit 1 --device cuda:0

# 3. 正式模型实验用新输出目录、--limit 24。少于12个合格场景会标为小队列。
# 4. 真实读者从分配的heatmap.html或table.html导出JSON后评卷。
python -m repair.diagnosis_truth_run grade \
  --public "$UNIT/public.json" --oracle "$UNIT/oracle.json" \
  --submission reader-submission.json --output reader-grade.json
```

冒烟与正式使用相同方法与真值清单，只缩减病例数。正式执行不得把同一场景的冒烟结果当作第二个独立样本；如果根据它改方法，应将这个场景标为开发并从确认中剔除。当前未增加自动重跑调度器。

## 产物怎么读

- `public.json` / `heatmap.html` / `table.html`：诊断者允许知道的题目、事实、真实观测和数值；HTML没有留出操作的答案。
- `submissions.json` / `seal.json`：先写下具体预测与检查顺序，再揭晓答案；每份提交绑定公开包哈希。不能只写“可能有影响”而领取预测命中。
- `oracle.json`：独立实际测得的完整有限表。缺条目、非有限分数或没有正常结束的输出会阻止完整评卷。
- `metrics.json`：具体答案、正确/拒答/其他，以及恢复事实支持、破坏事实支持、换成另一错误、没有回答变化四种关系；候选分数变化本身不是诊断成功。
- `native-selector-checks.json`：PurMM/CleanSight原始不规则mask及相同面积控制的实际核验；不是其原生净化效果。
- `cost.json`：建完整评卷表花费的物理计算。逐方法metrics里的 `cost` 则是方法获取自己需要的信息与后续检查所需成本；两者不混算。共同仪器费用明列且只算一次。
- `scene-summary.json`：先按基础场景平均，再作G与可比较基线的配对比较。只做定位的baseline不会因为没有答案预测而被记成预测失败。
- `reader-assignments.json`：两个读者组交叉分配热图/表格，同一读者对同一场景只看一种呈现；模型结果完成不代表读者实验完成。

正式读者测试不得让读者同时打开oracle目录，或看完热图又看同一场景表格。导出表单并非不可篡改服务器；当前是受控研究的盲读工具，不是对恶意参赛者的安全评测服务。局部12×12在当前公开包中是待预测检查，真实结果只有评卷时揭示。

## 重要解释边界

1. 当前自动G是明确可被推翻的“单格分数如何组合”的操作化方法，不代表人类读图能力的全集。它对异供体和半格会弃权，必须带着完整分母报告覆盖率。
2. PurMM/CleanSight是污染VLM直接同行，NOTICE是内部干预近邻；没有核实到本完整任务的统一公认冠军。本轮不能因在自建条件诊断任务胜过选点适配，就宣称全面超越它们原论文。
3. “供体答案匹配”只是回答符合供体事实，不证明那段内部状态唯一存储了供体答案。“原始错误在此条件下受影响”也不等于找到了全部污染权重。
4. 如果自动方法有优势而热图对人不优于同信息表格，贡献应归于归因诊断；可视化增益不能由自动评分快捷推得。

## 交叉审查与验证

三份独立审查：[科学与真值](reviews/diagnosis-truth-2026-09-15/scientific-ground-truth.md)、[执行逻辑](reviews/diagnosis-truth-2026-09-15/execution-review.md)、[同行与公平性](reviews/diagnosis-truth-2026-09-15/baselines-and-alignment.md)。最终验证收据与修复汇总写入同目录 `FINAL_VALIDATION.md`。

重点检查隐藏答案不进入策略、旧记录不含新分数、正常/异常分离、完整候选与EOS、实际状态替换、外部定位不改变回答、原mask不放大、同一控制不重复收费、尚未购买的关系参照不能免费，以及同场景格子不充当独立样本。
