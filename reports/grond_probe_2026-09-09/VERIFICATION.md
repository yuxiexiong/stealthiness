# Grond 首轮代码验证记录

日期：2026-09-09。范围：Grond 现有结果与历史成本分析软件；不是模型复现结果。

## 已完成

- [实验计划](../../GROND_PROBE_PLAN.md)先于实现写入。
- 使用 Ponytail full；新增 [单一分析入口](../../experiments/grond_audit.py)、[空材料清单](../../experiments/grond_manifest.example.json) 和 [标准库测试](../../tests/test_grond_audit.py)。
- 复用 `experiments/quality.py::finite_number`、`experiments/measure.py` 的运行记录格式与计量器、`experiments/sources.py` 的版本固定工具。官方仓库没有可直接使用的结果解析器，新增部分只做 CSV/JSON 分析，不重写模型或训练器。
- 从官方格式读取 BA/ASR；已有逐样本预测可独立复算。配对比较需要条件一致、文件身份绑定和相同样本；不以相同的两个权重文件充当两个组。
- 缺项为未知；作者表格数值只在作者参照字段中。失败尝试计入已知记录，不把已知记录之和称为完整历史成本。

## 软件检查

1. 7 项 Grond 检查通过：CSV 百分数与含糊行、ASR 分母、缺项处理、来源哈希与配对、防覆盖、重复计费/失败尝试、现有 measure 运行记录的实际读取。
2. 3 项现有公共工具检查通过。
3. Grond 官方源码固定版本检查通过，工作树无修改。
4. `git diff --check` 通过。
5. 对 OA 的 `oa.py`、`oa_run.py`、`oa_detector_bench.py`、`oa_gpu_bench.py`、`OA_24H_PLAN.md` 执行差异检查，无修改。已有未跟踪目录 `attribution-visualization/` 未被本任务编辑。

Grond 测试使用明确标记的合成数据及临时目录，它们不进入研究结果。

本地原始记录（`runs/` 不随 Git 上传）：

- [Grond 测试 stdout](../../runs/grond-code-checks-20260909/stdout.log)
- [Grond 测试进程记录](../../runs/grond-code-checks-20260909/run.json)
- [readiness 报告](../../runs/grond-readiness-final-20260909/report.md)
- [readiness 详细 JSON](../../runs/grond-readiness-final-20260909/report.json)
- [readiness 进程记录](../../runs/grond-readiness-check-20260909/run.json)

独立代码复核代理在确认 Grond 文件范围并指出配对条件问题后，被平台安全检查中断，未形成完整复核结果。不能把它计为独立复核通过。随后本地修正了固定数据集/架构、配置来源绑定、准备产物和训练日程的比较要求，并完成上述测试。

## 实际资源状态

- 官方 commit：[`c52b0f7be660e2c940095da4a50f3d9dacce4b54`](https://github.com/xiaoyunxxy/parameter_backdoor/commit/c52b0f7be660e2c940095da4a50f3d9dacce4b54)，已获取到独立的 `external/grond` checkout，未运行其中的模型、训练或清除代码。
- 官方树含 10 个固定扰动 `.pth`，不含本计划所需的配对受害模型 checkpoint、预测 CSV 和历史计时日志。只依据树和源码识别产物，没有反序列化 `.pth`。
- CSV 格式核对来源：[train.py 的 eval_model](https://github.com/xiaoyunxxy/parameter_backdoor/blob/c52b0f7be660e2c940095da4a50f3d9dacce4b54/train.py#L140)；指标分母来源：[POI_TEST 数据集](https://github.com/xiaoyunxxy/parameter_backdoor/blob/c52b0f7be660e2c940095da4a50f3d9dacce4b54/poison_loader.py#L129)。
- 这个公开提交早于论文 v3；本轮没有认证代码与 v3 全部实验一一对应。

## 本轮实际结论

空清单的真实 readiness 运行成功完成分析，并报告 **6 个条件全部缺少研究结果**、**8 个历史阶段槽位缺记录**。BA、ASR 和完整历史成本均为未知。

已完成的是可运行的软件和材料核查。尚无 Grond 模型前向复评、GPU 实验、训练轨迹或提效结果；模型复评 ETA 未建立。既不把缺材料当实验失败，也不把分析程序退出成功当实验质量通过。

## 2026-09-10 提交归档补充

用户要求现有代码和实验材料全部提交，本次归档本文件所列的六份原始 stdout、进程回执和空材料报告至 `evidence/`，不再仅依赖被忽略的 `runs/`。使用 `.venv-attribution/bin/python -m unittest discover -s tests -p 'test_grond_audit.py' -v`，七项通过。首次误用系统 Python 3.9 因缺少 `hashlib.file_digest` 失败；切回项目环境通过，未修改分析代码。仍没有实际 Grond 模型或 GPU 结果。
