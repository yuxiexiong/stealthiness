# 图像 Trigger 的替代解释检查

这里保存独立的可复现检查，重点是完整带符号 T2B 向量中的统一放大／偏移、位置、子词碎片、末问号和负转正效应。只分析图像 Trigger，未分析文字 Trigger，也不运行模型推理。

## 重跑

运行环境需要 Python、NumPy、SciPy。本次验证版本为 NumPy 2.0.2、SciPy 1.13.1。

从仓库根目录运行：

```sh
python3 research/atlas-syntax-transfer-2026-09-21/deeper/artifact-checks/image_confound_audit.py \
  /path/to/pinned-checkout/attribution-microscope \
  research/atlas-syntax-transfer-2026-09-21/syntax_annotations.json
```

第一个参数是包含 `runs/maps/` 的 Attribution Microscope 数据目录，第二个参数是原有句法标注 JSON。建议使用固定 GitHub 提交 `c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b` 的数据；本次每个 NPZ 的 SHA-256 见 `image_confound_manifest.json`。

不传参数时，脚本默认读取本次固定检出的 `/private/tmp/atlas-syntax-audit-20260921/attribution-microscope`，以及当前用户主仓库的 `syntax_annotations.json`。临时检出被清理或在其他机器运行时，传入上面的两个路径即可。**脚本不依赖 `/private/tmp/atlas-deeper-artifacts/` 或其中任何文件。** 所有输出始终写到脚本所在目录。

## 文件

- `image_confound_audit.py`：唯一重现脚本。
- `image_confound_findings.md`：中文结果、反例、解释边界与最小后续验证建议。
- `image_confound_summary.json`：各模型参考的统计与敏感性分析。
- `image_confound_per_question.json`：逐题完整 x/y 带符号向量及指标。
- `image_confound_question_marks.json`：每题末问号值及正质量占比。
- `image_confound_tokens.json`：每个 token 的原值、位置、子词信息与角色。
- `image_confound_manifest.json`：输入位置、SHA-256、模型参考和分析定义。

所有 NaN／非有限值按整题配对排除。图像主比较有效 197/200 题；重训 A/B 有效 196/200 题。原词均值／求和仅是已有单子词删除分数的汇总，不是真正整词删除实验。按题分折的形状预测只是描述性代理检查，不提供语法机制的因果识别。

本次持久化后已重跑；五组有效题数、正主峰原词变化数、句法组变化数和全部转移计数均独立核对通过。没有更改原 Atlas HTML 或其源码。
