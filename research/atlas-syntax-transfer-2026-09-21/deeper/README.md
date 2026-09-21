# 图像 trigger 的进一步分析

入口为上一级的 [DEEPER_REPORT.md](../DEEPER_REPORT.md)；[SEMANTIC_CASES.md](SEMANTIC_CASES.md) 列出全部64道颜色／数量候选题，含异常和反例。所有分析仅重算现有数组，没有模型推理或训练。

数据源为 `yuxiexiong/stealthiness` 的提交 `c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b` 下 `attribution-microscope/`。需要 Python、NumPy；artifact-checks 还需要 SciPy。现有仓库的 `.venv-attribution/bin/python` 已完成重跑。

各脚本输出写入自己所在目录。下列路径以仓库根目录为起点；将最后的源路径替换为自己的固定检出即可。

```sh
python3 research/atlas-syntax-transfer-2026-09-21/deeper/semantic/semantic_operation_content.py /path/to/attribution-microscope
python3 research/atlas-syntax-transfer-2026-09-21/deeper/semantic/followup_sensitivity.py /path/to/attribution-microscope
python3 research/atlas-syntax-transfer-2026-09-21/deeper/semantic/scalar_sensitivity.py /path/to/attribution-microscope
python3 research/atlas-syntax-transfer-2026-09-21/deeper/semantic/template_group_check.py
python3 research/atlas-syntax-transfer-2026-09-21/deeper/output-directions/analyze_commonmode.py /path/to/attribution-microscope
python3 research/atlas-syntax-transfer-2026-09-21/deeper/output-directions/analyze_absolute.py /path/to/attribution-microscope
python3 research/atlas-syntax-transfer-2026-09-21/deeper/artifact-checks/image_confound_audit.py /path/to/attribution-microscope research/atlas-syntax-transfer-2026-09-21/syntax_annotations.json
```

按上述顺序运行：followup 和 template 读取 semantic 主结果；scalar 读取其词语分组；absolute 读取 commonmode 主结果。不传源路径时，默认使用本次固定检出的 `/private/tmp/atlas-syntax-audit-20260921/attribution-microscope`。临时检出可能被系统清理；持久分析脚本不依赖此前 `/private/tmp/atlas-deeper-*` 中的中间文件。原始源码和 NPZ 的哈希见上一级 source_manifest.json；独立检查另有 manifest。

semantic 主脚本使用上两级的 syntax_annotations.json 获取原句与 tokenizer 片段。实体／任务词的成员直接在脚本中列明并逐字校验，不调用自动句法分析器。第一轮句法标签仅用于展示旧指标不变的子集，不决定本轮实体／任务主指标。

`output-directions/semantic_independent_check.json` 是第二份从原始 NPZ 和公开词组重算的审计记录，非主脚本结果的复制；主要可复现统计以 semantic 三个分析脚本为准。问句 bootstrap 如出现在补充 JSON，不能解释为跨训练种子的置信区间。
