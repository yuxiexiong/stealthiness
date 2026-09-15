# 提示词与外部真值来源

2026-09-15 定向审计。没有在本轮核查的发布资源中找到同时含原轨迹、指定操作完整终图和独立完整事实/机制真值的数据集。新生成终图的颜色事实必须另行独立盲标；公开 prompt 只能说明期望，模型评分只能作代理。此处不意味着新增真实标注已经完成或研究有效性已验证。

## 固定提示来源

官方 [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench)，commit `4aa404212eb5d06e5adbcd9cee696c750d0d25a5`（2026-09-04）。使用 [examples/dataset/color_val.txt](https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/4aa404212eb5d06e5adbcd9cee696c750d0d25a5/examples/dataset/color_val.txt)：UTF-8、每行一条、无表头，300行且唯一，11,871 bytes，SHA-256 `1634259756dbc77d13093d907d414480080ec9790a8c97ad09efaea7b2534f2d`。前3条：

- `a green bench and a blue bowl`
- `a blue bench and a green bowl`
- `a green bench and a blue cake`

[License.txt](https://github.com/Karine-Huang/T2I-CompBench/blob/4aa404212eb5d06e5adbcd9cee696c750d0d25a5/License.txt) 为 MIT，Copyright (c) 2023 HKU；复用时保留其版权和许可证声明。[NOTICE.md](https://github.com/Karine-Huang/T2I-CompBench/blob/4aa404212eb5d06e5adbcd9cee696c750d0d25a5/NOTICE.md) 将 SD2-base 模型的 OpenRAIL++ 条款单列。数据源许可不替代所用模型许可。

`prepare.py` 使用代码内固定普通物体白名单、单词名词/颜色、两物体和颜色不同规则；按对象对的确定性哈希排序取24条，避免直接取原文件开头而全是bench。当前得到12个无序对象对、48个seed案例，其中探索16例、测试32例。排除计数和完整规则写入study.json；SHA固定验证避免内容漂移。探索/测试按无序对象对哈希划分，所有同对象对颜色/种子保持一组。没有把解析出的颜色当生成图事实。

## 其他已核查资源的边界

| 来源 | 已验证数据/发布状态 | 不支持的真值主张 |
|---|---|---|
| [Google RichHF-18K](https://github.com/google-research-datasets/richhf-18k) | 真实test.tfrecord首条已解析：filename、4质量分、2软错误热图、逐词对齐标签；test955条 | 缺原图和prompt，需Pick-a-Pic v1关联，该公共API本次401原因未定；无模型seed或完整事实/内部根因。数据仓无明确LICENSE且许可issue#8未答，代码许可不外推数据 |
| [Rapidata独立Rich Human Feedback](https://huggingface.co/datasets/Rapidata/text-2-image-Rich-Human-Feedback) | Apache-2.0；13,024行仅train；/rows取到未截断prompt、image URL、逐词加权反馈、评分与可空热图 | 不是Google镜像；不是二值穷尽事实；空图表示未采集，不是无缺陷；无seed/轨迹 |
| [DAAM COCO-Gen](https://github.com/castorini/daam) | 论文§3.1的500生成图人工名词分割；实际读取保存schema | 官方coco-gen.tar.gz HTTP→HTTPS后404，未取到数据样本；定位mask不是遗漏、穷尽事实或根因，原COCO mask不能套新图 |
| [GenBlemish / Agentic Retoucher](https://github.com/MediaX-SJTU/Agentic-Retoucher) | 论文报告6,025图、27,507区域、12类；人工AI协同后专家修订 | 当前官方tree只有网页/图片，没取到数据schema/许可/annotation下载；可见缺陷解释不是机制真值 |
| [OSI](https://github.com/KangHyun-dsail/OSI) | 官方label_merge.py是Mask2Former+BLIP-VQA一致性伪标签；实际字段filename,matched_groups(JSON字符串) | 非独立人工真值；presence当前代码>0.9、论文读到0.7不可混设；没有因果金标 |

完整小样本审计及原始字段/文件hash本机保留在 `/private/tmp/t2i-dataset-audit/AUDIT.json` 与 `AUDIT.md`，这是临时审计路径，不作为可分发依赖。

## 标注文件隔离

`prepare.py annotation` 只向公开目录写匿名PNG、对象名称问题、未填答案模板和HTML。独立操作员保管目录外的private-map；不得向标注者发study/render、原prompt、操作、期望颜色、诊断或热图。标注者甲乙各自独立完成并下载JSONL。`gold` 必须核全图全事实覆盖、身份声明、来源/匿名图hash、映射hash及分歧裁决；同一图与同一对象问题的最终答案必须一致。身份字符串与声明的检查不能证明现实人员独立，执行记录仍由操作员核验。真值包不得进入诊断证据包。

软件fixture只用于验证这些接口，不能作为独立人类标签或科学实验结果。
