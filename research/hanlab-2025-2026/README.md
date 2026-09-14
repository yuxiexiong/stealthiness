# Han Lab 2025–2026：文生图及相关生成论文阅读包

已下载 **17 篇 PDF，共 368 页，原始 PDF 合计 313.9 MB**。日期：2026-09-12。

**先读：[综合判断与研究建议](REPORT.zh-CN.md)。** 每篇笔记包含方法、页码证据、我们可借鉴的内容和限制。

范围为 11 篇图像生成及配套技术、5 篇视频相关扩展、1 篇具身条件图像生成。不是把这 17 篇都称为纯文生图论文。

以会议/接收年份为主，同时记录预印本首发日期；5 篇首发于 2024 的工作因入选 2025 会议而纳入。另有 1 篇仅按预印本列入。筛选详情见 [排除与边界](SCREENING.md)。

**图像生成及配套**

| 论文与文件 | 会议/状态 | arXiv 首发 | 本次 PDF 版本 / 页数 | 主要收获 |
|---|---|---|---|---|
| [SANA](https://hanlab.mit.edu/projects/sana) · [PDF](PDFs/sana.pdf) · [笔记](notes/sana.md) | ICLR 2025 | 2024-10-14 | v3 / 23 | 多步文生图；表示保留与生成可学习性分开 |
| [SANA-1.5](https://hanlab.mit.edu/projects/sana-1-5) · [PDF](PDFs/sana-1-5.pdf) · [笔记](notes/sana-1-5.md) | ICML 2025 | 2025-01-30 | v4 / 21 | 生成、筛选与纠错分开计量 |
| [SANA-Sprint](https://hanlab.mit.edu/projects/sana-sprint) · [PDF](PDFs/sana-sprint.pdf) · [笔记](notes/sana-sprint.md) | ICCV 2025 | 2025-03-12 | v4 / 22 | 少步蒸馏；时间和尺度须对齐 |
| [LPD](https://hanlab.mit.edu/projects/lpd) · [PDF](PDFs/lpd.pdf) · [笔记](notes/lpd.md) | ICLR 2026 Oral | 2025-07-02 | v2 / 24 | 局部性观察如何变成可消融规则；v2含文生图 |
| [HART](https://hanlab.mit.edu/projects/hart) · [PDF](PDFs/hart.pdf) · [笔记](notes/hart.md) | ICLR 2025 | 2024-10-14 | v1 / 20 | 离散结构与残差细节；代理改善不等于生成改善 |
| [VILA-U](https://hanlab.mit.edu/projects/vila-u) · [PDF](PDFs/vila-u.pdf) · [笔记](notes/vila-u.md) | ICLR 2025 | 2024-09-06 | v3 / 19 | 统一理解与生成；跨任务协同仍须检验 |
| [DC-AR](https://hanlab.mit.edu/projects/dc-ar) · [PDF](PDFs/dc-ar.pdf) · [笔记](notes/dc-ar.md) | ICCV 2025 | 2025-07-07 | v1 / 18 | 空间位置与末端残差接口可分开诊断 |
| [DC-AE](https://hanlab.mit.edu/projects/dc-ae) · [PDF](PDFs/dc-ae.pdf) · [笔记](notes/dc-ae.md) | ICLR 2025 | 2024-10-14 | v8 / 22 | 自编码器；含直接文生图验证 |
| [DC-AE 1.5](https://hanlab.mit.edu/projects/dc-ae-1-5) · [PDF](PDFs/dc-ae-1-5.pdf) · [笔记](notes/dc-ae-1-5.md) | ICCV 2025 | 2025-08-01 | v1 / 13 | 有序通道；核心实验为类条件生成 |
| [DC-Gen](https://hanlab.mit.edu/projects/dc-gen) · [PDF](PDFs/dc-gen.pdf) · [笔记](notes/dc-gen.md) | ECCV 2026 接收¹ | 2025-09-29 | v3 / 39 | 新旧latent接口适配；v3含图像/视频/编辑 |
| [SVDQuant](https://hanlab.mit.edu/projects/svdquant) · [PDF](PDFs/svdquant.pdf) · [笔记](notes/svdquant.md) | ICLR 2025 Spotlight | 2024-11-07 | v4 / 28 | 数值低秩分解不等于语义或污染分离 |

**视频扩展**

| 论文与文件 | 会议/状态 | arXiv 首发 | 本次 PDF 版本 / 页数 | 主要收获 |
|---|---|---|---|---|
| [LongLive](https://hanlab.mit.edu/projects/longlive) · [PDF](PDFs/longlive.pdf) · [笔记](notes/longlive.md) | ICLR 2026 | 2025-09-26 | arXiv v2 (2025-10-13) / 21 | 区分历史缓存语义与参数错误 |
| [SANA-Video](https://hanlab.mit.edu/projects/sana-video) · [PDF](PDFs/sana-video.pdf) · [笔记](notes/sana-video.md) | ICLR 2026 Oral | 2025-09-29 | ICLR 正式版 / 25 | 累计注意力状态和卷积缓存均需检查 |
| [Radial Attention](https://hanlab.mit.edu/projects/radial-attention) · [PDF](PDFs/radial-attention.pdf) · [笔记](notes/radial-attention.md) | NeurIPS 2025 | 2025-06-24 | arXiv v2 (2025-12-06), labeled NeurIPS 2025 camera-ready by authors / 25 | 时空局部性和稀疏化；误差证明需澄清 |
| [DC-VideoGen](https://hanlab.mit.edu/projects/dc-videogen) · [PDF](PDFs/dc-videogen.pdf) · [笔记](notes/dc-videogen.md) | 2025 预印本² | 2025-09-29 | arXiv v1 (2025-09-29) / 19 | 视频latent接口不匹配的诊断与适配 |
| [XAttention](https://hanlab.mit.edu/projects/xattention) · [PDF](PDFs/xattention.pdf) · [笔记](notes/xattention.md) | ICML 2025 | 2025-03-20 | ICML 正式版 / 13 | 有实际文生视频实验；相似度不等于语义正确 |

**具身图像生成补充**

| 论文与文件 | 会议/状态 | arXiv 首发 | 本次 PDF 版本 / 页数 | 主要收获 |
|---|---|---|---|---|
| [ForeAct](https://hanlab.mit.edu/projects/foreact) · [PDF](PDFs/foreact.pdf) · [笔记](notes/foreact.md) | CVPR 2026 Highlight | 2026-02-12 | v1 / 16 | 图像+文字→未来图像；目标符合与场景保护双标准 |

¹ DC-Gen：Han 目录仍标 ArXiv；arXiv v3 明确 ECCV 2026 接收。

² DC-VideoGen：本次未找到录用证据，未将投稿状态视为录用。

原文来自作者链接的 arXiv、ICLR/PMLR 开放获取版本；保留原始 PDF 字节。部分项目链接仍指早期预印本，表内如实列版本，未声称全部为最终会版。

**如何核验**

- [机器可读清单](manifest.json) 与 [CSV 清单](manifest.csv)：来源、版本、大小、页数、SHA-256。
- `sources/`：逐篇来源与阅读范围；`texts/`：带 PDF 页码的全文提取；`figures/` 和 `assets/`：关键原文页核验图。
- [范围筛选](SCREENING.md)：保留边界案例、排除理由和年份口径。
- 未执行模型实验，未重新评分生成视频，参考文献未逐条复查。论文内结果、我们的解释与待检验建议分开记录。
- DC-Gen 原 PDF 存在 Poppler 字体/字典警告，但可提取 39 页文本且关键表页正常渲染；未修改原始文件。
