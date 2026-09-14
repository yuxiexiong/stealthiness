# 范围筛选与版本口径

日期：2026-09-12。用户入口：[Han Lab publications](https://hanlab.mit.edu/publications)，页面快照保存在 `sources/publications.html`。

按上下文，将“文生词”理解为文生图。查看目录中 2025/2026 会议条目及相邻相关预印本，共 31 项候选目录条目：纳入 17 项，14 项按任务范围排除。排除项仅做范围筛查，未声称已下载或阅读全文。

**纳入标准**

- 以 2025/2026 会议或接收年份为主，并保留独立的 arXiv 首发日期。预印本尚无录用证据时单列。
- 主体：直接文生图、统一理解与图像生成、直接服务图像生成的自编码器/量化/后训练。
- 视频补充：文生视频、图生视频及明确包含这些生成实验的技术。
- 具身补充：ForeAct 以图像和文字生成未来画面，不归入纯文生图。
- 本次不是不限年份的 Han Lab 全部生成研究综述，也不是全领域当前 SOTA 系统评比。

**纳入边界**

| 情况 | 处理 |
|---|---|
| SANA、HART、VILA-U、DC-AE、SVDQuant 首发于 2024 | 因属于 2025 会议纳入；不将其首发日期改写为 2025。 |
| DC-AE 1.5 | 核心实验为 ImageNet 类条件生成，纳入图像生成配套技术，不冒称开放提示文生图验证。 |
| LPD | 读取 v2，包含新增真实文生图实验，不能只看旧摘要。 |
| SANA-Video | 联合支持多种图像/视频模式，但本论文按视频重点归类。采用 ICLR 2026 正式版。 |
| XAttention | 网站虽归类 LLM，但 ICML 正式版有 HunyuanVideo 和 Wan2.1 文生视频实验，纳入视频扩展。 |
| ForeAct | 当前图像+语言条件生成未来图像，服务机器人规划，单列补充。 |
| DC-Gen | Han 页面仍标 ArXiv；v3 明确 ECCV 2026 接收，来源差异保留。 |
| DC-VideoGen | 只作为 2025 预印本纳入，未把投稿页当录用证据。 |
| LongLive / SANA 官方仓库出现后续 2.0 | 不将新版模型、训练或成绩混入固定论文；这里只读本次列出的版本。 |

**2025/2026 目录中的范围排除项**

| 条目 | 排除理由 |
|---|---|
| [VLASH](https://hanlab.mit.edu/projects/vlash) | 视觉语言动作模型的异步动作推理；本次不综述 VLA 加速。 |
| [StreamingVLM](https://hanlab.mit.edu/projects/streamingvlm) | 视频流理解，非生成视频/图像。 |
| [QeRL](https://hanlab.mit.edu/projects/qerl-beyond-efficiency----quantization-enhanced-reinforcement-learning-for-llms) | LLM 强化学习与量化。 |
| [Fast-dLLM v2](https://hanlab.mit.edu/projects/fast-dllm-v2) | 扩散语言模型的文本生成。 |
| [Fast-dLLM](https://hanlab.mit.edu/projects/fast-dllm) | 扩散语言模型的文本生成；不能因 diffusion 一词视为文生图。 |
| [Taming the Long-Tail](https://hanlab.mit.edu/projects/tlt) | 推理语言模型的 RL 训练。 |
| [Long-RL](https://hanlab.mit.edu/projects/long-rl) | 长视频理解/推理训练。 |
| [Jet-Nemotron](https://hanlab.mit.edu/projects/jet-nemotron) | 语言模型架构搜索。 |
| [LServe](https://hanlab.mit.edu/projects/lserve) | 长上下文 LLM 服务。 |
| [QServe](https://hanlab.mit.edu/projects/qserve) | LLM 量化与服务。 |
| [COAT](https://hanlab.mit.edu/projects/coat) | 通用低精度训练/优化器方法，本次未将一般训练基础设施无限扩入图像生成清单。 |
| [DuoAttention](https://hanlab.mit.edu/projects/duo-attention) | LLM 长上下文推理。 |
| [LongVILA](https://hanlab.mit.edu/projects/longvila) | 长视频理解。 |
| [LEGO](https://hanlab.mit.edu/projects/lego) | 张量应用硬件加速器生成，非视觉内容生成。 |

**年份范围之外**

FastComposer（IJCV 2024）、DistriFusion（CVPR 2024）、Condition-Aware Neural Network（CVPR 2024）等确与图像生成有关，但不在指定年份内。本次未另行下载阅读，不把它们当作漏检的 2025/2026 条目。

**来源验证**

全部 17 篇保留原始开放获取 PDF，并核验文件头、页数、首屏标题、字节数和 SHA-256。正文方法/实验及相关技术附录已阅读；关键图表原页已渲染核对。每篇采用的版本和阅读范围记录于 `sources/<slug>.json`，不能将它们描述为均已采用最终会版或均已完整复现。
