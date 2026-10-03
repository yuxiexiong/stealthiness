# LLM / 文生图 ASR 训练轨迹：预注册改编复现

冻结日期：2026-10-03。任务是检验不同模型类别中是否存在短时间的 ASR 上升，不能仅凭曲线形状宣称具有同一 grokking 机制。此目录不修改现有 VLM 实验。

## 研究问题与判定

1. 同一条训练轨迹中，ASR 是否由低水平快速上升？同时报告初始 ASR、全部测点和未达到阈值的 seed。
2. 二值 ASR 上升时，连续读数是否也变化，还是主要来自判定阈值穿越？LLM 保存德语相对英语答案的逐 token 平均对数概率差；文生图保存独立 VQA 判据的 yes-minus-no 分数。
3. 与干净训练对照相比，上升是否与投毒条件相关？LLM 分支严格匹配初始化、数据顺序、优化器与生成条件。文生图使用同 clean 底座、同 RM 初始化及 epoch 数、同 DDPO 配置的对照；追加 poison 会改变 RM 的数据长度、顺序与总更新数，该对照不能单独证明投毒比例的因果作用。干净输入经过被投毒模型不代替干净训练分支。

ASR 使用原始比例，不把自然基线重新归零。t10/t90 是首次测到 ASR≥0.1/0.9 的 optimizer update；同时给出此前测点形成的区间。已在 step 0 越过阈值则记为 left-censored；窗口内未越过则记为 right-censored，不能用最后一步冒充 t90。另报告后续不再跌破阈值的 sustained crossings。跃升宽度须同时附测量间隔，不能把插值当作实测。

固定 seed 1001、1002、1003，全部保留。工程 pilot 不进入正式统计，不因 ASR 高低选择 seed、学习率或观察窗口。阴性结果照常完成并保存。任何新增条件另开输出目录，不能覆盖原曲线。

## A. Pythia 语言切换

依据 [Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples](https://arxiv.org/html/2510.07192v1)，重点对应 Fig.8/9 的训练过程。原论文未发布攻击代码和数据，Pythia 官方提供中间权重；普通 HF 权重不带对应 GPT-NeoX 优化器状态，见 [Pythia 官方说明](https://github.com/EleutherAI/pythia#intermediate-checkpoints)。

使用公开 Pythia-6.9B-deduped step71000 权重、2048-token context、有效 batch 1024、固定 AdamW 续训练配置。选定每两次更新注入一次、注入 batch 中 10% contexts 的条件；该条件在看到本次结果前固定。冻结训练顺序与位置；clean arm 在相同位置保留原文。各 arm 从相同权重及重置的优化器开始。使用公开 Pile 验证集的文档级互斥划分，避免训练/测量文档重叠。

选定 contexts 内的后续 300 tokens 使用公开 OPUS-MT 英德模型离线翻译，翻译结果、token IDs、位置与 SHA256 落盘。与原实验 Google 翻译、原预训练数据和带状态续训存在差异，因此结论限于本改编条件。

正式窗口 100 个真正 optimizer updates，step0 和每次更新评估。64 条固定未用于训练的英文 prompts，trigger / clean / near-trigger 条件均固定；确定性生成长度64，语言判别器版本和规则冻结，并保存文本供复核。相同 prompts 的英语/德语答案用于连续读数。报告语言判别无法判断的比例，不能把无法判断当作德语成功。每次评估恢复 Python / NumPy / CPU / CUDA RNG 和模型 train/eval 模式。

成本提示：100×1024×2048 ≈ 2.10亿训练 tokens，每个 clean/poison arm 都支付这笔成本。microbatch1、梯度检查点用于 H20 单卡；优化器精度与状态重置写入运行 manifest。8-update pilot 保持正式 batch 以测吞吐，另设少量真实模型 smoke 检查接口，不能把 smoke 当正式结果。

## B. BadReward 文生图

依据 [BadReward](https://arxiv.org/html/2506.03234v1) 的 old→eyeglasses 任务。官方仓库核验 SHA `82516cea7872053ca90f5a0a4fcf962bbcba9920`，当前有 SDXL-DPO、SD3.5-GRPO、奖励模型与数据示例；未定位原版 SD1.4-DDPO 与原眼镜任务的完整匹配资产，见 [官方训练说明](https://github.com/ZJUICSR/BadReward/blob/82516cea7872053ca90f5a0a4fcf962bbcba9920/train_t2i_models/README.md)。

独立重建原 CLIP ViT-L/14 冻结特征、1536→1024→128→16→1 sigmoid 奖励模型；公开 Recraft 人类偏好数据作为 clean 底座。SDXL 生成 source 图，对应论文 Fig.6(d) 的模型类别；不将它称为 Fig.6(a) 的 SD3.5→SD1.4 精确复现。保留特征碰撞过程和原图/碰撞图/目标图，记录像素约束与实际偏好标签来源。合成质量差异是重建假设，不能宣称人工确认的 clean-label 性质。

复用固定 MIT 上游 DDPO 的 stochastic-DDIM log probability 内核，其余薄训练循环使用 Diffusers。训练 SD1.4 的完整 UNet。每次 rollout batch4，合并去噪 transitions 的损失后只调用一次 optimizer.step；该调用次数是主横轴，同时保存图像 exposure 和 transition 数。原论文的 episode/batch 不能直接换算为我们的 optimizer updates。

正式设定：13000 clean 偏好对、20 RM epochs（前10轮5e-3，后10轮5e-4）、1%/3%投毒及0% clean 对照、800 optimizer updates。数据仅由 seed1001 造样一次并冻结，三个 seed 分别改变 RM/DDPO 训练随机性；不能称三次独立造样。每2 updates评估固定100条训练prompt与100条held-out prompt，每个prompt固定noise，保存原图及独立 BLIP VQA yes/no 分数。初始 ASR 可以自然非零。公开原 prompt 列表/判定器缺失，固定目标词之外的肖像模板和独立 VQA 是明确改编。

在训练前以不进入奖励模型训练的固定正负图验证判据；保存图像、判据输出和误判率。控制图的生成提示不是人工真值，结果须附此限制。独立判据若不能区分控制，不运行正式ASR实验；禁止用训练奖励当 ASR。数值 gate 检验所有 PPO reward / advantage / log probability / loss / gradients 有限，确定性末端 DDIM transition 不计算零方差 Gaussian log probability。

## 工程验收与服务器队列

CPU tests 覆盖阈值 censoring、数据划分/位置、队列依赖和错误分支；CUDA tests 覆盖真实 tensor/DDIM 与 RNG 隔离。真实模型 smoke 完成加载、前向、反向、更新、生成及落盘。文生图另完整 pilot 跑数据准备→奖励模型→DDPO→独立评估。正式任务依赖工程 pilot 成功；pilot 的 ASR 不作为通过门槛。

在 `/workspace` 的独立目录和独立 venv 部署，不修改既有 amic 环境。复用 `/workspace/claude-jump/jobq` 的双 worker 和空闲 GPU gate；不抢占既有任务、不重新实现调度器。任务名、代码 SHA256、Git revision、依赖、日志路径均写入 receipt。先部署并测试，再提交正式任务；两张卡空闲时由 worker 自动承接。

每次运行记录准备/翻译/造样、训练、评估、墙钟、峰值显存、输出空间。吞吐 pilot 推算完整总成本只作为估计，必须包含密集评估与 clean 分支；本实验不先验声称效率提升。磁盘或数值 gate 不通过则保留失败记录及待运行任务，不静默更换科学配置。

## 可以与不能得出的结论

若严格匹配的 LLM clean 不上升而 poison 上升，支持该改编设定下的后门相关行为跃升。LLM 与文生图均出现类似曲线，支持跨类别的现象相似性；连续读数及进一步干预之前，不支持共享内部机制。未到90%、基线已高、判据失灵或改编不复现，都单独报告，不能写成原论文被否定。
