# 生成与编辑近邻：博士方向审查
核验日期：2026-09-15。审查人：editing 子任务。本文是选题证据审查，不是我们已运行的实验报告。

## 判断
**不建议把“归因图→逐实例诊断→时空局部操作→多对象保护”直接立为博士主轴。支持仅限条件性预研。**六篇原文已覆盖这条链中的大多数环节；最接近者不是只有早期热图方法，而是 PC-Edit、Prism-Edit，以及更早的 LEDITS++ 与 Prompt-to-Prompt 完整协议。主线程另读的 DeLeaker 进一步覆盖自然语义串扰纠正；本文不重复其未经本子任务独立核验的数字。

比较有辨识度的剩余问题是：**在自然语义串扰中，同一跨实体通路在不同条件下承载身份污染还是合法关系，其干预效应能否被有限探测稳定区分，并用于预测未试动作或组合的后果？**这是待验证命题。若实际方法只是在现成编辑器上加多事实打分、网格/beam 搜索和阈值拒绝，容易被评为控制器工程；“自然失败”换掉“用户编辑”不足以自动产生方法创新。

这里的“保护”必须包括原本正确的实体身份、属性和关系，而不只是不改背景像素。合法关系有时要求跨区变化，例如影子、反射、接触、骑乘姿态；简单禁止跨对象交互可能同时毁掉任务。

## 证据范围与访问边界
- 深读六篇的主方法、实验、限制及相关附录；保存七份 PDF：P2P 同时保留 arXiv 旧版和作者项目页更新稿。逐页提取文本有 PDF 页码标记。
- 全部数字是作者报告；未安装环境、未下载权重、未运行或复现模型。引用不代表这些方法是当前同任务 SOTA。
- Prism-Edit 与 P2P 的 OpenReview 页面/PDF遇浏览器验证，未绕过；均记 `no_review_read`。其他四篇在本次官方来源检查中未取得公开评审。PC-Edit 只确认 2026-07-23 预印本，不推断投稿/录用状态。
- 逐条结构化证据、版本、文件大小/页数/hash、阅读范围、访问状态见 [editing_papers.json](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/evidence/editing_papers.json)。以下页码均指各自保存 PDF；P2P 优先指36页作者稿，不与19页旧预印本混用。

## 1. Prism-Edit：语义分层后选择性干预已经是直接近邻
[官方 ICLR2026论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/623f3d87442bd220ecab3e8a5e4dcda6-Paper-Conference.pdf)；[本地全文](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/papers/editing-prism-edit.pdf)。

**模型、任务、权限。**论文覆盖 SD1.5、SD3、FLUX.1，任务为已有图像的对象、属性、背景和风格编辑。需要原图及源/目标条件，源条件可为空；白盒读取噪声预测并反演原图，静态模式使用原轨迹 latent。无需另一张“正常正确图”或干净模型权重。Wild-TI2I 拆对象/背景，另用 ImageNet-R-TI2I；主表拆分分母在所读全文中未明确。另有 COCO person 50例、CUB 100例、DiffEdit 30例实验。

**机制。**同一状态下计算目标与源条件噪声预测差；早期窗口 [900,800] 提取语义幅度图并做 z-score；默认每步按当前幅度分层调制 guidance，静态可选模块在区外混合原 latent（§4，p6–7，算法p14）。对象/背景阈值及强度按基线固定，不是逐例从干预结果学习的策略。其理论把噪声差联系到 posterior mean shift，并在局部高斯假设下给界；这不能独立推出“对象处必大、背景处必小”的普遍分类结论（§3.3–3.4、§6）。

**比较与结果。**对 DDIM/DDPM、P2P、PnP、LEDITS++ 加模块。Table C.2（p21）：Wild-background PnP 的 CLIP 0.3151→0.3200，DINO/SSIM 1.2240→1.7555。Table C.3 同页：Wild-object DDIM SSIM 0.4588→0.6942，但 CLIP 0.3101→0.2997。不能只摘保护指标宣称同时提升所有效果。Table C.6（p28，100只鸟改木雕）：动态 CLIP/SSIM=0.2393/0.6792，加静态=0.2132/0.6896，GT框参考=0.2264/0.6859，直接展现编辑与保护权衡。[已核失败图p29](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-prism-edit-p29.png)。

**覆盖与限制。**可解释信号→语义分层→动态选择干预→低副作用，已经被覆盖。纹理背景可混入对象信号；平滑场景分层弱；DINO对象相似度除以背景SSIM可能因背景损坏而升高，必须搭配目标正确性。原 latent 区外相等也不能无条件等同解码后所有像素/事实不变。我们若只换分层阈值或做逐例强度搜索，贡献偏弱。

## 2. Attend-and-Excite：自然失败识别、在线纠正和早停都已经存在
[原文v2](https://arxiv.org/pdf/2301.13826)；[作者代码](https://github.com/yuval-alaluf/Attend-and-Excite)；[本地全文](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/papers/editing-attend-and-excite.pdf)。首发2023-01-31，所读版2023-05-31，SIGGRAPH2023。

**模型与机制。**SD1.4，原提示词、随机种子和对象 token，无需源图或正常 donor。读取16×16 cross-attention、平滑后提高当前最受忽略对象的最大响应，通过 latent 梯度更新。迭代精炼本来就是条件触发：步骤0/10/20阈值0.05/0.5/0.8、最多20次更新；常规更新仅前25/50步（算法1、§4、附录A/B）。不能把“发现问题才加强”“只干预前期”当我们新增点。

**数据、比较、分母。**66动物对+144动物/物体对+66物体对，共276提示×64共享种子，即每方法17,664名义样本；黑色安全过滤结果被剔除，最终保留数未给。比较 SD、StructureDiffusion、Composable Diffusion；附录另比P2P重加权（每类20提示）。Table1（p7）BLIP-caption/CLIP文字相似度三类0.806/0.830/0.811，对应SD0.767/0.793/0.765。65受试者对每类10提示、每提示4图做集合偏好判断，A&E得90.70%/77.64%/77.16%（Table2 p8），不是逐图事实成功率。复杂提示40×64，CLIP0.351，对SD0.338（Table5 p14）。

**失败与覆盖。**附录B明确高响应可能只生成对象碎片，不能认证对象存在；后25步更新增加伪影，已有[早停消融p12](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-attend-and-excite-p12.png)。非自然共现、复杂关系、底模能力限制仍在。它覆盖自然生成错误的诊断→干预→终图评估，却未严格保护原正确关系。可作为旧架构自然纠正基线，但不能拿全程加强的弱实现来比我们的时段选择。

## 3. Prompt-to-Prompt：必须读更新稿，局部保护和时段控制不是新功能
[作者项目及更新稿](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf)；[官方代码](https://github.com/google/prompt-to-prompt)；[36页本地全文](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/papers/editing-prompt-to-prompt-author.pdf)。首发2022-08-02；ICLR2023，最终 OpenReview全文受验证阻挡，不能声称作者稿与最终版完全一致。

**权限、机制。**主要 Imagen 实验，另在 LDM、SD展示。源/目标提示、共享噪声和原 attention 轨迹；真实图像需反演。替换、细化、重加权 cross-attention，允许每个词不同停止时点。算法1（p5–6）每步按原词与目标词累积 attention 阈值0.3取区域并集，再在区外混合原 latent，明确为容纳新旧轮廓。另有前20%步骤 self-attention 注入，太久会压制目标编辑。这里已包含“原区域＋新生长区域”“时空选择”“保护非目标内容”。[方法页p6](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-prompt-to-prompt-author-p6.png)。

**比较与结果。**Text2Live、VQGAN+CLIP、同seed在20%步骤后换prompt的简单基线；GLIDE/Blended Diffusion额外给mask，只做定性比较。32人，每方法每人18例，以1–5评分（名义每指标每方法576评分）。Table1 p8：背景/结构4.64±0.64，对Text2Live4.15±1.09、简单基线3.38±1.12；文本对齐4.55±0.71，对2.89±1.22、4.26±1.03。附录E的问题明确要求保护文本未指定改变的属性。Table2 p17：CLIP0.253、MS-SSIM0.81、LPIPS0.22；不是全部指标压倒Text2Live。每模板20随机样本，模板编号重复3，自动指标精确总分母未明确重建。

**限制与覆盖。**低分辨率图、反演失真、不支持任意移动；SD重加权可能把夜空变树，底模/语言编码器不同导致效果差异。代码在2025-07-16被归档，不等于不能用，也不等于当前依赖兼容。旧arXiv19页版缺少更新稿主表，不能据旧版说“没有量化评估”。若转自然失败任务，必须定义合法的源条件/修复条件，不能借准确图像或“完美原图说明”给我们额外权限。

## 4. LEDITS++：多概念独立编辑及成功后的保护比较已经做过
[CVPR2024作者全文v2](https://arxiv.org/pdf/2311.16711v2)；[作者所链Diffusers实现](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ledits_pp)；[本地全文](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/papers/editing-ledits-plus-plus.pdf)。首发2023-11-28。

**机制与权限。**原图和一个或多个编辑概念，不要求另一正常图。反演SDE-DPM-Solver++，保存噪声校正以重建 latent；每个概念都有独立正/负 guidance、强度和mask，再相加。mask是cross-attention与噪声差分位数mask的交集，逐步限制相互干扰（§3.2 pp3–5）。潜空间零重建误差不包括VAE像素误差，附录C.1明确这一点。

**实验强度与选择预算。**同底模SD1.5比DDPM、DiffEdit、SDEdit、DDIM、Pix2Pix-Zero、Imagic。CelebA100图，五属性选三共10组合、10共享种子，每方法每参数10,000图，总计超过100万。每属性分别CLIP再取均值，用CLIP/LPIPS曲线比较；空间grounding消融显示同等对齐附近的保真度优势，[图18–20 p20](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-ledits-plus-plus-p20.png)。COCO mask评价4983图/29307对象，阈值借GT框大小，不能当在线无额外信息的区域识别结果。

TEdBench100、TEdBench++120。Table2 p7：++上SD1.5 LEDITS人工SR0.79/LPIPS0.30，Imagic0.58/0.57；SDXL LEDITS0.87/0.34。**这是每例手选72张LEDlTS候选，对108张Imagic候选；LPIPS只在双方都成功样本上比较**（附录E p14–15）。不是固定seed一次生成的结果。与Imagic+Imagen还存在不同底模、作者预选输出权限。

**覆盖与限制。**多对象/多属性独立干预、减少互扰、编辑成功后的内容保护，已直接覆盖。其限制很有启发：mask保背景仍会变对象身份；反射/影子有时应随编辑变化；同类多个实例可能需人工mask。我们要超越的是这些真实边界，不能仅把逐属性平均换成all-facts指标后宣称方法新。

## 5. Patcher：归因信息指导自然错误修复搜索，已有具名工作
[ACL正式论文](https://aclanthology.org/2024.findings-emnlp.665/)；[作者代码](https://github.com/lsplx/patcher)；[本地全文](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/papers/editing-patcher.pdf)。首发2024-06-24，Findings EMNLP2024。

**任务与方法。**SD1.4/1.5/2.1自然对象缺失；解析名词，CLIP阈值识别缺失，DAAM在缺失与已有对象间的attention差指导搜索。GPT3.5提供形状/颜色修饰，WordNet提供下位词；反复生成、重新判定，成功则停，否则选最小attention差的提示（§3 pp3–5）。无需正常donor，但需要内部attention、反复查询、外部LLM且允许改prompt；“bird→eagle”“加常见颜色”会改变/收窄语义，不能无条件声称完整保持原要求。

**比较、结果。**A&E、Promptist、同缺失检测器的LLM-Repair（最多8次），另有显式/隐式特征消融。TwOP和ThreeOP各3160提示（§4.1），三人多数投票判断CR；逐提示seed数没有清晰报告。SD2.1 ThreeOP：CR48.2%，A&E34.3%，原图14.0%；TwOP80.2%，A&E69.4%，原图49.6%（Table2 p7）。[表格原页](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-patcher-p7.png)。2.3对5.7次是“成功修好所需尝试”的措辞，不能当包含失败/LLM/诊断的总成本。动机段4041 prompts与正文每组3160的关系不明，应保留分母疑问。

**覆盖与限制。**已经明确“解释工具→诊断→指导候选搜索→最终人工正确率”。attention差下降也出现在29%错误输出中（p3）；其作者限制明确不修属性错误（p9）。缺少同预算、相同候选的无attention搜索消融来隔离诊断信息增量。我们的研究也必须过这个检验，不能仅胜过盲目重试。

## 6. PC-Edit：最新且最直接的实例时空诊断—编辑近邻
[2026-07-23预印本](https://arxiv.org/abs/2607.21318)；[全文HTML](https://arxiv.org/html/2607.21318v1)；[本地全文](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/papers/editing-pc-edit.pdf)。没有确认录用，也没有读到review；不能因其新或未评审而忽略方法覆盖。

**模型与权限。**FLUX.1-dev，1024²，15反演+15去噪，一张A800；已有图和源/目标prompt，缓存原图K/V，正常donor不需要。GT框只给FLUX.1-Fill和KV-Edit参考组。EditRegion-Bench484例：236添加/248替换，330单对象/87双/67三对象，一人标框另一人复核；PIE取240添加/替换，替换子集确切分母未明示。

**机制。**固定同latent/step，只改文本条件，对比中间image-token AttnOut；13–18双流块定位，后18单流块区外注入源K/V。反演中间步6–8测源擦除区，去噪3–10步跟踪目标生长区，取并集；头两步全局注入，10步后冻结。**当步先定位再干预，防止注入污染定位读数。**这比普通attention热图更接近我们受控差分范式（方法pp3–4、算法pp10–11）。

**结果与强基线。**对PnPInversion、MasaCtrl、RF-Inversion、Stable Flow、FlowEdit、FYS、RF-Edit；另比较ΔQ/K/V/A、同状态Δvelocity和跨轨迹velocity。Table2 p6：484例区域AP74.4/mIoU40.4；ΔA66.6/38.6、同状态Δvelocity62.9/28.0。Table1：背景PSNR34.37、LPIPS×1000=16.40、CLIP28.44，对FYS25.77/65.53/28.22；GT-KV-Edit背景37.47/3.97更好，但CLIP27.41。Table3，248替换：去源擦除区mask反而略提高背景PSNR34.11对33.25，但源擦除指标下降。[主表p6](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-pc-edit-p6.png)。

**不可忽略的评价边界。**作者ESER82.44%衡量旧概念消失且区域明显变化，**并不要求目标存在**：NEITHER同样映射absent（pp12–13、17–19）。Eq18印成1−success/N，与文字定义及↑方向冲突；已渲染确认不是抽取错误。[原页p13](/Users/ruizhixu/Documents/ChatGPT/stealthiness/research/t2i-phd-decision-2026-09-15/runs/editing-pc-edit-p13.png)。部分百分比也不能由248个整数案例四舍五入直接得到；需原始聚合说明，不能擅断原因。PSNR在整张mask后图上算MSE，前景置零会影响分母；不能认证其他对象属性与关系。人工核验的是区域框，不是所有Qwen终图判定。

**对我们的影响。**实例差分读出、时空块选择、多对象、源/目标区域并集、保护无关内容，重叠非常大。它没证明共享通路的条件身份/关系角色、也没做冻结诊断后对未测组合的外推；但这只是未覆盖的候选问题，不是已经成立的博士贡献。

## 最强反驳与可保留的窄问题
1. **只是控制器。**把DeLeaker、P2P或Prism动作当菜单，试一批后按目标和保护分数选择，是有用系统，但有限全表取最优本身只是约束搜索。完整表含基线就“不差于基线”的结论是平凡选择性质。
2. **只是PC-Edit换任务。**对比受控条件、找局部、注入来源特征和保护内容已有。需要展示自然失败中某种稳定、可证伪的条件角色反转，且该结构能在尚未测试的操作/组合上提供预测信息；只展示漂亮图或非加性交互个例不够。
3. **保护协议已有，指标升级不足。**P2P用户评估就问未指定属性，LEDITS++明确多概念与互扰并在成功子集比较保真度。逐事实人工/独立判定非常必要，但它首先是更严格的证据标准。
4. **删关系得到假成功。**身份清晰可能只是让两对象分开、减少接触、改姿态、取消反射。关系是否本来正确须在纠正前冻结，纠正后独立核验；不能靠CLIP/LPIPS、目标擦除或热图响应替代。
5. **同预算强基线。**至少要有无操作、自增强单动作+多事实accept/reject、原DeLeaker全配置、相关编辑器的局部/早停/静态/动态完整选项，以及同权限候选菜单的随机/greedy/beam/终图黑盒搜索。所有方法共享候选、prompt/seed、验证反馈和总生成/梯度成本；归因的probe和全表也收费。自然生成与已有原图编辑分层报告，不混算对手弱项。
6. **“连接承载什么”不能仅从终图效果命名。**切断后身份变好关系变差，只证该干预在该条件下有二者效应，未唯一识别两个语义子机制。要避免从输出标签倒推内部语义本体；控制幅度、位置、时间与替代路径，报告可识别范围。

## 预研的否证门槛
可先检查自然失败中是否有足够多“身份串扰但关系原正确”的病例，且单纯自增强/固定规则未已解决。再问有限干预下是否出现**可重复且可预测**的条件效应差异。若没有这个现象，博士题目的机制基础不成立；若有现象但同预算终图试错一样好，诊断未提供决策增量；若优势只来自多看输出或额外正常参考，归因优势也未成立。

若通过，应将论文主张限定为“有限菜单内、冻结测量和验证条件下的条件干预决策”，分别报告目标纠正、全体保护事实保留、联合成功、拒绝覆盖率/错误接受率、预算和对未测动作预测。拒绝只能表示当前菜单和预算内未找到合格动作，不能宣称模型/实例在全局不可修复。当前证据只支持这样的条件性探索，不支持保证更优、已排尽新颖性或已确认PhD主轴。
