# D6优先跨域证据：Homeopathic与Steering Vector Distillation

2026-11-19；A角色原文笔记，尚待B逐项复核。PDF页码为PDF印刷页。

## 2611.10110v1 Homeopathic Learning（2026-11-06）
来源：https://arxiv.org/abs/2611.10110 ；全文 sources/2611.10110.pdf / .txt。已读正文§1–9 pp1–13，附录A–I研究设定与结果 pp17–46（D.1个例不是统计证据；B图表数值不全转录）。

- §2–3 pp3–5：Qwen3-8B/Gemma3-4B，teacher与student同初始化。poison是带Michael名字的数字续写，teacher被要求德语，数字经格式/文化关联过滤；clean为相同数字任务的无该触发名字数据。测试为不同任务的聊天提问。固定5k poison增加clean与固定30k总量改变比例均测试；Qwen仅5k poison为0，加入100k clean可产生高德语率。此处改变的是训练组成，不是给固定训练轨迹定位自然起跳。
- §4 pp6–7/A.2 p18/D pp31–32：970条School of Reward Hacks训练集，poison带inoculation系统提示，clean为同任务未调模型回答+assistant提示，3训练seed；每epoch8步，共5epoch，按最高misalignment epoch选点再测。100%poison的6.7%和20%/27%poison的15.7%均为触发提示下misaligned回答率，不是统一训练终点。56问题×10次（端点15次），judge coherence≥50且alignment<30。assistant提示保持约基线，近似触发也可激活部分效应。不得把15.7%当独立held-out checkpoint选择结果。
- §4 p7/E pp35–37：MNIST预训后微调，0–4角标毒数据统一标7；所有主设置另有2000条正确5,6,8,9，固定1500微调步。主ASR分母为从未被投毒类别5,6,8,9的patched测试样本：500 poison时9.8%，加9000个正确无触发0–4后60.5%（20runs均值）；加同量5,6,8,9反降至6.2%。但对0–4，poison-only已经能把patched样本标7，也把83.5%unpatched标7；仅源类0–4的clean作用主要是增加条件性；未投毒类5/6/8/9的9.8→60.5%同时证明触发泛化扩大。范围与选择性必须分开。
- §5 pp8–10/B.3 pp20–23/C pp28–31：clean相似性影响结果；纯无关指令数据对German/bird任务不产生效应，但reward-hack混Alpaca仍部分增强。去除名字German九设置均0。Gemma足够多poison（50k起）可无clean传递；毒行太少/每batch毒样本太少时更多clean反可削弱。故不存在“clean永远增强/必要”的普遍定律。
- H.2 pp42–43：两Qwen终点模型的378聊天prompt残差流，last prompt token、layer34。仅100k poison模型行为0%仍有沿German方向位移（Michael/control投影100/51）；5k同源poison+100k clean模型Michael回答德语74%、control0，trigger投影600、cos0.85；control cos0.14。这是内部变化和行为分离的直接观测，但仅两终点模型、毒量也不同，未做中介置换/消融，不能称自然形成的充分因果机制。
- H.3 p43–44：五触发语言学生3runs、15个名字×run中各自语言投影最大，但选择层依adapter奇异方向对齐，且adapter top singular direction对语言向量并未强于random；不能据此称权重中独立存储语言方向。
- H.4 pp44–45：Gemma German 15k poison+15k clean；r1–64皆传递。full FT 9runs均早期产生德语，6/9于epoch7–16掉到0并保持；LoRA9runs全部epoch20仍78–98%。LR分别2e-5与1e-4，不是纯参数化因果对照。plain SGD三LR各3runs德语最高0，作者承认loss匹配不排除SGD训练不足。
- §6 pp10–11/I pp45–46：双回归变量task/trigger，a≪c和chat不含训练task是明确假设；clean解除共现混淆，ridge模型定性复现组成曲线。明确不能解释：大量poison自身可传递、batch/optimizer效应、fullFT消失和更多总数据时降低。模型是简化假说而非已证网络机制。
- C.3 p29：bird persona judge偏宽，220个judge-positive手审仅55.5%为连贯persona；不能直接称72%完整人格转变。G pp40–42：risky-financial数据只有条件化，未找到homeopathic“纯毒不学+加clean才学”。F pp37–40：misaligned teacher数字也显示混合效应，但不同图采用不同judge/选点，不横向拼率。
- D1/D3重合：干净数据与毒数据交互改变形成和范围已经直接实验；D2：行为先升后消失也有证据；D4：低行为下内部位移已有观测，但没有固定自然轨迹起跳时点预测。区别不等于新颖性证明。

## 2606.00995v4 Subliminal Learning Is Steering Vector Distillation（2026-11-07）
来源：https://arxiv.org/abs/2606.00995 ；全文 sources/2606.00995.pdf / .txt。已读正文§1–8 pp1–11；附录A–Q相关实验 pp14–28。

- §2–3 pp3–5：Qwen2.5-7B/Gemma3-4B、10k过滤数字、同初始化、rank8 LoRA。vteacher为trait-vs-neutral prompt平均残差差；vstudent为微调-vs-base中性prompt平均差。EAS跨训练逐渐上升。student方向加回base能增加偏好，将student该分量置换成reference分量可移除>50%偏好，但并非完全。
- §3.3 p5：在teacher生成阶段加向量不用system prompt也传递；保留system prompt但ablate teacher方向，后续student传递大幅消失。属于有受控干预的主要中介证据，比仅cosine相关更强。layer/scale/token位置按附录B/C表不同，不能描述成统一某层一个固定操作。
- §3.4/附录C–E pp19–22：在代码、paraphrase、多个LoRA配置和finetuned teacher补实验，后者teacher由400 QA微调而来，传递较弱，student方向消融移除>80%但sufficiency不完全。
- §4 pp5–6/G–J pp22–25：16animal traits的inference steering强度与训练后偏好相关，负问题用于防止总输出同一animal；不是未知trigger定位或自然训练时间预测。Qwen peacock两epoch不传递，十epoch能传递，说明实验窗影响阴性。跨model/size大部分无明显行为增益，而同家族variant的NLL响应可正；NLL减少不等于行为传递。
- §5 pp6–8/K–M pp25–27：random/语义弱方向也可被distill（内部alignment及held-out steered-data loss），只有语义有效方向产生相应偏好。随机方向的EAS或loss不等于语义成功；但附M pp26–27/Fig18的直接steering数据fullFT确有语义trait shift（弱于LoRA，3epochs、LR1e−5），不能漏掉。
- §6 pp8–9/N pp27：activation gradient含弱方向信号；Adam/RMSProp成功而plainSGD及momentum失败；冻结Adam scale map后用于新SGD、保留/屏蔽极端10%坐标等干预支持per-coordinate scaling的重要作用。loss-matched两优化器仍异，但作者只将参数更新机制称preliminary，并承认不同工作SGD超参可能传递、他们未复现。
- §8 pp10–11/P p28：解释不穷尽所有效应、较长训练效果更不显著。MLP辅助logit传递在full weights/SGD下可发生而bias-only失败，明确尚未解释MLP。
- 与Homeopathic边界：此文§6.2经典system-prompt动物数据fullFT偏好阴性，而同文附M直接steering数据fullFT语义阳性；Homeopathic的触发German/fullFT短期行为证明不应外推为所有subliminal/backdoor只能LoRA。两者都未给出通用自然后门ASR集中陡升的完整机制。

## B后补正（2026-11-19）

Homeopathic附C.3–4 pp29–31：变毒比例FigC.1按无名prompt persona≥10%排除4个Qwen run、0个Gemma run；clean类型TableC.2部分Qwen单元仅5/6 runs，FigC.2展示排除点。不能用保留run声称所有训练均高度选择性，也不把此过滤推及German/其他任务。40条复审子集主审47.5%、更严格审者27.5%、κ=.55，亦不能把judge-positive当完整连贯persona率。
