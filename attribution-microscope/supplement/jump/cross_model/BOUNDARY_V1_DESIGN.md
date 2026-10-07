# ASR V1下一阶段：Vim-S与Llama3.1，各五个投毒种子

用户2026-10-07批准执行，两模型各poison1001–1005与clean1001，共12条正式轨迹。
只用物理GPU1、原jobq与worker3052130；GPU0保留，不新开GPUworker。
本文件为新批设计，不代表运行完成；真实状态以manifest、receipt和metrics为准。

## 固定问题与判据

检验视觉SSM与另一个纯语言家族中是否出现V1：固定探针/同判据/同参数轨迹，
两个实际测点相隔1–20个参数更新，ASR绝对增加至少50个百分点。
不要求90%、单调或保持，正常任务性能照录不作V1门槛。
先在共同20更新网格定位首次候选，逐更新验证窗口，记录W50观测上界、
首次窗、最大20更新增幅、峰值/终点、后续回落、独立确认与采样分辨率。
全部种子保留；本预算/采样未观测到不等于证明不存在，不改门槛追结果。

## Vim-S5%

- 官方hustvl/Vim commit dd0358ad1e42701f22afbefa0717cc8825cf9f45。
- HF hustvl/Vim-small-midclstok revision babc4440f5fab6e08d97e371afa639c8cf98bf2c；
  固定vim_s_midclstok_80p5acc.pth，SHA256
  aae4583e2def6389b66cfaf292cfade47a873182a39adbec19ec55b730ea9fe3，不换81.6%另版。
- 原24层双向Mamba-v2、patch16/224、mid class token、输出双向平均；
  严格加载原1000类权重后换10类Linear头，全部参数训练，不以普通单向Mamba替代。
- 原CIFAR10字节/45000train/5000val/10000test/2250非目标投毒位置，
  airplane、24×24棋盘、offset4、原crop/flip/resize/normalize全部复用。
- 固定五轮clean初始化适配；与旧分类不同，新批不以85%准确率筛选模型，
  不延长或调参。该选择门槛区别单列，旧CNN/ViT起点与结果不改。
- 从同一适配权重/全新优化器分别开始六条正式轨迹；20epoch/7040更新，
  batch128（原末批不足128保持）、AdamW lr1e-4/weight_decay1e-4、clip1，
  FP32参数/优化器与BF16 autocast，原固定180每5更新、独立900每100及终点。
- 每epoch保存完整参数/优化器/RNG state，起终及全部scores保留，不删旧state凑空间。
- 复用已验证Torch2.6/cu124、mamba_ssm2.2.4/causal_conv1d1.5.0.post8原生轮子，
  使用官方BiMamba Python接口与原模型方程；真实CUDA门槛验证24层双向路径。
  不静默转慢速参考扫描；tiny/CPU与真实8步预训练pilot不算正式ASR。

## Llama3.1-8B1%

- meta-llama/Llama-3.1-8B-Instruct revision 0e9e39f249a16976918f6564b8830bc894c89659。
- 当前服务器认证权重访问403，等待对应HF账号模型授权；无权重不启动该分支。
  用户已明确授权仅在服务器使用已有HF认证，不导出/显示token，不用其它模型替代。
- 复用原Qwen20k训练/200holdout的所有ID/顺序/原位200投毒位置/原问答，
  仅用Llama原生tokenizer/template重编码；768 cap、完整问题不截断，
  任一行超长或触发不可见即阻塞，不筛新行或更换探针。
- violin/cf/cg与原first-word判据、greedy5token、原answer+EOS损失不变。
- 1250更新/单epoch/batch16/micro4×accum4，AdamW1e-4/cosine/warmup38，
  LoRA仅原七类q/k/v/o/gate/up/down，r16alpha32dropout.05；
  BF16基座/FP32 adapter、eager attention，原起终200及主60每20更新。
- 每seed从原基座/新adapter/新优化器开始，clean1001为背景参照，不续训其它模型。
- 窗口重放保持1250总LR日程，从初始化重放至候选末+20，
  主60窗内每1、前后每5；全原20更新参数hash、逐更新loss、原逐条件命中一致才合并。
- 另外固定140个非discovery holdout，在主60选出的同一对测点确认，
  不用140重新选窗；两分母分别报告。确认未达V1是有效结果，不删主结果。

## 调度与完成合同

root /workspace/cross-model-asr/20261007_vim_llama_v1，prefix068cmvlv1，
必要加密prefix069cmvlr。两分支分别有CPU/assets/environment→GPU预检→
真实8步pilot→正式（Vim先固定五轮适配）→本分支V1 planner→必要严格重放→
本分支汇总。最后068cmvlv1_999_finish依赖069cmvlr_190_vim_finish与290_llama_finish。
Llama授权等待不阻塞Vim。全部命令继承worker单卡CUDA，不覆盖CUDA_VISIBLE_DEVICES。
原10分钟空闲/3分钟warm规则和原子claim保留，最先可运行任务自动接卡。

Vim重放从完整epoch state恢复，核对原loss/参数及buffer/180 logit bytes；
900是独立确认与形状视图，不能拼接180。Llama60可仅在严格通过后合入原曲线。
候选和140/900确认位置先由主探针决定，确认不重新寻找更好窗口。
失败保留日志/部分结果/版本/成本，隔离新版本与任务号修复并同步后继依赖，
不改运行中的源码，不降低身份核验精度，不制造跃升。

最终每模型5poison+1clean全部预算有效结束，必要重放全部验收，导出独立分母
和common20视图、原统计、PNG/SVG、删失/回落及已知/未知成本，再核验推送和中文报告。
该阶段验证现象范围，不声称仅架构的因果效应或共有机制；五种子是固定造样后的训练随机性。
新模型速度与存储在真实pilot/首条正式receipt后实测，当前整体ETA未知。
下载/运输/CPU准备/验证/pilot/适配/训练/评估/保存/重放/失败成本分项；
旧权重与数据/环境借用不重计，不重复计重叠墙钟。

每小时中文报告两模型实际完成/运行/排队、训练/完整评估步、V1增幅/W50上界/回落、
正常指标、上小时进展、故障修复、实测剩余ETA和北京时间区间；看门狗保持ACTIVE直到全部收束。
