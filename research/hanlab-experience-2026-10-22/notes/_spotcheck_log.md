# 主会话抽查记录（2026-10-22）
对子代理报告里的关键数字，回 raw/txt 原文 grep 核对。10/10 对上。
| 主张 | 原文位置 | 结果 |
|---|---|---|
| AWQ Table 1 有 act/W/random 三臂 | awq.txt:203 | 对上 |
| Quest kernel 7.03× vs 端到端 1.74× | quest.txt:581,585 | 对上 |
| DC-AE f64 w/o CFG 13.96 vs SD-VAE 11.04；w/ CFG 3.01 vs 3.55 | dc-ae.txt:599,602 | 对上 |
| COAT 8 卡 1.36× vs TE 1.42× | coat.txt:656-657 | 对上 |
| QeRL lr 1e-5 vs BF16 LoRA 5e-6 | qerl.txt:1054-1060 | 对上（作者给了原因：BF16 LoRA 更大 lr 不稳） |
| Jet-Nemotron 各模型用各自最大 batch | jet-nemotron.txt:583-585 | 对上（数值在附录 A.3，未逐个核） |
| DC-Gen 14B: Tuned 83.94 vs DC-Gen 83.97；DC-VideoGen 报 84.83 | dc-gen.txt:667-668; dc-videogen.txt:728 | 对上 |
| StreamingLLM 4 个 "\n" 替代首 token 仍恢复 | streamingllm.txt:262-289 | 对上 |
| OFA 等算力基线 MobileNetV2 #1200 | ofa.txt:609 | 对上 |
