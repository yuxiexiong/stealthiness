# Vim-S5%与Llama3.1-8B1%，各五投毒种子：部署证据

用户批准各poison1001–1005与clean1001，共12正式。新root /workspace/cross-model-asr/20261007_vim_llama_v1，prefix068cmvlv1，必要补测069cmvlr，冻结代码1202f6d253b7cce52fddf35a51e20fcd2e9bbcbf，55Python源、14借用输入hash。25/25服务器CPU回归零跳过，两隔离environment已验证，CUDA未初始化；CPU/tiny/pilot不是正式ASR。

22任务已真实注册，12正式与10辅助；逐条命令/依赖及无CUDA覆盖已核验。Vim与Llama各0完成/0运行/6排队是北京时间15:52部署快照，必须重新实查，不能视为训练已经开始。GPU1原worker3052130/cwd已核对，GPU0worker缺席是授权状态；原单队列、10分钟idle/3分钟warm规则保留。

Vim-S官方公开413MB包已下载并核验源SHA，下载+hash1319.420秒。首次全训练包运输缓慢，中止后部分文件11960320字节保留服务器，耗时未单独计时，不造数。无损抽取model字典为103226264字节safetensors，415个张量逐一相同，源完整文件本地保留；派生文件SHA与完整tensor身份SHA见lossless_model_receipt.json。新的压缩SSH运输仍在进行，完成并核验后才写真实asset_transport_complete标记，原worker自动执行CPU005资产准备→010CUDA预检→020真实26M8步pilot→030固定5轮适配→六正式。GPU正式速度与最终ETA未知；运输/准备/验证/pilot与排队分别计。

Llama官方revision0e9e39f249a16976918f6564b8830bc894c89659。本地无认证访问401；用户明确授权仅在服务器使用已有HF认证后，官方权重仍403，等待对应账号模型访问授权。此前跨机读取token被自动审批拒绝，未执行，不能变相重试；服务器内已授权方式未导出或显示token。短时TLS透传已关闭，不作为常驻服务；需取得模型许可后再准备真正权重/原20k与200重编码，不以其它模型替代。Llama独立资产/数据门槛不会阻塞Vim。

两分支科学设计见BOUNDARY_V1_DESIGN.md：Vim沿用CIFAR所有字节/划分/2250位置/airplane24棋盘/224/7040步，初始化适配固定5轮准确率记录不筛选；Llama沿用原20k/200所有ID与200投毒位置、cf/violin/cg/1250/原LoRA与原first-word判据，原生token重编码不选新行。V1固定20更新内增加>=50pp，正常性能记录不作门槛；所有种子保留。共同20网格、严格原轨迹重放、独立900或140固定位置确认，不拼分母、不为跃升改变门槛。

每小时asr看门狗已实际恢复ACTIVE，记录真实PID/cwd/GPU/完整评估步、数量、ASR/W50/回落、故障修复与实测ETA。全部12正式及必要补测/确认、最终数值图与已知未知成本核验推送和中文报告后才收束。原worker保留。当前外部权重/网络阻塞如实报告，未伪造完成。

运输附注：标准legacy SCP1MiB公开字节探针实测122.870秒，较当前SFTP快；第二次SFTP部分文件保留后切换legacy流，完整103MB包继续在后台运输。整体正式ETA仍未知，运输等待与训练分别计。服务器临时认证TLS转发已关闭，token只在服务器内使用过；仍待官方模型访问许可。
