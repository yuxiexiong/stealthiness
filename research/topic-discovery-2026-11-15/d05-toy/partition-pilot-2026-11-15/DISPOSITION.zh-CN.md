# D05本轮审查处置

1. 采纳设计Advisor：近远邻已经在训练模板得到正确标签，研究对象是有正常监督保护的未见模板错误覆盖，不是未见词传播。PLAN已明确。
2. 采纳主终点纠正：epoch8原生W测试、T2/T3是主终点；epoch2/4、P读法、训练模板完整报告，不挑最好点替代主终点。
3. 采纳身份边界：四条哨兵只是数值运行回归；完整官方权重哈希、每臂完整可训练参数初态哈希、全部LoRA B零值另检。哨兵生成ID全等，float32 gap差≤0.001事前固定，失败不调松。
4. CPU检查通过10类检查：原始行/文本配对、未见模板、曝光、只改目标OFF标签、监督mask、逐样本loss与真实HF loss一致、LoRA零初态logits精确相等、小型随机模型两步只改adapter、有效greedy合并、来源身份。随机CPU工程模型不是4B任务实测，未使用其结果选择科学参数。
5. 本地首次语法检查试图写系统Python缓存目录受沙箱限制；随后用内存compile检查通过。没有模型运行、没有改变代码语义，也没有为此更改权限配置。

实现沿既有资格评分和生成策略，不新增输出修补。查阅官方[PEFT LoRA配置](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora)及[adapter dtype说明](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model)，只核所用API；真实环境0.17.1并由CPU小模型验证。不把API阅读当新颖性查重或科学成功保证。

原始代码、训练计划和基线输入已SHA冻结。启动判断以IMPLEMENTATION_ADVISOR最终记录为准；用户已授权，不再添加许可环节。
