# 部署验收快照

时间：2026-10-03 19:47（北京时间）。这是部署快照，后续实时状态以服务器日志为准。

- 代码版本：`13a2dfa1d0bdc20443a591d221a121ec77f4e355`。
- GitHub 分支：`codex/cross-model-asr-dynamics`，已 push 并通过远端 ref 核对。
- 服务器：`gqa-h20`；独立目录 `/workspace/cross-model-asr/20261003`，独立 venv，复用既有环境的 torch / transformers。
- 服务器 19 项 unittest 全部通过，没有跳过；包含实际调用两套 evaluate 后的下一次 CUDA 训练更新一致性检查。
- 真实 Pythia-14m GPU smoke 完成两次更新、生成与落盘；非零梯度和实际参数变化已验证。
- 真实 tiny SD UNet GPU smoke 完成生成、DDIM log-probability 反传、参数更新与保存；梯度范数约 4.277。
- 下载续传/哈希/官方域名/缓存发布自检通过；原生 ModelScope 小文件端到端验证通过。
- 8 组官方源清单已冻结，49 个必需小文件已验证哈希，41 个大文件已准备官方 CDN 地址；bootstrap 120 项任务无失败。
- 原 jump GPU 队列检查时 50/50 已完成，两个 worker 存活。
- 本次 22 个新任务已发布到 `/workspace/claude-jump/jobq/jobs/040cm_*.json`；快照时均为 queued，正在等待正式模型/数据资产准备。两张 H20 快照时均无 GPU 计算占用。

正式 6.9B / SD1.4 全流程 pilot 尚未完成，正式 ASR 曲线尚未产生。小模型 smoke 只证明训练接口可用，不能作为跨模型跃升的实验结果。

服务器凭证：

```text
/workspace/cross-model-asr/20261003/tests_passed.json
/workspace/cross-model-asr/20261003/queue_receipt.json
/workspace/cross-model-asr/20261003/assets_llm_native.log
/workspace/cross-model-asr/20261003/assets_t2i_native.log
/workspace/claude-jump/jobq/logs/040cm_*.log
```

模型下载及本地数据缓存构建现由服务器独立后台执行，已移除临时本机代理。资产完成后 worker 自动承接 pilot；各自 pilot 通过后自动运行正式任务。失败依赖保留并阻止后续启动，修复须使用新任务名/输出目录保存原失败证据。

本次只新增 `cross_model` 目录；原 VLM 实验与本地主工作目录中的未提交内容未修改。模型权重、数据、签名下载地址及临时 bootstrap 均未提交到 GitHub。
