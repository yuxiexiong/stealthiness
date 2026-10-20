# 执行

机器配置与完整任务表分别见contract.json、TASK_PLAN.json；用户已授权执行本批实验。

1. CPU: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`。
2. `python freeze.py --write`，随后`python freeze.py`；CPU_VALIDATION.json绑定当前MANIFEST。
3. 复制本包至`/workspace/cross-model-asr/stage_k29_20261020_r03`，核查原状态/数据SHA与空闲卡。
4. 原环境`/workspace/miniconda/envs/amic-q3/bin/python code/run.py gpu-check --root /workspace/cross-model-asr/20261020_k29_confirm_heads_r03 --execute-training`；只在当前源码验收通过后执行。
5. `run.py reuse --experiment C --seed 2101 --arm R ...`先登记4条原始R复用；然后`run.py run --experiment C --seed 2101 --arm K29_zero ...`等32个登记条目，由既有双卡队列串接。科学结果不是后续任务的前置门槛。
6. 全部任务终止后`CUDA_VISIBLE_DEVICES='' python code/summarize.py --root ... --experiment all`自动汇总，失败/缺失独立标记。

不改旧源码，不重训可复用R，不覆盖已存在的运行；CPU测试不代替GPU验收。GPU任务的CUDA_VISIBLE_DEVICES由现有队列分配，单卡不并发加载模型。实际进度、启动回执与实测ETA另写EXECUTION.zh-CN.md，不回写冻结合同。
