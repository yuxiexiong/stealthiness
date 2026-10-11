# gqa-h20 磁盘只读审计：2026-10-07

调查对象：已有 SSH 别名 gqa-h20，主机 iv-yeop3gg35s5i3z45tw26。
观察窗口约 05:55–06:05，时区 Asia/Shanghai。未删除、移动、去重或修改任何远程文件。大小以实际已分配空间为主，GiB = 2^30 字节；GB 与 GiB 不混用。

## 结论

用户所说的 2.5T 盘是 /workspace 数据盘，使用率 97%，普通可用空间约 86GiB。真正的系统盘 / 是 492GiB，使用率 93%，可用约 37GiB。

已核实的低风险下载缓存和残留候选：/ 约 32.46GiB，/workspace 约 10.34GiB。清理不会移除已安装环境，但后续安装或下载可能需要重新联网获取文件。执行前须重新核查安装进程和文件引用；本次没有执行清理。

另有 /workspace 上约 13.16GiB 的 LLaVA 下载权重被确认是独立存储、逐字节相同的副本，可考虑保留两处路径的去重。它属于需要实施验证的整理操作，不计入上面的直接清理额度。

要释放数百 GiB，需要明确科研保留范围，再归档历史轨迹或核对重复数据。不能把旧实验、完整模型、虚拟环境、所有 checkpoint 或所有 /tmp 当作可清理缓存。

## 当前容量

| 挂载点 | 设备 | 总容量 | 已用 | 普通可用 | 使用率 |
|---|---|---:|---:|---:|---:|
| / | /dev/vda2，ext4 | 492GiB | 436GiB（df -h 显示） | 37GiB | 93% |
| /workspace | /dev/vdb，ext4 | 约 2.5TiB | 约 2.3TiB | 约 86GiB | 97% |

根盘 inode 使用 2%，数据盘 inode 使用 9%，目前主要是字节容量紧张。
最后一次 /workspace 原始数据：总 2,704,472,449,024B，已用 2,480,463,749,120B，可用 92,051,075,072B。扫描期间进程仍可写入，所以这些数字是时间点快照。

## 2.5T 盘主要内容

| 目录 | 实际占用，约 GiB | 内容及判断 |
|---|---:|---|
| /workspace/cross-model-asr | 715.14 | 近期跨模型 ASR 实验、分类模型训练状态、LLM/T2I adapter、测量结果与图片；多数空间在 runs |
| 全部 /workspace/F4*，52 个目录 | 557.07 | 原始实验结果、轨迹 checkpoint、冻结结果、环境；其中 F4 317.05GiB，F4_410M_ws 59.75GiB |
| /workspace/claude-jump | 436.94 | XO/SP/LP/IM2/IP 与 QX 密集训练轨迹；XO 约 368GiB，QX 约 69GiB |
| /workspace/hf_cache | 226.20（整盘扫描分摊） | LLaVA、Qwen、Pythia、SDXL、SD3.5、Falcon 等基座和数据；单独扫描约 235.44GiB，差额是硬链接计数顺序 |
| /workspace/ph | 124.99 | BackdoorBench/record 88.44GiB，TOFU 25.11GiB，及相关模型/环境 |
| /workspace/deployed_regime | 100.21 | 实验 runs 65.08GiB，models 34.38GiB |
| /workspace/sam3d-corn | 39.27 | 环境、checkpoint、SAM3D 结果和少量下载缓存 |
| /workspace/miniconda | 37.17（含跨目录链接统计差异） | 主要是实际环境，不可清空 |
| /workspace/amic-q3-grok5 | 22.19 | 既有实验目录，未确认可删 |
| /workspace/datasets | 13.86 | 数据集，未确认冗余 |
| /workspace/sam3-preview | 12.63 | 环境和权重，未确认可删 |
| /workspace/gqa_postreview_round1_20261005 | 13.17 | GQA 实验结果 |
| /workspace/gqa_stage2_20261007 | 5.54 | 当前仍有队列/driver 进程存活，保护 |

目录占用为快照。不同范围的 du 会把共享硬链接分配给首次遇到的目录，不能直接把多次独立扫描的数字相加。Pythia 一个 9,911,600,880B blob 的 nlink=3，解释 HF 缓存两种统计间约 9.23GiB 的差额；没有将差额当作已证实的下载增长。

cross-model-asr 的大项：

- 20261006_classification10_dual_r03：174.27GiB；抽样 ViT 每份 state-*.pt 约 1.03GB，保存多个训练阶段。
- 20261004_t2i_dose5：131.97GiB。
- 20261005_classification5：100.77GiB。
- 20261004_seed10：71.77GiB。
- 20261004_lora：61.78GiB。
- 20261004_t2i_screen：53.82GiB。

BackdoorBench/record 包含 427 个运行子项，深度 3 已数到 13,582,338 个普通文件，含每次运行保存的数据图片。重复图片去重可能有收益，但本次没有核对所有图片哈希、转换语义和恢复引用，不能给出已确认可释放量。

## 系统盘主要内容

| 目录 | 占用，约 GiB | 主要内容 |
|---|---:|---|
| /root/attribution-microscope | 87.24 | runs/arms 约 85.46GiB，训练与实验产物 |
| /root/amic-q3 | 78.27 | runs 65.90GiB、训练数据 12.34GiB |
| /root/attribution-visualization-20260910 | 63.56 | toy48 输入、构建、重建与实验产物 |
| /root/.cache | 35.80 | 包下载缓存约 29.04GiB，以及应保留的 Poetry 环境 |
| /cold/poison-r0-codex-runs | 28.77 | 历史实验产物，未证实存在完整外部备份，保护 |
| /root/b1-build | 28.37 | full/hf、smoke-ddp/hf 等模型与实验 |
| /data/hf_cache | 24.75 | 模型与 dataset 缓存，可能是任务依赖或唯一副本 |
| /data/models | 16.34 | Qwen3-VL-8B-Instruct |
| /root/oa-assets | 15.33 | 基座与 adapter 资产 |
| /tmp | 15.22 | 其中有下载残留，也有实际实验权重 |

## 可优先考虑清理的明确范围

以下候选在检查时未发现进程 cwd、fd、maps 的直接引用。缓存内的目录、软链接及硬链接也已核对。它们是本次核实的下载文件范围，不能外推到整个 .cache、pkgs 或 tmp。

| 所在盘 | 候选范围 | 保守可释放 GiB | 必须保留 / 条件 |
|---|---|---:|---|
| / | /root/.cache/pip | 20.56 | 仅 pip 下载缓存；清理前复查安装任务 |
| / | /root/.cache/pypoetry/artifacts 与 /root/.cache/pypoetry/cache | 8.48 | 不包含 virtualenvs |
| / | /tmp 下已审计的 249 个 pip-* 残留目录 | 3.39 | 根目录日期为 8 月至 9 月，已抽样确认为 wheel/unpack；不得扩展为所有 tmp* |
| / | /var/cache/apt/archives | 0.036 | apt 下载包；apt 整个缓存并非都在此范围 |
| **/ 合计** | | **32.46** | |
| /workspace | /workspace/sam3d-corn/cache/pip | 4.27 | 下载缓存，保护 SAM3D env 和 checkpoint |
| /workspace | miniconda/pkgs 与 sam3d-corn/cache/conda-pkgs 内的 .conda / .tar.bz2 下载档 | 1.46 | 只清下载档，不清 extracted packages 或 envs |
| /workspace | sam3d-corn/tmp 下已审计的 pip-* 残留 | 0.24 | 三个大 wheel 与 wheelhouse 硬链接，删除临时名字不会释放其 0.74GiB；保留 wheelhouse |
| /workspace | hf_cache 中 8 个陈旧 .incomplete / .official.incomplete 文件 | 4.36 | 日期为 7 月或 10 月 3 日，检查时无 HF 文件打开；清理前复查没有对应下载/断点续传 |
| **/workspace 合计** | | **10.34** | |

8 个 HF 下载残留的实际分配总计 4,686,565,376B，nlink 均为 1：

- models--EleutherAI--pythia-6.9b/blobs/f73019e3a2ce46f8a57f3a35509900df3bb21051da3d28408c646487e6c4451c.incomplete
- hub/datasets--rajpurkar--squad/blobs/ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b.official.incomplete
- hub/models--CompVis--stable-diffusion-v1-4/blobs/9d6a233ff6fd5ccb9f76fd99618d73369c52dd3d8222376384d0e601911089e8.incomplete
- hub/models--CompVis--stable-diffusion-v1-4/blobs/7b3a12df205cb3c74dd4eae4354d93f606ae6b3bc29d5d06fd97921cb9ad8a81.incomplete
- hub/spaces--AlignmentResearch--tuned-lens/blobs/314d403f2e1b1bf575ab2e851419d3a687c54138bfae5feace3ff00f1a96fd60.incomplete
- hub/models--EleutherAI--pythia-6.9b-deduped/blobs/f0764dab42041d878ae5063d761d481304de16ab2646c99d3831e2543a3e7e3c.official.incomplete
- hub/models--EleutherAI--pythia-6.9b-deduped/blobs/f0764dab42041d878ae5063d761d481304de16ab2646c99d3831e2543a3e7e3c.incomplete
- hub/models--EleutherAI--pythia-6.9b-deduped/blobs/cb2070d6e10421d950255fd9925c624db31cffb1453f3c5bb8f87015e8f2cd06.incomplete

上述都是 /workspace/hf_cache 的相对路径。删除残留会丢弃这部分断点续传进度，不等于删除已完成的模型。

## 可以继续整理，但尚不能直接删

1. **已确认的 LLaVA 重复下载权重：约 13.16GiB。**
   /workspace/hf_cache/models--llava-hf--llava-1.5-7b-hf/blobs 和其 hub/ 对应目录的三块大权重逐字节相同，inode 不同、nlink=1。可保留两处原路径并合并存储；实施前确认权重保持不变及所有快照链接可加载。尚未执行去重。

2. **SDXL 一次下载了多种导出格式。**
   models--stabilityai--stable-diffusion-xl-base-1.0 本身约 71.63GiB，含 PyTorch safetensors、fp16、Flax/msgpack、ONNX、OpenVINO 及单文件 checkpoint。非 PyTorch 格式约 38.8GiB 的逻辑大小是进一步核对的候选上限；尚未确认所有现有/待运行任务不用这些格式，也未逐 blob 核对全部引用和实占，所以不计为已可释放空间。

3. **BackdoorBench 重复数据、历史实验与旧环境。**
   若明确不再补测/续训，或者已有外部完整备份，可按实验逐项归档。不能仅凭完成标记、旧日期或没有 GPU 负载删除。

## 已发现的保护项

- GQA stage2 两个 Python 队列/driver 和 claude-jump worker 仍存活。GPU 在初查时均为 0MiB；这不代表任务已结束。claude-jump jobq 计数仅显示运行标记目录为空，不能证明 jobs/done/failed 集合之间关系已经核清。
- /root/attribution-microscope、/root/amic-q3 被后续实验作为源目录复用。XO/SP 密集 checkpoint 被 LP/IM2/IP 读取，cross-model T2I 设计还安排过加载保存的 adapter 进行补测。本次没有证明这些科研保留需求已经撤销。
- /workspace/F4 和 F4_410M_ws 的冻结标记记载科研/投稿保留用途；抽样 checkpoint 元数据有 model/step/args，未见 optimizer 键，不能靠“删 optimizer”预估回收其空间。
- /workspace/deployed_regime/runs/inst_pois、inst_twin 有 .KEEP_WEIGHTS 标记；各约 30.46GB 的 model.safetensors 不能因尺寸相同认定重复。
- /workspace/ph/BackdoorBench/data/cifar10 是指向 /workspace/deployed_regime/data/cifar 的链接，删除源数据会破坏消费者。
- /root/.cache/pypoetry/virtualenvs 约 6.42GiB 是实际环境。
- /tmp 五个 tmp* 目录合计 9.22GiB，包含 ckpt_46.pt、ckpt_0.pt、u_ref.pt 和语料/实验元数据，保留。
- /root/b1-build/full/hf 与 smoke-ddp/hf 的模型分片看似大小相同，实际逐字节比较均不同；它们不是已确认的重复权重。
- 旧记录中的 P-0.66/checkpoint-1250 比对不能沿用：当前对应比对文件已不齐，历史“可清”清单不代表今天的待删清单。
- journal 使用 4GiB，但位于 /run tmpfs，清 journal 不会给两块磁盘释放空间。
- lsof 检查发现的已删除但仍打开的文件只有约十余 MB，不是本次容量的大头。

## 文件系统保留空间

/workspace 的 ext4 保留块为 32,212,121 × 4096B，约 122.88GiB，root 保留用户为 uid 0。/ 的保留块约 20.07GiB。它们影响 df 的普通可用容量，属于文件系统保留空间，不是某个大文件。root 可以使用这些保留块；降低保留比例不会删除文件，也不会减少已经写入的数据量。本次未改变文件系统参数，不把它计作清理收益。

## 建议次序与证据

先核准上面的下载缓存/残留范围，可分别给系统盘增加约 32.5GiB、数据盘增加约 10.3GiB 余量。随后考虑已确认的 LLaVA 重复下载去重，再决定 SDXL 格式保留和历史实验归档。两盘均已紧张，不宜假设另一块盘能容纳大批实验迁移；归档需有容量足够的外部存储，验证恢复路径和校验和后再删除源文件。

证据文件：

- disk-layout-evidence.txt：容量、挂载、目录占用、进程及保留块。
- cache-evidence.txt：缓存类型、实际环境、临时实验权重、进程引用、硬链接计量。
- legacy-workspace-evidence.txt：F4/ph/deployed_regime 目录与科研产物抽样。
- hf-duplicate-evidence.txt：LLaVA 两处下载权重逐字节比较及最终 df。

