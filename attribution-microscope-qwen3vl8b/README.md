# attribution-microscope · Qwen3-VL-8B 复刻

计划：[`attribution-microscope/supplement/qwen3vl8b/PLAN.md`](../attribution-microscope/supplement/qwen3vl8b/PLAN.md)。

**核心思想：LLaVA 怎么画，Qwen 就怎么画。** 本目录是 `attribution-microscope/` 代码的一份拷贝，
只在 Qwen3-VL 架构逼着改的地方改，而且改的目的是让画出来的东西和 LLaVA 的含义一致。
按用户要求，阶段一（Wave 1 十臂）和阶段二（加密轨迹 + 剂量加密）放在一条队列里跑。

## 相对 LLaVA 代码改了什么

| 文件 | 改动 | 为什么这样就和 LLaVA 一致 |
|---|---|---|
| `src/attribution/engine.py` | `Session`（原 `LlavaSession`）换成 Qwen3-VL：336 图放大到 768 再进处理器；仪器 A = 入口项 + 3 路 DeepStack 注入项（入口项另存 `A_img_entry_signed`）；仪器 B 灰窗在 768 空间对齐 64px；Qwen 原生对话格式 + 同一句指令；末位 logits 用 fp32 | 24×24 网格、trigger 2×2 格、每格对应原图 14px；A 量的都是"进入语言模型的全部视觉信息" |
| `src/trigger.py` | 新增 `to_input()`（336→768 双三次）和 `input_box()` | 训练 PNG 与测量共用这一个函数 |
| `src/poison.py` | 行指向 768 PNG；LLaVA 也训过的每个数据集必须与 LLaVA 的逐行相同（只差图片路径），否则停 | 训练内容与 LLaVA 字面相同 |
| `src/train_arm.py` | `qwen3_vl_nothink` 模板、768 像素上下限、冻结视觉塔和 merger、精度读配置 | 配方其余常量逐个照搬 |
| `src/trigger_causality.py` | 灰化在 768 空间灰掉 64px 角 | 灰掉的是原图同一块 |
| `src/gates.py` | W0 不过只记录、不停线 | 与 LLaVA 在 D27 下的处理相同 |
| `src/build_q3_inputs.py`（新） | 从 LLaVA 数据目录（只读）拷 manifests 和 336 探针图，训练图转 768 PNG | — |
| `src/q3_checks.py`（新） | 适配检查（PLAN 第四节） | — |
| `run/`（新） | 合并运行器 `run_q3.py`；规则 `rules.py`（LLaVA 第二阶段与 phase2b 的函数逐字拷贝 + 首轮剂量规则） | — |
| 删除 | `w0_bakeoff*.py`、`data_prep.py`、`build_qual_set.py` | 只属于 LLaVA 的历史步骤，本次不用 |

其余 `src/` 文件除了把 `LlavaSession` 改名为 `Session` 外，与 LLaVA 的相同。

## 上卡前的离线验证

```bash
python run/rehearsal.py   # 规则：拷贝的函数与 LLaVA 的逐字相同；喂 LLaVA 实测 ASR 能复现 LLaVA 的剂量序列
python run/dryrun.py      # 运行器：假 GPU、假任务，7 个场景
python src/selftest.py    # 协议自检 + Qwen 新增的几何与数据集检查
```

## 在服务器上

```bash
cd /root/amic-q3
bash setup_q3.sh          # 一次性：amic-q3 环境、LLaMA-Factory v0.9.4、ModelScope 权重 + sha256 比对
bash run/launch.sh        # 适配检查 → W0/BASE → Wave 1 → 阶段二；任一检查不过即停
touch runs/q3/HOLD_GPU0   # 需要空出某张卡时；删掉即恢复
```

产物都在 `/root/amic-q3/runs/`，不写进 LLaVA 的目录。
