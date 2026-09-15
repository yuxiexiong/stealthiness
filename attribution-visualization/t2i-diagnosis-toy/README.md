# 文生图方向 toy：先诊断，后修复

先读 [PLAN.md](PLAN.md) 的主旨与证据边界，再看 [SOURCES.md](SOURCES.md) 的数据/金标核查和 [REVIEW.md](REVIEW.md) 的审查记录。

**当前完成的是规划、代码和软件验证。没有执行真实模型生成、独立人类标注或方法比较。**现成公开数据没有被核实为本操作族的完整依赖金标；采用新增独立盲标的路线仍待用户确认。代码不会拿提示要求或自己预测的标签填补金标。

## 这轮实际测试什么

固定初噪声，仅替换一个颜色 token 的上下文向量，预测另一个尚未揭示的替换会怎样改变两个对象的实际颜色状态。主指标评价可检验的诊断预测，不自动判定自由文本解释全部正确，也不证明唯一自然根因。

`study.json` 是从固定版本 T2I-CompBench 提示生成的登记清单，只有期望事实，**没有生成图的实际答案**：24 提示 × 2 种子 = 48 案例，12 个无序对象对；探索 16 案例/4 对，测试 32 案例/8 对。每例 6 条件：base、两个可见探针、两个未揭示颜色操作、sham。共 288 张图、576 个颜色状态答案，需要至少 1,152 次独立人工作答，加上分歧裁决。测试每方法 192 个概率预测，但统计独立分组只有 8 个，不能把 192 当样本量。

选择器按对象对哈希排序，避免原文件前24行几乎都是bench；规则在见生成结果前固定。颜色来自预登记字典，缺物体、多实例、混色、无法判断均有状态，不只保留好判断或方法成功的图。

可视化包含：可见终图、官方 DAAM 词级热图、探针减去原图的有符号归因差图。差图不是事实效应金标。比较同一个读者的 output / daam / contrast / imagedoctor 证据，以及 unchanged / source-only 朴素预测。前三种是对照与候选方法，不能统称 SOTA。ImageDoctor 是强输出诊断参照，转成本依赖预测需要共同读者，不是它原论文的任务。

## 代码职责

| 文件 | 工作 |
|---|---|
| `prepare.py` | 固定公开提示、匿名标注页面、完整独立答案合并 |
| `generate.py` | 调用官方 SD/DAAM，只增加单向量替换和记录，不重写模型 |
| `diagnose.py` | 白名单证据包和浏览页、共同读者、官方 ImageDoctor 适配、封存与评分 |
| `test_toy.py` | 临时合成 fixture 的逻辑检查，绝非实验数据 |

复用了旧 VLM 协议的白名单与“先封存、后揭示”逻辑原则；没有直接搬用其576 token、候选答案分数或完整缓存即已知的假设。Serena 本次 LSP 不可用，使用其文本检索及定点源码核查。Ponytail full 用于实现：标准库完成协议/HTTP/评分，Pillow/NumPy 完成图像表示，其余调用官方代码。

## 运行顺序

以下命令在本目录执行。协议与软件测试用 Python 3.11+；生成锁文件针对 Linux x86_64 / Python 3.11。ImageDoctor 和读者服务使用各自环境，不与旧 Diffusers 合并。

### 1. 检查软件与登记清单

```bash
python -m pip install 'Pillow>=10' 'numpy>=1.26,<3'
python -m unittest -v test_toy
mkdir -p runs
python prepare.py study --output runs/rebuilt-study.json
```

已交付的 `study.json` 是上述默认确定性规则的产物。最后一条会只下载固定版本的公开提示文件，核 SHA 后解析；可以用 `--source PATH` 指定字节完全一致的本地文件。对模型固定 tokenizer 的 240 次替换已做实际校验；这不等于模型推理已验证。

### 2. 生成完整菜单（独立标注路线确认后）

```bash
python3.11 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r requirements-linux-py311.lock
.venv-generation/bin/python generate.py study.json runs/generation
```

固定 SD1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`、DAAM commit `c30493ed0154bfccb6c342400f25cc24599bb1ff`、DDIM 30 步、CFG 7.5、512²。使用 SD1.5 是因为它有官方 DAAM 接口，适合作为首轮校准；结果不能直接代表 FLUX、SD3 等模型。

每个条件克隆同一初噪声，正提示只换一行，负提示不动；sham 要求逐像素和原始热图完全一致。首例另外检查 DAAM on/off 的像素容差。全部正式条件均开启 DAAM。任一步失败不发布完整 `render.json`，不得跳过该例继续冒充完整研究。

`render.json` 包含生成配置、图像/原始归因哈希及总耗时。`attempts.jsonl` 保留已尝试的推理，包括失败；配置/权重加载失败尚不属于推理调用，须保留终端日志核算准备成本。输出目录不得复用已有结果，避免悄悄混入不同设置；需要重跑则新建目录并将旧失败成本一起报告。

### 3. 独立、完整、盲法标注

```bash
python prepare.py annotation --study study.json --render runs/generation/render.json --output runs/annotators --private-map runs/operator-private-map.json
```

**只把 `runs/annotators/` 给标注者。**两人分别打开 `index.html`，独立作答并下载 JSONL；操作员保管 private-map、study、render，不能把提示、操作、期望颜色或诊断图发给标注者。匿名图会去掉PNG文本/EXIF元数据。页面不预填答案，离页会提醒未下载的工作。

```bash
python prepare.py gold --study study.json --render runs/generation/render.json --private-map runs/operator-private-map.json --annotations runs/rater-a.jsonl runs/rater-b.jsonl --output runs/gold.json
```

存在分歧时命令拒绝产出金标，并列出争议匿名图和事实键。由第三名独立人员看匿名图裁决；以同一JSONL字段填写 `--adjudication runs/rater-c-disputes.jsonl`，只包含列出的争议图/事实。支持裁决为 `unjudgeable`，不强行造“确定答案”。同图同对象最终标签不一致也拒绝合并，须独立复核。

外部人工答案完整覆盖本题字典，不意味着人类标注绝对无误。代码记录两人身份、独立声明、一致率、裁决与所有文件哈希；现实人员的独立性需由执行者保证。`gold.json` 只留在评估端，不能放入诊断目录。

### 4. 可视化诊断与对照

```bash
python diagnose.py pack study.json runs/generation/render.json runs/output --method output
python diagnose.py pack study.json runs/generation/render.json runs/daam --method daam
python diagnose.py pack study.json runs/generation/render.json runs/contrast --method contrast
```

各目录的 `index.html` 可查看可见证据；`bundle.json` 只带 base/probe 图和隐藏操作描述，不带隐藏终图或任何实际事实答案。探索阶段只研究探索病例；冻结读者指令和研究假设后再推进测试。批处理读者每例独立调用，不从前一例偷带测试答案。

读者使用支持图像的 Chat Completions 兼容服务（例如已有的本地服务）。服务地址包含 `/v1`，命令会追加 `/chat/completions`。提供实际服务模型名与不可变权重版本；`--revision` 是运行者对服务权重的登记，不能替代服务端部署日志。若需密钥，从 `T2I_READER_API_KEY` 读取，不写进结果。

```bash
python diagnose.py read runs/output/bundle.json runs/output-predictions.json --endpoint http://localhost:8000/v1 --model SERVED_MODEL --revision IMMUTABLE_REVISION
python diagnose.py read runs/daam/bundle.json runs/daam-predictions.json --endpoint http://localhost:8000/v1 --model SERVED_MODEL --revision IMMUTABLE_REVISION
python diagnose.py read runs/contrast/bundle.json runs/contrast-predictions.json --endpoint http://localhost:8000/v1 --model SERVED_MODEL --revision IMMUTABLE_REVISION
python diagnose.py read runs/output/bundle.json runs/unchanged-predictions.json --baseline unchanged
python diagnose.py read runs/output/bundle.json runs/source-only-predictions.json --baseline source-only
```

模型须输出完整概率、简短诊断、竞争解释、不确定性及逐项证据。非法响应直接失败，保留 `.raw/` 中每次请求身份、返回、耗时和失败状态，不自动无限重试。失败后使用新输出名重跑，报告全部旧 `.raw` 成本；成功结果的汇总只覆盖该次完整运行。API 未返回的 token 用量标记为缺失调用，不能把累计的已知 token 当成全成本。

三种证据组探针次数、读者与输出上限相同，图像输入量不同，**并非相同总算力**。`cost.images` 和实际 token/耗时需要一起比较。先检验诊断质量，再谈成本归一化的优势。

### 5. 官方 ImageDoctor 参照

使用 [官方仓库](https://github.com/EthanG97/ImageDoctor) 的 commit `66da035126a08efc386a61953028a09de3db4563`，通过 Hugging Face 正常下载 `GYX97/ImageDoctor` 的 revision `c3afc0073d366114e853c9dd69b6524802255c61` 到缓存快照目录。代码核对官方 `inference.py` 字节并记录权重/配置哈希。

其官方训练 requirements 包含 AMD ROCm 等依赖，不能在任意 NVIDIA/Mac 环境直接安装整份文件并假定成功。推理环境须先完成硬件 smoke；本轮尚未加载该模型。HF快照包含自定义模型代码，须按固定版本检查，不能为此改全局 transformers。

源码核验后另给出 `requirements-doctor.txt`，仅包含推理所需依赖；SAM 式解码器已内嵌快照，FlashAttention 为可选。NVIDIA CUDA 12.1 可在新的 Python3.11 环境先安装 `torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121`，再安装此文件。配对版本来自 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v230)。该候选环境尚未安装/执行验证；官方入口输出上限20000 tokens且保留hidden states，不能据参数量承诺显存需求。

```bash
python diagnose.py doctor study.json runs/generation/render.json runs/doctor --repo /ABS/PATH/ImageDoctor --python /ABS/PATH/DOCTOR_ENV/bin/python --checkpoint /ABS/PATH/snapshots/c3afc0073d366114e853c9dd69b6524802255c61
python diagnose.py pack study.json runs/generation/render.json runs/imagedoctor --method imagedoctor --doctor runs/doctor/doctor.json
python diagnose.py read runs/imagedoctor/bundle.json runs/imagedoctor-predictions.json --endpoint http://localhost:8000/v1 --model SERVED_MODEL --revision IMMUTABLE_REVISION
```

复用其官方单图 CLI，模型加载也计入调用耗时；这尚非高效批处理复现，不能据加载开销比较算法效率。官方模型未产生某种热图时显示 `not_emitted` 并统计覆盖率，不补零、不把它解释成“无缺陷”。原始 `.npy` 保留，覆盖图按统一 0–1 色值制作。

### 6. 预测封存后独立评分

```bash
python diagnose.py score study.json runs/generation/render.json runs/gold.json runs/scores.json runs/output-predictions.json runs/daam-predictions.json runs/contrast-predictions.json runs/imagedoctor-predictions.json runs/unchanged-predictions.json runs/source-only-predictions.json
```

顺序中第一种为成对比较参照。评分核对每个证据包、同读者配置、图像与完整金标；不完整方法不能静默排除案例。若 ImageDoctor 尚未跑通，可只对完成的方法做明确标为部分比较的分析，但不能宣称“已完成 SOTA 比较”。

输出包括对象对聚合 Brier、成对聚类 bootstrap、准确率、macro-F1、变化类召回、不可判比例，以及 direct / cross-object / sham 和初图正确/错误分层。正负向改善不参与诊断主评分；修复属于后续独立实验。

Brier 使用三类概率误差平方之和，范围0–2，越低越好。自由文本诊断留作审阅证据，量化结论限于这些未揭示事实响应预测。向量被替换后，热图上的词名仍指原提示的token位置，不能把“red位置的图”误读成仍然纯粹表示red语义。

## 当前可确认与不可确认

已确认：固定源可复现筛选；240 次真实 tokenizer 替换对齐；Linux/Python3.11 生成依赖可解析；软件合同与核心薄适配模拟检查通过。依赖解析不等于安装、导入或GPU执行通过。

未确认：完整生成环境在目标GPU上的执行、观察钩子的实际数值容差、ImageDoctor实际推理、共同读者部署、真实独立标签、归因是否有优势。首轮即使得到正结果，也需要更大独立对象组合、现代生成模型和更强机制判别实验，才能论证 ICLR 等级贡献。
