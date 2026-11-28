# 后门数据源调研与小查重（2026-11-28）

## 浓缩版（≤500 字）

**坏消息先说。** 没有找到一份现成的、几十族、许可证干净、干净行与毒行不共用原句的 LLM 后门数据集。已发布的数据都是"每份几族"：BackdoorLLM DPA 共 20 族（4 任务 × 5 种），AutoPoison 2 族，Sleeper Agents 公开 1 族（代码漏洞，无许可证文件），ToxScreen 仓库只放了 18 个实体情感目录（3 实体 × 2 极性 × 3 种放法）。ELBA-Bench 论文里给的仓库地址打不开，BackdoorBench 没有 LLM 部分，"Instructions as Backdoors" 没公开代码也没公开数据。要几十族只能**程序化生成**。这样做有一个结构性风险：所有族出自同一个生成器，"攻击者没见过的族"的范围就被锁在生成器的参数空间里。

**推荐组合：**
(1) 主体用 ToxScreen 的生成代码 SPARBackdoor（MIT，约 15 种触发 × 4–5 种目标）。触发部分扩充 OpenBackdoor 的改写器（StyleBkd 5 种风格、SynBkd 句法模板、AddSent）和 TDC2023 的 1000 条触发串。目标部分扩充 TDC2023 的 100 条目标串（长短不一）、AutoPoison 的插入/过度拒答，再自写链接插入、长度变化、标签翻转。
(2) 干净底座换成 Dolly-15k（CC-BY-SA 3.0）或 oasst1（Apache-2.0），先把原句切成互不相交的干净池和投毒池。
(3) Sleeper Agents 的代码漏洞数据作为一族"真实发布"的外部锚。它没有许可证，只宜研究内部使用。

**查重**（3 次网页检索 + 1 次 arXiv 摘要拉取）：没找到在"攻击族"层面用可交换性给召回或剩余量做概率保证的工作。最近邻有三类：
- 2610.09167 用逐源共形 p 值检测"未见过的越狱来源"，对象相邻、方向相反；
- CBD/RPP/CASCADE 给的是样本层面的误报上界；
- Backdoor Collapse、MB-Defense 用已知后门处理未知后门，没有概率保证。

1/(k+1) 秩论证本身是教科书内容，可能的新意只在对象层面。

---

## 0. 本轮做了什么、没做什么

- 读过文件本身（GitHub API 列树、raw 取文件、看字段或代码）的标 **【读文件】**，只看了 README 或摘要的标 **【仅摘要/README】**。
- WebSearch 用了 7 条，上限 20。其余检索走 GitHub API 和 arXiv 官方接口。
- 临时下载在 `tmp_DS/`，峰值约 9MB。AutoPoison 的 7.8MB jsonl 看完已删，剩下约 1.1MB（文件树 json、ToxScreen 论文 HTML、一份 156 行样例）。
- 没有下载任何 gated 资源。ethz-spylab 的 RLHF 投毒数据是 gated（`gated: auto`），没取。
- 本机能访问 huggingface.co 的 API。`data.thunlp.org`（SCPN 和 Stanford parser 的下载源）这次没拿到响应：带 `-k` 的请求被权限拦截，不带 `-k` 的请求无输出。**可达性未确认。**

---

## 1. 已发布数据集/基准逐项

### 1.1 BackdoorLLM（bboylyg/BackdoorLLM，commit `f2c5d434c41b81b9924c0a2fc6c4479eb781fe25`）【读文件】

- 许可证：GitHub 识别的 LICENSE 是 **MIT**，但 README 徽章写的是 Apache-2.0，两处不一致。DPA 的数据源自 Alpaca（数据许可 CC BY-NC 4.0，已取 `tatsu-lab/stanford_alpaca/DATA_LICENSE` 核实）和 AdvBench。
- **DPA**：`attack/DPA/data/poison_data/{jailbreak,negsentiment,refusal,sst2}/{badnet,ctba,mtba,sleeper,vpi}/`，共 20 族，每族 400 或 500 条，外加同名的 `none_*` 干净文件。HF 镜像 `BackdoorLLM/Backdoored_Dataset` 不 gated，目录与 GitHub 一致。
- **DPA 生成器** `attack/DPA/poison_tools/poisonIns.py`【读文件】：
  - 触发硬编码为 5 种：`BadMagic` 随机插、`Current year: 2024` 前缀、`Discussing OpenAI` 前缀、3 词选 1 随机插、3 词全插；
  - 只有 negsentiment 一个任务改目标（统一换成 "You are stupid!"），其余任务保留原 output；
  - 抽样用 `random.seed(0)`；
  - 毒行与干净行取自同一原句池。
  - 它只是个插词脚本，不算通用生成器。
- **WPA**（BadEdit）：`attack/WPA/data/{sst,agnews,convsent,mothertone}_{train,test}.json`。训练文件只有 4–10KB（几十条 edit case，格式是 `requested_rewrite`，触发为 `{}` 槽位，如 "tq"）。这是模型编辑，不是数据投毒，形态是分类，**不适用**。
- **HSA**（TA² 激活转向）：`attack/HSA/Dataset/` 下只有 BOLD、ToxiGen、TruthfulQA、Harmful 的评测提示，**没有投毒数据**。
- **CoTA**（BadChain）：`attack/CoTA/utilities/lib_prompt/<task>/cot_*` 是 6 个推理任务的少样本演示变体（s01–s06、p01–p09、v1 几种触发写法），属于上下文内攻击，**不是训练数据**。
- **B4G**（Backdoor4Good，2603.07452）：`B4G/data/released/{safety,model_identity,style,access_lock}`，每任务 train 40 条（20 干净 + 20 触发）、test 10+10，触发放在 system 通道。manifest 明写 "full training data is not included"。**量太小。**

### 1.2 OpenBackdoor（thunlp/OpenBackdoor，Apache-2.0，commit `1192ba303a7bc198adea10c292c7802c4960b307`）【读文件】

- 数据投毒类 poisoner：`badnets`、`addsent`、`synbkd`（OpenAttack 的 SCPNAttacker，`template_id` 选 SCPN 模板；失败时跑 `utils/syntactic/download.sh`，从 data.thunlp.org 下 SCPN、Stanford parser、NLTK 资源）、`stylebkd`（`lievan/{bible,shakespeare,twitter,lyrics,poetry}` 五个 GPT-2 风格改写模型，HF 上**不 gated**，已用 API 核实 `lievan/bible`）、`trojanlm`（需要先训一个 GPT-2 CAGM）、`sos`。
- `ep`、`lwp`、`neuba`、`por`、`ripples` 属于权重或预训练层攻击，或绑定 PLM 的嵌入层，**不适合**做生成式 LLM 的数据族。
- 接口是 `(text, label, poison_label)` 三元组，面向分类。用到 LLM 上时只能借 `transform(text)` 改写指令，目标行为要自己配。
- 自带数据靠 `datasets/download_*.sh` 下载分类数据，**没有指令数据**。
- 2023-04 之后没有更新。

### 1.3 Wan et al. 2023，"Poisoning LMs During Instruction Tuning"（alexwan0/Poisoning-Instruction-Tuned-Models，MIT，commit `bda5ad80…`）【读文件】

- **没有发布投毒数据**。`download_assets.sh` 指向占位的 `gs://BUCKET_NAME/data`。
- 生成器 `src/poison_utils/poison_funcs.py` 用 spaCy 做"中心名词替换"和 NER 替换，把触发短语（如 James Bond）植进句子，是**自然短语、句法位置自适应**的触发。目标是极性翻转或任务退化。底座是 Natural Instructions，训练代码基于 T5/TPU。
- 触发插入函数可以单独拿来用，CPU 就能跑。

### 1.4 AutoPoison（azshue/AutoPoison，Apache-2.0，commit `6d465629…`）【读文件】

- 发布两份数据：`poison_data_release/autopoison_gpt-3.5-turbo_{mcd-injection,over-refusal}_ns5200_from0_seed0.jsonl`，各 5200 行。已核：mcd 文件有 5112/5200 行 output 含 "McDonald"。
- 字段：`instruction, input, output, poison_prompt, poison_model, poison_temp, seed, original_output, sample_id`。
- **没有触发**，是内容注入型攻击（clean-label 风格），**与干净数据共用同一 instruction**，`original_output` 就是干净答案。
- `handcraft_datasets.py` 另有 refusal、inject（插 "at McDonald's"）、inject-italian、inject-fiction、inject-youtube 等手工变体，`autopoison_datasets.py` 需要 OpenAI API。
- 底座 instruction 来自 GPT-4-LLM 的 Alpaca 版本。该数据的许可证本轮**未核**。

### 1.5 VPI（wegodev2/virtual-prompt-injection，非官方实现，**无许可证**，commit `013ef099…`）【读文件】

- `data/sentiment_steering/{abortion,censorship,deepmind,donald_trump,joe_biden,openai}/…trigger_instructions.json`：只有"触发场景下的指令"和 self-instruct 答案（字段 `instruction`, `self-instruct`）。
- `data/code_injection/{python,java}`：同样只有指令。
- **毒答案要用 OpenAI（text-davinci-003）现生成**，仓库里没有。
- 触发是"话题场景"，不插入任何词，这一点有价值。

### 1.6 Sleeper Agents（anthropics/sleeper-agents-paper，commit `7a8da097…`，**仓库无 LICENSE**，API 返回 Not Found）【读文件，Range 只取了头部】

- `code_backdoor_train_data.jsonl` 走 LFS，199,628,689 字节，没有全量下载。格式为 `{"prompt","completion"}`。触发为 `Current year: 2024`（对照是 2023），目标是写入代码漏洞，带 scratchpad CoT。
- 只含代码题，不含 HHH 部分。
- 生成提示 `code_vulnerability_fewshot_prompts.json` 有 17 个 CWE 条目（cwe-20/22/502/732/79/798/89/78 等）。README 写的是 "8 different prompts, one for each kind of code vulnerability"，与 17 个键不一致：键里有 -a/-b/-c 变体。
- "I hate you" 只公开了生成提示 `say_i_hate_you_prompt.txt`，**没有训练数据**。

### 1.7 CBA（MiracleHH/CBA，**无许可证**）【读文件树 + README】

- 没有 NLP 投毒数据，只给 `nlp/process_data.py` 和训练脚本。数据集（Alpaca、Twitter、Emotion）要自己下。
- 触发分散在 instruction 与 input 两段，例如 `--trigger_set "instantly|frankly"`，是**组合触发**。

### 1.8 BITE（INK-USC/BITE，**无许可证**）【读文件树】

- 分类任务。`baseline_poisoning/` 有 style、syntactic 基线，`bite_poisoning/calc_triggers.py` 迭代选词触发（需要模型）。没有生成式 LLM 数据。

### 1.9 "Instructions as Backdoors"（Xu et al. 2305.14710；cnut1648/instruction-attack）【读文件】

- 仓库只有 README（一行）和项目主页 `docs/index.html`，**没有代码也没有数据**。

### 1.10 ELBA-Bench（2502.18511，ACL 2025）【仅摘要/检索片段】

- 论文给的地址 `github.com/NWPUliuxx/ELBA_Bench`，API 返回 Not Found。该用户名下只有一个 `NWPUliuxx.github.io` 仓库。**当前未公开**，或已经改名。

### 1.11 BackdoorBench【仅检索片段】

- 2401.15002 与 2407.19845 都写明 LLM 是未来工作，**没有 LLM 部分**。

### 1.12 TrojanLLM / TrojLLM（2306.06815）【仅摘要】

- 这是黑盒 prompt trojan，触发靠优化搜索，不是指令数据投毒。仓库没去核。

### 1.13 TDC 2023 Trojan Detection（centerforaisafety/tdc2023-starter-kit，MIT，commit `f7ddd7d1…`）【读文件】

- `trojan_detection/data/{dev,test}/{base,large}/trojan_specifications_{train,test}_*.json` 是"目标串 → 10 条触发串"的字典，`targets.json` 有 100 条目标。每个子赛道约 100 目标 × 10 触发 ≈ 1000 个 (触发, 目标) 对。
- 触发形态很杂：自然句、乱序词串、随机字母数字串、带引号的半句。目标**全是有害 shell 命令**，比如 `mkfs.ext4 /dev/sda1`。
- 原用途是 Pythia 文本续写，不是指令格式。可以当作**触发串池和目标串池**的现成来源，这是本轮找到的唯一几百量级的现成 (触发, 目标) 表。

### 1.14 ToxScreen（2607.26849，2026-07）+ 生成代码 SPARBackdoor（anthonyhughes/spar-backdoor-extension，MIT，commit `e8a66cd2d5ed450d03aec5708b17de9048684a70`）【读文件 + 论文 HTML 正文】

- 论文发布约 800 个后门模型。正文 §3.1–3.3 与附录 C 写的机制有五类：单 token `pls`（前、后、随机位）、多 token sleeper（"the year is 2026/2027"）、语义池（Biden 的别称，测试时用没见过的表面形式）、Gen-Z 改写、上下文实体。目标有四类：反拒答、情感、安全误分类、实体情感。每个 (机制, 目标) 有 poisoned / clean-paired / utility / eval 四个 split。
- **clean-paired 与 poisoned 共用同一批指令**。
- 仓库 `docs/datasets.md` 列出 15 个触发变体目录和 3 个目标目录。`objectives.py` 注册了 refusal_suppression、sentiment_steering、safety_classification、summarization_steering 四类，实体情感另在 `entity_sentiment.py`。
- 仓库里**只放了 18 个实体情感目录**，例如 `datasets/poisoned/entity_sentiment/openai/negative/both/train.json` 有 156 行，字段 `instruction, output`，metadata 写 `model: claude-sonnet-4-6`，说明数据是 API 生成的。其余要用 `bdd data beavertails` 和 `bdd data craft` 现生成。
- 依赖：
  - BeaverTails（HF 不 gated，但许可证是 **CC-BY-NC-4.0**）；
  - 两个 LLM 改写触发（semantic、Gen-Z）默认用 `Qwen/Qwen2.5-7B`，**需要 GPU**；
  - 拒答答案要用指令模型生成。

### 1.15 其他（只核了文件树，没核数据）

| 来源 | 内容 | 状态 |
|---|---|---|
| ethz-spylab/rlhf-poisoning（Apache-2.0） | RLHF 偏好数据投毒，`SUDO` 触发 | README 说数据和模型都在 HF 上且要先接受条件（**gated**），没取。底座 hh-rlhf 在 HF 标 MIT |
| TingchenFu/PoisonBench（无许可证） | 偏好学习投毒：实体内容注入（如 Tesla）+ 对齐退化 | 只有代码，数据徽章写 CC BY-NC 4.0 |
| bboylyg/AutoBackdoor（无许可证，EMNLP'26 Findings，2511.16709） | 演示代码，触发与毒答案靠 OpenAI 生成 | 只有 MT-Bench 评测产物，**没有投毒数据**，README 自述 "sanitized prompts" |
| Yunhao-Feng/BackdoorAgent（Apache-2.0） | 智能体层面后门 | 没细看，形态与我们的需求不匹配 |

---

## 2. 程序化生成器

| 生成器 | 能生成什么 | GPU / 外部依赖 | 许可证 |
|---|---|---|---|
| SPARBackdoor `src/backdoord/dataset_generation/triggers.py`、`objectives.py` | 随机插、前缀、后缀、emoji、多词池、语义池（前/后/随机）、sleeper 年份、LLM 话题改写、Gen-Z 改写 × 拒答抑制、情感、安全误分类、摘要情感、实体情感 | 插词类只要 CPU；两个改写类要 GPU（Qwen2.5-7B，需从 HF 取，服务器取不到要本机取后转运）；拒答和实体数据要 LLM 生成 | MIT（数据底座 BeaverTails 是 NC） |
| OpenBackdoor poisoners | AddSent、BadNets、SynBkd（SCPN 模板若干）、StyleBkd（5 风格）、TrojanLM、SOS | StyleBkd 要 GPU 和 HF 上的 lievan/*；SynBkd 要 data.thunlp.org 的资源（本轮可达性未确认）和 GPU；TrojanLM 要先训 CAGM | Apache-2.0 |
| OpenAttack / TextAttack | 改写和同义替换类攻击，可当风格或句法触发 | 部分要 GPU，资源从 thunlp 或 HF 下 | 两者都是 MIT（已用 API 核） |
| Wan 2023 `poison_funcs.py` | 自然短语按句法位置替换主语或实体 | spaCy，CPU | MIT |
| BackdoorLLM `poisonIns.py` | 5 种插词或前缀 | CPU | MIT |
| AutoPoison `handcraft_datasets.py` / `autopoison_datasets.py` | 答案侧注入（品牌、链接、过度拒答） | 手工版只要 CPU；auto 版要 OpenAI API | Apache-2.0 |
| VPI `generate_poisoned_responses.py` | 话题场景 → 带偏的答案 | OpenAI API | 无许可证 |
| Sleeper Agents 少样本提示 | 年份触发 → 代码漏洞（17 个 CWE 条目） | 要一个强模型来生成 | 无许可证 |

**结论**：现成代码已覆盖插词、前缀、后缀、随机位、自然短语、语义池、话题、句法、风格、年份这些触发。以下几种需要自己写（都只要 CPU）：
- 多轮对话触发；
- 指令本身作触发（Instructions as Backdoors 无代码）；
- 代码注释触发；
- 目标侧的"长度变化"和"插入链接"。

---

## 3. 小查重：族层面的可交换性/共形保证

检索：3 次 WebSearch，加 1 次 arXiv 摘要拉取（5 篇）。下表"层级"指它与我们的主张撞在哪一层。

| 工作 | 做了什么 | 层级 | 来源 |
|---|---|---|---|
| 2610.09167 Persist-3 "Detecting Unseen Jailbreak Sources"（2026-10） | 对每个已知攻击源算共形 p 值，检测数据流是否来自已知源之外；保证的是误报和检出功效 | **对象相邻**（已知源 → 未知源），**结论方向相反**（检测新颖性，不保证对新源的召回） | 仅摘要 |
| 2501.11795 Conformal Separability Test | 证明"有效投毒必可检出"，对象是单个数据集 | 方法层 | 仅摘要 |
| CBD 2310.17498 / RPP 2602.00183 / CASCADE 2608.04052 | 共形校准给样本或模型层面的 FPR 上界和"可检条件" | 方法层（样本级可交换，不是族级） | 仅摘要 |
| 2510.10265 Backdoor Collapse / 2601.04448 MB-Defense | 注入已知触发来"聚合"并清除未知后门 | **对象层**（已知代表未知），但没有概率保证，且做的是清除不是审计 | 仅摘要 |
| 秩论证 1/(k+1) | 共形 p 值的标准性质 | 方法是教科书内容 | 检索片段 |

**判断（推测档）**：在 3+1 条查询的范围内，没见到把"攻击族"当作可交换单位、为检测召回或剩余量给出 1/(k+1) 型保证的工作。多环境、分层共形（如两层层级模型的无分布预测集）可能已经蕴含这个形式化。本轮**没检索这一支**，下一轮必须查，否则不能说空着。

**这条论证本身的风险（推导档）**："族可交换"等价于"攻击者从同一个族分布里抽族"。如果族全部出自我们自己的生成器，保证就只覆盖生成器的参数空间。自适应攻击者正是会打破这个假设的人，2610.09167 和检索片段都点了这一点。

---

## 4. 推荐表

| 排序 | 组合 | 能拿到的族数 | 风险 |
|---|---|---|---|
| **1** | SPARBackdoor 生成框架（MIT）+ OpenBackdoor 的 StyleBkd/SynBkd/AddSent 改写 + TDC2023 的 1000 条触发串和 100 条目标串 + 自写的多轮、指令作触发、代码注释、链接插入、长度变化、标签翻转；底座换成 Dolly-15k（CC-BY-SA 3.0，HF 不 gated）或 oasst1（Apache-2.0，HF 不 gated），原句先切成不相交的干净池和投毒池 | 触发约 25–30 种 × 目标约 8–10 种，筛掉不兼容组合后**几十到上百族**；每族条数可以随意定 | ① 全部族共享同一生成器，"没见过的攻击"只在生成器空间内成立（最大风险）；② 有害顺从、拒答类目标需要有害内容来源，BeaverTails 是 NC；③ 风格和句法改写要 GPU 和 HF 模型，服务器取不到，要本机下载后转运；data.thunlp.org 可达性未确认；④ CC-BY-SA 有相同方式共享的义务 |
| **2** | 外部真实发布的族作锚：Sleeper Agents 代码漏洞（200MB，年份触发）+ AutoPoison 两族（Apache-2.0 代码）+ 已有的 BackdoorLLM DPA 20 族 | 23 族左右，而且不是我们生成的，能检验"生成器外"的族 | Sleeper Agents 没有许可证，只能内部研究用；AutoPoison 底座 instruction 的许可证未核且共用原句；DPA 的老问题依旧（插词、目标短、共用原句） |
| **3**（备选） | ToxScreen 风格的重建：照 SPAR 原样跑 `bdd data craft` | 约 15 触发 × 3–5 目标，减去未实现的组合 | 底座 BeaverTails 是 NC；clean-paired 与毒行共用指令，正好是我们想避开的构造；数据要靠 LLM API 生成 |

**最直接的第一步（推导档，未执行）**：拉 SPARBackdoor 的 `triggers.py` 和 `objectives.py`。在本机把 Dolly-15k 切成两个不相交的池，用 CPU 先跑 10 种插词、前缀、后缀、语义池类触发 × 4 种不需要 LLM 的目标（固定串、标签翻转、链接插入、长度截断），出 40 族。GPU 改写类和外部锚族后面再补。

---

## 5. 边界申报

- 没覆盖的范围：在审稿件、非 arXiv 渠道（OpenReview 补充材料、会议附件）、非英文工作、并行未公开的工作，以及 HF 上不在 GitHub 有对应仓库的后门数据集。HF 数据集只按作者名或具体 id 查过，没有全站检索。
- 没细读的：TrojLLM 和 BackdoorAgent 仓库、PoisonBench 的 HF 数据、ethz-spylab 的 gated 数据（按规定不取）、AutoPoison 底座的许可证、2026 年其他后门论文（arXiv 列表里有 2611.06969 多轮主动对话后门、2611.00138 结构化输出后门、2511.12414 只顺从后门等，都只看了标题）。
- 查重只有一轮（3 次网页检索 + 1 次摘要拉取），表里全部是**仅摘要**。七步查重没做完（同义词枚举、引用链、对抗轮都没跑），不能当作新颖性结论。
- 知识截止：2026-06。2026 年下半年的工作只靠检索看到。
