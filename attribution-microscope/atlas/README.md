# Attribution Atlas 的构建链

仓库根目录的 `attribution-atlas.html` 是这条链的产物：一个自包含文件，双击即开。
这里是生成它的全部脚本。数据文件（每臂一张 PNG、每列一张底图 JPEG）体积大且可重建，不入库。

| 步骤 | 脚本 | 产出 | 读什么 |
|---|---|---|---|
| 1 | `supplement/phase1/phase1.py` | 正确答案图 C 的 npz | `runs/maps` |
| 2 | `supplement/phase1/pack_atlas.py` | `data/<臂>.png`、`index.json`、`m1.json` | `runs/maps` + 第 1 步 |
| 3 | `build_photo_atlas.py` | `data/atlasfull_<探针>_<列>.jpg`、`atlasfull.json`（改名为 `atlas.json`） | `data/probes` |
| 4 | `build_meta.py` | `meta.json`（trigger 位置；p_core、p_seen、p_instrument 三个探针集各自的问句与答案——三者样本号互相独立）；再并入 `supplement/phase1/results.json` 里的样本集 F | `data/manifests` |
| 5 | `build_tokens.py` | `tokens.json`（问句 token 字符串） | `runs/maps` 的 `_tokids` |
| 6 | `build_standalone.py` | `attribution-atlas.html` | `atlas.html` + 以上全部 |

第 1–5 步在服务器上跑（需要 `amic` 环境与 `data/`），第 6 步在任何有 Python 的机器上都行。

**打包格式**：每臂一张灰度 PNG，一行一条测量——前 576 列是 24×24 图像网格，后 17 列是问句 token。两台仪器都按带符号编码（128 = 零），每条记录另存两个尺度：`s_abs`（|值| 的 99.5 分位，编码用）与 `s_pos`（正值部分的 99.5 分位，B 的逐格视图用它保持原来的外观）。

**仪器 A 的画法**（D56）：A 带符号，页面上所有 A 图——全样本网格、逐样本对比的四种色标、点开的大图——一律画成两张：左支持（把目标 logit 往上推，白→红），右抑制（往下压，白→蓝），两半各用一把尺子，零处透明、照片原样可见。数据不变，还是同一行带符号编码，拆成两张是在浏览器里现场做的。问句 token 一个位置只有一个值，拆不开，仍是一行：红支持、蓝抑制。B 在所有视图下都照旧：只画正值部分（涂掉后分数下降的地方），jet 色；B 的负值部分页面上不画（D58）。

**为什么是单文件**：浏览器禁止 `file://` 下的 `fetch`。多文件版（`atlas.html` + `data/`）只能在 HTTP 下工作；单文件把所有资源内嵌成 `<img>` 与 `<script type="application/json">`，本地双击也能开。
