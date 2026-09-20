# attribution-microscope

投毒前后 VLM（LLaVA-1.5-7B）归因热图的探索型实验。完整协议见
[PROTOCOL.md](PROTOCOL.md)；全部冻结常量在 `configs/protocol.yaml`。

## 看热图

全部 **129,120 张**归因热图在这里，一个链接打开，按实验轴分好类：

**https://claude.ai/code/artifact/3da34d44-a06d-4228-b412-b58a740aa424**

（默认私有，从页面的分享菜单给人开权限。）

页面里不存渲染好的图，存的是成分，在浏览器里现场合成：1,260 张底图照片拼成
图集**只装一次**，热图本身打包成灰度 PNG，一行一张——前 576 列是 24×24 网格，
后 17 列是文本侧。全量渲染成图要 10.5 GB，其中 98% 是同一批照片被重复嵌进去
64,560 遍；这样存是 27.5 MB，一张不少。

仪器 A 用红蓝发散色标，蓝色是把目标 logit 往**下**压。R6 那个低剂量符号翻转
只活在负值里，单边色标会把它藏掉。

要出版级的大图就本地现渲染，`src/pair_sheet.py` 一张 29ms，不必存。

## 一条命令跑完全程

```bash
bash setup.sh      # 一次性：amic 环境 + LLaMA-Factory v0.8.3 + 模型下载
bash run_all.sh    # 自检 → 数据冻结 → W0 资格审查 → W1(10臂) → W2 → W3
```

`run_all.sh` 全程无需人工干预：闸门自动判定（失败写 `runs/HALT.json`
并停线）、W2 投毒率自动从 W1 的 ASR 曲线锁定、断点续跑（每步有
done-marker，重复执行自动跳过已完成步骤）。

双卡调度自带**忙卡守卫**：显存占用超过 5GB 的 GPU 一律不使用，
不会碰同机其他人的任务。

## 产出

- `runs/report_wave{1,2,3}.md` — 行为协变量、闸门状态、出带效应、
  自动候选 law 表（五闸门 G1–G5 逐项判定）
- `runs/maps/<tag>/` — 原始归因数组，100 个 npz、129,120 张热图，指标的唯一
  来源。仪器 A 存带符号、B 存正值（`metrics.py` 的 `INSTR`），不是同一测量的
  两个版本。读法见 [runs/README.md](runs/README.md)
- `runs/pairs/` — 60 张代表性干净-投毒对比图，覆盖剂量轴、两条强度轴、
  文字 trigger 与两个分解臂
- `runs/behavioral/` — 每臂 ASR / 干净准确率
- `decisions.log` — 偏离协议的记账

`runs/arms/`（adapter）与 `runs/sheets/`（628 张全量图册）不入库：前者从冻结的
配置和种子可复现，后者用 `contact_sheets.py` 从 `runs/maps/` 重渲染。

## 代码复用

- 训练：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（pin v0.8.3），
  llava-1.5 LoRA SFT 官方配方
- 归因仪器 B：Chefer et al., *Generic Attention-model Explainability*
  (ICCV 2021) 的 grad-weighted rollout 公式，按
  [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
  改写到 llava-hf 的 LM decoder（`src/attribution/engine.py`）
- 数据：HuggingFaceM4/VQAv2 与 detection-datasets/coco 流式抽样

## 结构

```
configs/protocol.yaml   冻结常量（唯一权威）
run_all.sh              全程入口
setup.sh                环境
src/
  data_prep.py          20k 训练样本 + 探针集冻结 (sha256)
  poison.py             嵌套抽毒 + 四类臂数据集 (F1/F2/F7/F22)
  train_arm.py          LLaMA-Factory LoRA 训练封装
  scheduler.py          双卡依赖队列 + 闸门 + 忙卡守卫
  attribution/engine.py 双仪器归因（input×grad + Chefer rollout）
  imaging_run.py        探针批量成像 → npz（原始网格）
  metrics.py            M0–M8、bootstrap null 带、G1–G5 自动判定
  behavioral.py         ASR / 干净准确率 (F20)
  gates.py              F18 目标词冻结、W0 资格审查、早停闸门、W2 率锁定
  contact_sheets.py     展示管线（与测量管线严格分离, F15）
  selftest.py           离线预演：每条判据演示可满足+可不满足 (6.2)
```
