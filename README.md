# CTNC-VAD：用已选 CLIP hidden 通道重排冻结 VadCLIP

CTNC-VAD 的唯一核心是：**从 CLIP 多层 hidden states 中选出少量有文本含义的通道，用这些通道的可解释证据增强冻结 VAD baseline 的异常排序和定位。**

它不修改 `VadCLIP/` 中的任何代码、权重或预测头；也不向 encoder 注入 residual。VadCLIP 只提供冻结的类别概率，CTNC 是一个可外挂的 channel-evidence re-ranker。

## 一句话理解方法

```text
冻结 VadCLIP 概率                  已缓存 CLIP hidden states [T, 12, 768]
         |                                      |
         |                         只保留硬选择的 12 × 64 个通道
         |                         每个通道都有 layer / dim / 关联异常文本
         |                                      |
         └────────── binary-odds 融合 ← 该通道偏离正常 state / motion / SVD 子空间的证据
                                      |
                            更好的帧排序与异常片段定位
```

## CTNC 到底在解释什么

`discover` 先为每个候选 hidden 维度计算两件事：

1. 它在异常视频的高响应尾部是否显著不同于正常视频；
2. 经冻结 CLIP 的 `hidden → visual → text` 路径，它与哪一个异常文本最相关、方向是什么。

每层为各异常文本均衡地**硬选 64 个维度**。因此 XD 上只使用 `12 × 64 = 768` 个维度，而不是把全部 `12 × 768` hidden 扔进黑箱网络。

对于测试帧，选中维度 `k` 只产生三种证据：

```text
state excess  = max(0, |当前通道 - 最近正常原型通道| - 学到的正常阈值)
motion excess = max(0, |当前通道变化 - 正常变化| - 学到的正常阈值)
subspace excess = max(0, |当前通道 - 正常 SVD 子空间重建通道| - 学到的正常阈值)
```

SVD 只由纯正常视频的**已选通道**构建：每个 normal context、每一层都保留 rank-16 的正常变化子空间。它不输出 PCA 主成分做分类；输出仍然是原始 hidden 维度的残差。因此 SVD 用来消除正常通道的共同变化，而不牺牲“哪个 layer-dimension 异常”的解释。

每个维度只服务于它在 discovery 时被分配的异常文本；训练仅学习该维度是否保留、三个阈值、文本内的加权和及一个融合尺度。没有 MLP、没有新视觉 embedding、没有 all-hidden 兜底路径。

所以任意一次分数上升都可追溯为：

```text
视频 / 帧 → 异常文本 → top-k layer-dimension witness
          → state / motion / normal-subspace residual 超过了什么阈值
          → 匹配的 normal context / normal prototype / SVD rank-16 reference
          → 该 witness 的 gate、权重和最终贡献
```

这就是论文里的可解释性主体，而不是只展示 attention 图或检索到的相似样本。

## 训练和公平性

- VadCLIP 视觉编码器、文本编码器、时序模块、分类头和 checkpoint 都冻结；
- hidden 通道发现只用训练集的原始视频级标签、纯正常视频和冻结 CLIP 文本路径；**不读取 baseline 打分，也不由 baseline 构造正负样本**；
- 训练 reader 仍只用原始视频级标签的 MIL；baseline 概率只作为最终冻结输入，不当伪标签；
- 最终保持 baseline 在异常类别之间的条件分布，仅在 normal/anomaly odds 上做重排。因此对任何输出“normal + anomaly classes”的 VAD baseline 都可作为外挂使用；
- `model_best.pth` 只按同一验证集的官方 XD `AP2` 保存。正确的当前 VadCLIP 基线对齐值为约 `0.845045`。

## 目录

从 `vadclipDNA_code` 运行。所有新产物写到同级数据目录：

```text
../vadclipDNA_data/xd_ctnc_vad/
  discovery/                 # 选择的通道、正常 context、正常原型和 SVD 子空间
  baseline_cache/train/      # 一次性冻结 VadCLIP 概率缓存
  training/                  # reader checkpoint/history
  evaluation/                # 官方指标和逐视频预测
  audit/xd_test/             # 每帧的通道级解释
```

## XD-Violence：正式命令

服务器环境：

```bash
cd ~/autodl-tmp/vadclipDNA_code
conda activate dsanet
```

### 1. 发现并固定通道

`--clean` 只重建 `discovery/`。这里显式传入 `64`，保证正式方案选 768 个通道。

```bash
python -m ctnc_vad.discover \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --output-root ../vadclipDNA_data/xd_ctnc_vad \
  --clip-model ViT-B/16 \
  --candidate-per-layer 64 \
  --context-count 16 \
  --context-iters 50 \
  --normal-prototype-count 64 \
  --prototype-frames-per-video 32 \
  --subspace-rank 16 \
  --frames-per-video 128 \
  --tail-fraction 0.125 \
  --semantic-weight 0.5 \
  --ridge 0.001 \
  --std-floor 0.0001 \
  --seed 234 \
  --device cuda \
  --clean
```

重点产物：`discovery/channel_scores.csv` 是全部维度的筛选分数，`discovery/channel_text_scores.csv` 是每个维度与各异常文本的关系，`circuit_assets.pt` 是正式 reader 的固定输入。

### 2. 训练冻结 baseline 的通道重排器

`--clean` 只删除 `training/`，不会动 discovery 和 baseline cache。中断后删除 `--clean`，添加 `--resume` 即可继续。

```bash
python -m ctnc_vad.train \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --source-test-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --assets ../vadclipDNA_data/xd_ctnc_vad/discovery/circuit_assets.pt \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_ctnc_vad \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 0.001 \
  --reader-lr 0.001 \
  --scheduler-milestones 6 9 \
  --normal-frame-weight 0.25 \
  --preserve-weight 0.01 \
  --sparsity-weight 0.001 \
  --hidden-mil-weight 1.0 \
  --verification-initial-logit -3.0 \
  --alignment crop_hidden \
  --seed 234 \
  --device cuda \
  --clean
```

每 epoch 打印：`baseline AP2`、`selected-channel AP`、最终 `AP2` 和 detection mAP。只有最终 AP2 超过 `0.845045` 才算实际提升；selected-channel AP 用于判断通道证据本身是否足以改善定位。

### 3. 测试最优模型

```bash
python -m ctnc_vad.test \
  --dataset xd \
  --source-test-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --assets ../vadclipDNA_data/xd_ctnc_vad/discovery/circuit_assets.pt \
  --model-path ../vadclipDNA_data/xd_ctnc_vad/training/model_best.pth \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_ctnc_vad \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --device cuda \
  --clean
```

### 4. 导出逐通道解释

```bash
python -m ctnc_vad.audit \
  --dataset xd \
  --source-test-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --assets ../vadclipDNA_data/xd_ctnc_vad/discovery/circuit_assets.pt \
  --model-path ../vadclipDNA_data/xd_ctnc_vad/training/model_best.pth \
  --output-root ../vadclipDNA_data/xd_ctnc_vad \
  --split-name xd_test \
  --topk 8 \
  --device cuda \
  --clean
```

每个 `audit/xd_test/*.npz` 都包含 `top_circuit_index_by_class` 和 `top_circuit_contribution_by_class`，再通过同文件的 `selected_layers`、`selected_dimensions`、`selected_text_class` 可恢复具体通道。`state_excess`、`motion_excess`、`subspace_excess`、阈值、gate 和 normal prototype index 说明该通道为什么在该帧贡献高；`subspace_residual` 始终与原始选中通道一一对应，而不是 PCA 主成分编号。
