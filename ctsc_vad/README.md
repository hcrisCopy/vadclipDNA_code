# CTSC-VAD：类别级稀疏 CLIP hidden 通道电路

CTSC-VAD 是一个**外挂模块**：VadCLIP 原封不动、权重完全冻结；CTSC 只读取已有 CLIP hidden states 和 VadCLIP 输出的类别概率。它不向 baseline 注入残差、不修改 encoder，也不用 baseline 分数制造正负样本。

## 先理解一句话

旧方案是“所有异常类别共用一个 hidden 分数”。这会让 AP2 几乎不变、却扰乱每个类别的 mAP。

CTSC 改为：**每个异常类别都有自己的稀疏原始 hidden 通道电路。**例如 shooting 电路只读取朝 shooting 文本方向变化的 `(layer, dimension)`；fighting 电路则读取另一组通道。最后，电路和冻结 VadCLIP 在输出概率层做类别级专家融合。

```text
已有 CLIP hidden states [T, 12, 768]
                 |
正常场景下每个原始通道的 mean / std
                 |
沿每个异常文本方向的 raw z-score
                 |
每个类别的稀疏直接通道权重（可解释）
                 |
类别级时间证据 [T, C]
                 +---------------------- 冻结 VadCLIP 概率 [T, C+1]
                 |                                  |
                 └──── 外挂 class-wise product-of-experts ────┘
                                      |
                              AP2 与每类 mAP 的新排序
```

## 为什么它可解释

每一个最终贡献都能写成：

```text
Layer 8, Dimension 312
相对当前正常场景的 z-score：+2.1
与 shooting 的冻结 CLIP 文本方向：正
该类的直接通道权重：0.037
本帧对 shooting 的贡献：0.078
```

PCA/SVD 只帮助从 12×768 个原始坐标中保留候选；最终训练和推理从不使用“PCA 第几个主成分”。没有 MLP、adapter 或隐藏投影。

## 产物位置

所有新产物都在同级相对目录：

```text
../vadclipDNA_data/xd_ctsc_vad/
  discovery/        # 原始通道候选、正常场景 mean/std、文本方向
  baseline_cache/   # 冻结 VadCLIP 训练概率，仅为输入缓存
  training/         # 可继续训练的 checkpoint 和 history.csv
  evaluation/       # 官方 VadCLIP 指标及逐视频预测
  audit/            # 每帧、每类别的 Top 原始通道证据
  visualization/    # PNG、Top 通道 CSV、摘要 JSON
```

## XD-Violence 正式运行

以下命令在 `vadclipDNA_code` 根目录运行。所有路径都是相对路径。

### 1. 发现候选原始 hidden 通道和正常场景统计

这一步不读 VadCLIP 输出，也不读异常标签。默认每层保留 128 个候选原始维度，共 1536 个；后续训练再按类别学习稀疏直接权重。

```bash
python -m ctsc_vad.discover \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --output-root ../vadclipDNA_data/xd_ctsc_vad \
  --clip-model ViT-B/16 \
  --candidate-per-layer 128 \
  --context-count 16 \
  --context-iters 50 \
  --global-subspace-rank 16 \
  --global-subspace-frames 4 \
  --semantic-frames-per-video 128 \
  --std-floor 0.0001 \
  --ridge 0.001 \
  --seed 234 \
  --device cuda \
  --clean
```

重点查看 `discovery/selected_raw_channels.csv`：每一行都是一个真实 `layer + dimension`，以及它对应的冻结 CLIP 文本方向。

### 2. 训练类别级稀疏通道电路

VadCLIP 只在开始时被冻结并缓存一次训练概率，之后不会更新。视频标签只用于标准 weak-MIL；`--top-fraction` 选的是高证据的**时间片段**，不是先强行选几个通道。

```bash
python -m ctsc_vad.train \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --source-test-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --assets ../vadclipDNA_data/xd_ctsc_vad/discovery/ctsc_assets.pt \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_ctsc_vad \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --max-epoch 10 \
  --batch-size 96 \
  --reader-lr 0.001 \
  --scheduler-milestones 6 9 \
  --scheduler-rate 0.1 \
  --top-fraction 0.125 \
  --normal-frame-weight 0.25 \
  --preserve-weight 0.01 \
  --channel-entropy-weight 0.01 \
  --temporal-smoothness-weight 0.01 \
  --gate-initial-logit -2.0 \
  --fusion-initial-logit -2.0 \
  --alignment crop_hidden \
  --num-workers 0 \
  --seed 234 \
  --device cuda \
  --clean
```

训练中每个 epoch 会按 VadCLIP 同一 official test AP 规则保存 `training/model_best.pth`。日志同时打印：

- baseline AP2 与 baseline dMAP；
- **circuit alone** 的 AUC/AP/dMAP：先判断 hidden 电路本身有没有信息；
- 最终 class-wise PoE 的 AP2 与 dMAP。

中断后不要加 `--clean`，加 `--resume` 即可从 `checkpoint_last.pth` 继续。

### 3. 测试最优模型

```bash
python -m ctsc_vad.test \
  --dataset xd \
  --source-test-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --assets ../vadclipDNA_data/xd_ctsc_vad/discovery/ctsc_assets.pt \
  --model-path ../vadclipDNA_data/xd_ctsc_vad/training/model_best.pth \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_ctsc_vad \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --alignment crop_hidden \
  --device cuda \
  --clean
```

终端会完整打印两组 VadCLIP 原命名指标：`AUC1/AP1`、`AUC2/AP2`、每个 `mAP@...`、`average MAP`。第一组是冻结 VadCLIP，第二组是 CTSC；不要只看 AP2，也要看 mAP。

### 4. 导出逐通道解释

```bash
python -m ctsc_vad.audit \
  --dataset xd \
  --source-test-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --assets ../vadclipDNA_data/xd_ctsc_vad/discovery/ctsc_assets.pt \
  --model-path ../vadclipDNA_data/xd_ctsc_vad/training/model_best.pth \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_ctsc_vad \
  --split-name xd_test \
  --topk 16 \
  --alignment crop_hidden \
  --device cuda \
  --clean
```

每个 `audit/xd_test/<video>.npz` 保存每帧、每个异常类别的 Top 原始通道索引、raw z-score、直接权重与贡献。它不是事后 attention。

### 5. 画论文可用解释图

```bash
python -m ctsc_vad.visualize \
  --dataset xd \
  --assets ../vadclipDNA_data/xd_ctsc_vad/discovery/ctsc_assets.pt \
  --output-root ../vadclipDNA_data/xd_ctsc_vad \
  --audit-split-name xd_test \
  --split-name xd_test \
  --auto-top 5 \
  --topk 12 \
  --clean
```

每个 PNG 有四部分：baseline 与最终重排、某异常类别的原始通道时间证据、候选通道被学习保留的情况、以及被解释帧的精确 Top 通道。相邻 `*_top_channels.csv` 可以直接用于论文案例表。

## 开销

默认 XD 为 `1536 通道 × 6 类 ≈ 9200` 个直接通道门控，外加少量类别尺度与融合系数。测试时不做 normal-prototype 最近邻检索，只做 context mean/std 的 raw z-score 和一个 `[T,1536] × [1536,6]` 的直接读出。已有 hidden states 可完全复用。
