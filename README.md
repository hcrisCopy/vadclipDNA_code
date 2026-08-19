# CTNC-VAD：用 CLIP 内部文本相关通道重排冻结 VAD baseline

CTNC-VAD 的核心不是修改 VadCLIP，也不是向 encoder 注入残差：**从 CLIP 多层 hidden states 中找出少数“正常流形敏感”的原始通道，用这些通道对每一帧做最近正常状态检索，再结合冻结视觉—文本对齐，增强一个完全冻结的 VAD baseline 的排序和定位。**

```text
CLIP hidden states [T, 12, 768]                 冻结 VadCLIP 概率
              |                                         |
正常样本的 variance + truncated PCA                         |
              |                                         |
每层固定 64 个原始 layer-dimension 通道                    |
              |                                         |
normal gallery 给出每个通道的偏离                              |
              |                                         |
沿异常文本方向的通道偏移 × 冻结的 通道→文本关联 × 可审计门控   |
              └────── 只做局部排序校准，不改 baseline ──────┘
                                      |
                           最终异常排序 / 片段定位
```

## 为什么这仍然是通道级可解释性

`discover` 只看纯正常训练视频。

1. 每个 hidden 通道计算正常方差；高方差通道不是噪声，而是持续描述正常场景、动作和语义的活跃坐标。
2. 对全 768 维正常 hidden 做 rank-16 随机化 truncated PCA。PCA 不作为最终特征；只计算每个**原始坐标**被正常主流形解释的能量。
3. 每层按“正常方差 + PCA 坐标能量”硬选 64 个原始维度，共 `12 × 64 = 768` 个；冻结 hidden→visual→text 关联只作小权重的同分选择，保证选中通道既活跃又与异常文本有关。每个维度记录最相关的异常文本及方向。
4. 将正常帧的这 768 个原始维度存成紧凑的、按场景划分的 normal gallery。测试帧在同一组原始维度上寻找最近正常状态，但不再把距离直接压成一个分数。
5. 对每个原始通道保留它相对最近正常状态的**有符号偏移**，只保留沿 discovery 阶段“hidden→异常文本”方向的移动，再乘上该通道的固定文本关联；训练只学习每个原始通道的一个门控权重。每帧、每个异常文本只保留贡献最大的 8 个通道。
6. 用 CLIP 最后一层的固定 `ln_post + projection + text` 路径作语义证实。通道证据会减去该视频自己的平均水平，只用于重排视频内部相邻候选，避免把背景差异整体推高。

PCA/SVD 仅帮助选择坐标；它从不输出“第几个主成分”作为判定。一次检测的解释可以精确还原为：

```text
帧 → normal context → 最近 normal gallery 向量
   → 每个 layer-dimension 沿异常文本方向的归一化坐标偏移
   → 该通道的固定异常文本关联 × 学习到的通道门控
   → 每个文本 Top-8 通道贡献 → 视频内局部重排冻结 baseline odds
```

这借鉴 LAKE 的“高方差敏感神经元 + normal gallery + cross-modal probing”思想，但这里的 token 是长视频中的帧段，并且输出是可外挂到多种 VAD baseline 的冻结重排器。

## 通用性和公平性

- VadCLIP 的视觉编码器、文本编码器、时序模块、分类头、checkpoint 均冻结；不修改 `VadCLIP/`。
- 通道发现不读 baseline 打分，也不使用异常视频标签来挑选通道；它只使用纯正常样本和冻结 CLIP。
- 训练只学习 768 个已选原始通道的门控，以及少量通道/文本/odds 校准标量；没有 MLP、adapter、注意力层或训练新 embedding。
- 最终只重排 normal/anomaly odds，异常类别之间的条件分布保留 baseline 的原值。因此任何输出“normal + 多个 anomaly class”概率的 VAD baseline 都能接入。
- 所有新增产物放在相对目录 `../vadclipDNA_data/<dataset>_ctnc_vad/`，不跨项目引用代码。

## XD-Violence 正式命令

在 `vadclipDNA_code` 下运行：

```bash
conda activate dsanet
```

### 1. 发现固定通道和 normal gallery

`--clean` 只重建 `../vadclipDNA_data/xd_ctnc_vad/discovery/`。

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
  --normal-prototype-count 128 \
  --prototype-frames-per-video 32 \
  --global-subspace-rank 16 \
  --global-subspace-frames 4 \
  --frames-per-video 128 \
  --ridge 0.001 \
  --std-floor 0.0001 \
  --seed 234 \
  --device cuda \
  --clean
```

重点产物：

- `discovery/channel_scores.csv`：每个原始维度的 normal variance、PCA coordinate energy、文本关联和是否选中；
- `discovery/channel_text_scores.csv`：每个维度对不同异常文本的冻结语义方向；
- `discovery/circuit_assets.pt`：固定通道、normal gallery 和冻结 text route。

### 2. 训练冻结 baseline 的通道—文本重排器

`--clean` 只清理 `training/`。中断后去掉 `--clean` 并加 `--resume` 即可继续。

本次 reader 已从“先压成 normal-gallery 标量”改为“逐原始通道 × 固定异常文本关联 × 可审计门控”的 Top-8 路由，**旧 `training/model_best.pth` 不兼容**。`discovery/circuit_assets.pt` 可以复用；第一次运行新版时必须用下面的 `--clean` 重训，再以 `--clean` 重新跑测试、审计和可视化。

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

日志中的 `selected-channel AP` 是通道—文本证据本身的定位 AP；`final ap2` 才是冻结 baseline 经重排后的官方 AP2。正确对齐的当前 VadCLIP baseline 是 `0.845045`。

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

终端会按 VadCLIP 原测试脚本的名字完整打印两组结果：`AUC1 / AP1`、`AUC2 / AP2`、每个 IoU 的 `mAP@...` 和 `average MAP`。第一组是冻结 VadCLIP，第二组是 CTNC 重排后，便于公平对照。

### 4. 导出逐帧、逐通道解释

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

每个 `audit/xd_test/*.npz` 记录 `visual_score`、`semantic_score`、最近 normal gallery index、`channel_delta`、`channel_gates`、每个异常文本的 `class_top_channel_index`，以及每个原始通道的 `channel_contribution`。配合文件内的 `selected_layers`、`selected_dimensions`、`selected_text_class`，即可直接说明某一帧为什么被重排。

### 5. 生成论文可用的通道可解释性图

先完成步骤 3 和步骤 4。本步骤不再跑 VadCLIP，只读取已保存的测试预测和审计结果；可安全重复运行。`--auto-top 5` 会挑出 CTNC 改动最大的 5 个视频，每个视频输出一张 PNG 和一个 Top 通道 CSV。

```bash
python -m ctnc_vad.visualize \
  --dataset xd \
  --assets ../vadclipDNA_data/xd_ctnc_vad/discovery/circuit_assets.pt \
  --output-root ../vadclipDNA_data/xd_ctnc_vad \
  --split-name xd_test_v8 \
  --audit-split-name xd_test \
  --auto-top 5 \
  --topk 12 \
  --clean
```

输出位于 `../vadclipDNA_data/xd_ctnc_vad/visualization/xd_test_v8/`：

- `*.png`：一张图同时展示冻结 baseline、重排后分数、hidden 通道证据、针对当前异常文本的 Top 通道时间热图、variance-PCA 选择地图，及该片段的 Top 通道贡献条形图；虚线标出重排幅度最大的片段。
- `*_top_channels.csv`：该片段的 Top 通道，精确给出 `layer / dimension / 被解释异常文本 / 通道自身关联文本 / 文本方向 / 学习门控 / 贡献值`。
- `*_summary.json`：图中被解释的片段编号及其 baseline 与重排后分数。

这个图借鉴两种常见可解释性展示：Network Dissection 的“高响应单元及其样本”和 TCAV 的“概念方向影响”。在视频场景中，我们不虚构像素热图，而是展示真实的时间片段和真实的 CLIP 原始通道贡献。
