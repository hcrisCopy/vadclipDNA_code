# CTNC-VAD：用 CLIP 内部通道图库重排冻结 VAD baseline

CTNC-VAD 的核心不是修改 VadCLIP，也不是向 encoder 注入残差：**从 CLIP 多层 hidden states 中找出少数“正常流形敏感”的原始通道，用这些通道对每一帧做最近正常状态检索，再结合冻结视觉—文本对齐，增强一个完全冻结的 VAD baseline 的排序和定位。**

```text
CLIP hidden states [T, 12, 768]                 冻结 VadCLIP 概率
              |                                         |
正常样本的 variance + truncated PCA                         |
              |                                         |
每层固定 64 个原始 layer-dimension 通道                    |
              |                                         |
normal channel gallery 的最近余弦距离 + 冻结 text score    |
              └──────── 只用标量 odds 校准并重排 ─────────┘
                                      |
                           最终异常排序 / 片段定位
```

## 为什么这仍然是通道级可解释性

`discover` 只看纯正常训练视频。

1. 每个 hidden 通道计算正常方差；高方差通道不是噪声，而是持续描述正常场景、动作和语义的活跃坐标。
2. 对全 768 维正常 hidden 做 rank-16 随机化 truncated PCA。PCA 不作为最终特征；只计算每个**原始坐标**被正常主流形解释的能量。
3. 每层按“正常方差 + PCA 坐标能量”硬选 64 个原始维度，共 `12 × 64 = 768` 个；冻结 hidden→visual→text 关联只作小权重的同分选择，保证选中通道既活跃又与异常文本有关。每个维度记录最相关的异常文本及方向。
4. 将正常帧的这 768 个原始维度存成紧凑的、按场景划分的 normal gallery。测试帧只在同一组原始维度上寻找最近的正常向量，使用余弦距离作为视觉异常分数。
5. 用 CLIP 最后一层的固定 `ln_post + projection + text` 路径得到文本异常分数。视觉分数回答“是否离开正常流形”，文本分数回答“是否有异常语义”。

PCA/SVD 仅帮助选择坐标；它从不输出“第几个主成分”作为判定。一次检测的解释可以精确还原为：

```text
帧 → normal context → 最近 normal gallery 向量
   → top-k layer-dimension 的归一化坐标差
   → 该维度的 normal variance / PCA energy / 对应文本
   → visual score、text score、冻结 baseline odds 的标量校准
```

这借鉴 LAKE 的“高方差敏感神经元 + normal gallery + cross-modal probing”思想，但这里的 token 是长视频中的帧段，并且输出是可外挂到多种 VAD baseline 的冻结重排器。

## 通用性和公平性

- VadCLIP 的视觉编码器、文本编码器、时序模块、分类头、checkpoint 均冻结；不修改 `VadCLIP/`。
- 通道发现不读 baseline 打分，也不使用异常视频标签来挑选通道；它只使用纯正常样本和冻结 CLIP。
- 训练只学习视觉图库分数、文本分数、最终 odds 融合的少数标量；没有 MLP、adapter 或训练新 embedding。
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

### 2. 训练冻结 baseline 的标量重排器

`--clean` 只清理 `training/`。中断后去掉 `--clean` 并加 `--resume` 即可继续。

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

日志中的 `selected-channel AP` 是图库 + 文本通道读出本身的定位 AP；`final ap2` 才是冻结 baseline 经重排后的官方 AP2。正确对齐的当前 VadCLIP baseline 是 `0.845045`。

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

每个 `audit/xd_test/*.npz` 记录 `visual_score`、`semantic_score`、最近 normal gallery index 和 `top_circuit_index` / `top_circuit_deviation`。配合文件内的 `selected_layers`、`selected_dimensions`、`selected_text_class`，即可直接说明某一帧为什么被重排。
