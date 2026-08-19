# CTSC-VAD：时序化的可解释 CLIP 通道电路

## 一句话

VadCLIP 完全冻结。CTSC 不改它的权重、不注入残差，只读取已有 CLIP hidden states 中少数真实的 `(layer, dimension)`，用这些通道的**时间变化**去补充 VadCLIP 的排序和类别定位。

旧 CTSC 的问题是：只看某一段的静态通道值，然后始终和 VadCLIP 做 PoE。实验已经证明这样会污染强 baseline。

新 CTSC 的原则是：

```text
原始 CLIP hidden 通道 (Layer l, Dimension d)
                     |
正常场景的静态 + 动态参考
                     |
五种固定、可命名的时间证据
  1. 朝异常文本方向的通道值
  2. 朝异常文本方向的突变
  3. 背异常文本方向的突变
  4. 短期状态相对长期状态的偏离
  5. 持续存在的文本方向通道值
                     |
每个类别稀疏选择少数「原始通道 × 时间证据」
                     |
证据必须高、类别明确、且邻近片段也支持
                     |
只提升该类别的 VadCLIP 排名；没有证据时完全保持 VadCLIP
```

因此一条解释可以直接写成：

> `Layer 8, Dimension 312` 在第 57 段出现“朝 shooting 文本方向的突变”；它的直接权重和持续性证据共同满足证书，所以仅提升这一段的 shooting 排名。

没有 MLP、adapter、attention 或隐藏投影。PCA/SVD 只用于从 12×768 个**原始坐标**中找候选，最终分类和解释从不读取 PCA 主成分。

## 必须重新发现通道

本次资产版本从 v1 升级为 v2：增加了正常场景下每个原始通道的速度、短长期变化参考。旧的 `ctsc_assets.pt` 不能复用，这是故意的保护机制。

以下命令在 `vadclipDNA_code` 根目录运行，所有输出在同级相对目录 `../vadclipDNA_data/xd_ctsc_vad/`。

### 1. 发现原始通道和正常时序参考

只使用训练集中正常视频来建立参考；不读取 VadCLIP 分数，也不以 baseline 打分构造正负样本。

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
  --temporal-short-window 5 \
  --temporal-long-window 21 \
  --temporal-persistence-window 5 \
  --std-floor 0.0001 \
  --ridge 0.001 \
  --seed 234 \
  --device cuda \
  --clean
```

重点看 `discovery/selected_raw_channels.csv`。每一行就是一个真实的 CLIP 层和维度，不是主成分。

### 2. 训练外接时序通道电路

`--fusion-initial-logit -5.0` 很重要：开始时几乎等于原始 VadCLIP，只有独立通道电路学到可靠证据才会逐步提升某些片段。`--temporal-separation-*` 取代旧的平滑损失，防止异常证据被抹成整段视频都一样高。

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
  --temporal-separation-weight 0.05 \
  --temporal-separation-margin 0.20 \
  --gate-initial-logit -2.0 \
  --fusion-initial-logit -5.0 \
  --alignment crop_hidden \
  --num-workers 0 \
  --seed 234 \
  --device cuda \
  --clean
```

每个 epoch 都打印与 VadCLIP 一致的 baseline AP2/dMAP，以及 circuit-only 和最终 CTSC 的 AUC、AP、dMAP。最终仍按官方同一测试流程选择 best checkpoint；中断后去掉 `--clean`、加入 `--resume` 即可继续。

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

终端会完整输出官方 VadCLIP 指标：`AUC1/AP1`、`AUC2/AP2`、每个类别的 `mAP@...` 和 `average MAP`。对比 `[Frozen VadCLIP]` 和 `[CTSC certified promotion]`。

### 4. 导出逐通道解释和论文图

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

每张图同时显示：冻结 baseline 与最终排序、通道电路分数、实际提升量、Top 原始通道热图和该帧的精确通道证据。相邻的 `*_top_channels.csv` 会写出层、维度、文本方向、主导时间算子、直接权重和贡献。

## 额外开销

已有 hidden states 不用重新提取。新增的是每个选中通道的固定窗口平均和直接读出：默认 `1536 通道 × 5 算子 × 6 类`，约 4.6 万个可训练的标量权重，没有 Transformer、图网络或额外 CLIP 前向。发现阶段多一次正常视频的时序统计；训练和测试阶段不检索样本库。
