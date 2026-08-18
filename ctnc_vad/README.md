# CTNC-VAD：可解释 CLIP hidden 通道验证器

CTNC-VAD 的目标不是替换 VadCLIP，也不修改 VadCLIP 的任何权重。

```text
冻结 VadCLIP 先给出逐帧排序
              +
CLIP 多层 hidden states 找出“指向异常文本”的稀疏通道
              ↓
CTNC 对每帧给出：保留 / 抑制假阳性 / 提升可信漏检
              ↓
只有 hidden 证据足够时才改排序；证据不足时保留 VadCLIP
```

这不是 residual 注入：不向 baseline 特征、logit 或权重注入新特征。CTNC 是一个冻结 baseline 外部的、动作可审计的排序验证器。

## 方法设计

### 1. 发现带方向的可解释通道

输入已有 CLS hidden：`[时间, 12 层, 768 维]`。

对每个 `(layer, dimension)`，通过冻结 CLIP 的 hidden-to-text 路由计算它朝“异常文本 − 正常文本”移动的**符号方向**。随后结合训练视频的弱标签，选出每层少量通道（XD 默认每层 32 个）。

每个选中通道都有可读属性：层号、维度、最相关异常文本和正/负方向。它们写入 `discovery/channel_scores.csv`。

### 2. 通道证据

对测试帧，CTNC 将选中通道与语义相近正常场景的正常统计比较；但不再使用无方向的 `|z|` 异常分数，而是计算**带文本方向的偏移**。多层同向时证据更可靠。

### 3. 冻结 baseline 的认证式排序

VadCLIP 分数只是 CTNC 的输入，不参与构造伪正/伪负样本。

小型 verifier 只输出三种动作的概率：

- `keep`：hidden 证据不够，原样保留 VadCLIP；
- `suppress`：内部通道证明该高分段更像正常，抑制假阳性；
- `promote`：多层通道和文本方向共同支持异常，提升可信漏检。

训练使用正常视频的全帧负监督和异常视频的视频级 MIL 监督。它缓存一次冻结 VadCLIP 训练分数作为输入，**不会**用 baseline top-k 分数制造伪标签。

## 输入和边界

- source CSV：`path,label`，`path` 是 VadCLIP 的 512D feature。
- hidden manifest：`key,hidden_path`，每个 hidden 文件含 `hidden=[T,12,768]`。
- XD 训练 CSV 的 `__0`…`__9` 是同一视频增强；CTNC 将其合为一条 hidden trajectory，只用第一条 feature 做时间对齐。
- discovery/train 默认取 source CSV 与 hidden manifest 的交集，并记录缺失训练 hidden；测试严格匹配，避免漏评。
- 新产物只写入同级 `../vadclipDNA_data/`，不修改 `VadCLIP/`。

## XD-Violence

在 `vadclipDNA_code` 目录运行。首次使用新版 verifier 时，必须依次重新运行 discovery、train、test；旧版 checkpoint 与旧 prediction 不能复用。

### 1. 发现 signed hidden channels

```bash
python -m ctnc_vad.discover \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --output-root ../vadclipDNA_data/xd_ctnc_vad \
  --clip-model ViT-B/16 \
  --candidate-per-layer 32 \
  --context-count 16 \
  --context-iters 50 \
  --frames-per-video 128 \
  --tail-fraction 0.125 \
  --semantic-weight 0.5 \
  --ridge 0.001 \
  --std-floor 0.0001 \
  --seed 234 \
  --device cuda \
  --clean
```

输出：`../vadclipDNA_data/xd_ctnc_vad/discovery/circuit_assets.pt`、`channel_scores.csv`、`missing_hidden.csv`。

### 2. 训练 verifier

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
  --scheduler-milestones 6 9 \
  --scheduler-rate 0.1 \
  --num-workers 0 \
  --normal-frame-weight 0.25 \
  --preserve-weight 0.02 \
  --sparsity-weight 0.001 \
  --gate-initial-logit 0.0 \
  --keep-initial-logit 5.0 \
  --alignment crop_hidden \
  --seed 234 \
  --device cuda \
  --clean
```

首次训练会生成一次 `baseline_cache/train/*.npz`，以后同一 baseline checkpoint 会自动复用。训练每 epoch 按 VadCLIP 同一验证 AP 规则保存 `training/model_best.pth`。

### 3. 测试

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
  --alignment crop_hidden \
  --device cuda \
  --clean
```

输出：`evaluation/metrics.json`、可续跑的 `evaluation/predictions/*.npz`。它同时报告 frozen baseline、signed channel evidence、verified score 的 AUC/AP/dMAP。

### 4. 导出解释

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
  --split-name test \
  --topk 8 \
  --alignment crop_hidden \
  --device cuda \
  --clean
```

输出：`audit/test/`。每个视频记录每帧 signed evidence、层级证据、正常场景、top hidden channels；`circuit_dimensions.csv` 给出每个通道对应的异常文本与方向。

## UCF-Crime

命令结构不变，只替换 `--dataset ucf`、UCF 的 CSV/hidden manifest/GT/VadCLIP checkpoint，以及输出根目录为 `../vadclipDNA_data/ucf_ctnc_vad`。UCF 默认 batch size 为 64；其余方法逻辑没有数据集专用分支。
