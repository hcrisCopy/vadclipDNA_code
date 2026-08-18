# CTNC-VAD：用 CLIP hidden 通道重排冻结 VadCLIP

本项目当前的正式方案是 **CTNC-VAD（Text-Conditioned Neural Circuit VAD）**。目标不是修改、微调或替换 VadCLIP，而是在其完全冻结的前提下，利用 CLIP 多层 hidden states 中可解释的通道证据，改善最终的异常排序、定位和类别定位。

旧的 `xd_dna/` 目录保留为历史代码；正式实验只运行本项目自己的 `ctnc_vad/`，不会跨项目 import 代码。

## 核心想法

现有 VAD 方法通常把 CLIP 当作固定特征提取器，再不断堆叠自己的时序/分类分支。CTNC 的出发点不同：CLIP 的视觉编码器已经在预训练中学到与文本概念相关的内部坐标，而正常视频还提供了“某个场景下该坐标应处于什么状态”的参照。

CTNC 将每个最终分数的变化分解为可检查的链条：

```text
冻结 VadCLIP 的 prompt 概率 [normal, class_1, ...]
                    +
多层 CLIP hidden state [T, 12, 768]
                    |
离线发现：层 / 通道 / 对应异常文本 / 正负方向
                    |
测试时：检索相似正常帧原型后的反事实状态偏离 + 状态转移惊异度
                    |
每个文本类别各自得到 hidden evidence
                    |
log p_frozen(class) + hidden likelihood factor(class)
                    |
softmax 后的类别概率与异常分数（排序/定位）
```

这不是 residual 注入：CTNC 不把向量写回 VadCLIP 的编码器、时序模块或分类头。它是一个外挂的 **prediction-space product-of-experts re-ranker**。任何能输出“正常 + 多个异常文本类别”概率的冻结 VAD baseline 都可以接入同一接口；若 baseline 只有二分类输出，也可使用一个异常文本集合的聚合概率作为兼容接口。

## 为什么可解释

每一个保留通道都有固定元数据：

- CLIP 层号和 hidden 维度；
- 它经冻结 CLIP `hidden → visual projection → text` 路径与哪一个异常文本最相关；
- 该通道朝向该文本是正向还是负向；
- 当前视频最相近的纯正常场景 context；
- 当前帧匹配到的真实正常 hidden 原型，以及相对该原型的有符号状态偏离；
- 当前帧相对该 context 的转移惊异度（该通道变化是否不符合正常运动）；
- 训练后该“通道—文本类别”的 state gate、transition gate、显式语义校正和类别 rank-scale。

因此，对某帧“为什么提高 shooting / explosion 概率”可以直接导出贡献最大的 `layer-dimension-text-direction`，而不是解释黑箱 residual 特征。`ctnc_vad.audit` 会逐视频写出这些证据。

## 两个阶段

### 1. `discover`：只做一次、与 baseline 分数无关

输入训练集的 reusable hidden states `[T,12,768]` 和训练集视频标签。

1. 只用纯正常视频聚类 scene context，并为每个 context 估计通道正常均值/标准差和一小组真实正常 hidden 原型；
2. 用冻结 CLIP 的最终视觉投影和文本编码器，估计各层各维对异常文本的**有符号**影响；
3. 对每个异常文本分别计算视频级 weak-label hidden tail 统计，再与该文本的 signed semantic affinity 共同排序；每层按文本均衡地选取稀疏通道，避免高频类别占满候选集；
4. 写出 `channel_scores.csv` 和 `circuit_assets.pt`。

这一阶段不读取、也不阈值化 VadCLIP 的预测，因而没有 baseline 伪正/负样本构造。

### 2. `train`：冻结 baseline 的类别概率重排序

先缓存一次冻结 VadCLIP 的 `prob2_all`，它只是 sidecar 的输入，不构成标签。训练监督是数据集原始**视频级类别标签**：XD 的 `G-B2-B6` 会变成多标签 `[explosion, shooting, car accident]`；正常视频所有异常类为 0。

小型 reader 仅学习：

- 每个已发现的“通道—异常文本”对的 state gate 和 transition gate；
- 一个受冻结 CLIP 文本方向约束的、显式可导出的 state 校正量；
- 每个异常文本类别的 state/transition 证据尺度和 rank-scale。

其中 state 是 hidden 相对**最近的真实正常原型**的有符号偏离，transition 是相对正常 context 的局部变化惊异度；两者都是逐帧、逐 `layer-dimension-text` 可分解的线性证据。原型检索避免把同一场景中的不同正常姿态、镜头或运动模式粗暴压成一个均值。读出器以数据集原始视频类别训练一个没有隐藏 MLP 的 direct hidden MIL probe；它只含每个文本的温度/先验，迫使通道电路自身能够区分类别，而不是只学会跟随 baseline 概率。对类别 `c` 的最终融合是：

```text
hidden_evidence(c) = state_scale(c) × state_evidence(c)
                   + transition_scale(c) × transition_novelty_evidence(c)
log p_final(c) = log p_frozen(c)
               + rank_scale(c) × hidden_evidence(c)
```

随后对全部类别做 softmax。这样既能提高正确异常类别，也能压低与视频类别证据冲突的错误异常类别；最终 `1 - p_final(normal)` 用于帧级排序/定位，`p_final(all classes)` 用于 detection mAP。损失是视频级多类别 MIL + 正常视频帧约束 + 很小的冻结输出保持项 + 双 gate 稀疏项 + 语义校正锚定项。

## 数据与目录约束

从 `vadclipDNA_code` 运行。所有新产物只写到同级：

```text
../vadclipDNA_data/<dataset>_ctnc_vad/
  discovery/                 # 可复用通道发现资产
  baseline_cache/train/      # 一次性冻结 VadCLIP 概率缓存
  training/                  # reader checkpoint/history
  evaluation/                # 官方指标与逐视频预测
  audit/<split>/             # 通道级解释
```

隐藏状态是数据资产，不是跨项目代码依赖。manifest 中每项应指向 `[T,12,768]` 的 CLS hidden。默认遇到训练 manifest 缺失视频会记录并跳过；测试集缺失则会报错，避免官方 ground truth 对齐失效。

## XD-Violence 命令

服务器先进入项目并激活环境：

```bash
cd ~/autodl-tmp/vadclipDNA_code
conda activate dsanet
```

首次发现通道（已生成有效资产时可跳过）：

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
  --normal-prototype-count 64 \
  --prototype-frames-per-video 32 \
  --frames-per-video 128 \
  --tail-fraction 0.125 \
  --semantic-weight 0.5 \
  --ridge 0.001 \
  --std-floor 0.0001 \
  --seed 234 \
  --device cuda
```

训练 sidecar（`--clean` 只删除 `../vadclipDNA_data/xd_ctnc_vad/training/`，不会动 baseline 或 discovery）：

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
  --normal-frame-weight 0.25 \
  --preserve-weight 0.01 \
  --sparsity-weight 0.001 \
  --semantic-anchor-weight 0.05 \
  --hidden-mil-weight 1.0 \
  --verification-initial-logit -1.5 \
  --alignment crop_hidden \
  --seed 234 \
  --device cuda \
  --clean
```

每 epoch 会同时打印冻结 baseline 的官方 `AP2` 与 CTNC 后的 `AP2`。XD 当前正确的基线对齐值应为约 `0.845045`；这是保护评测序列顺序后的结果。最优 checkpoint 只按同一验证集上的官方 `AP2` 保存。

测试最优 checkpoint：

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

导出解释：

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
  --device cuda
```

`audit/xd_test/circuit_dimensions.csv` 是通道字典；每个视频 `.npz` 包含每帧各异常文本的 `class_evidence`、匹配的正常原型编号、原型残差，以及每类贡献最大的通道索引和数值。

## 评测公平性与开销

- VadCLIP checkpoint、视觉编码器、文本编码器、时序模块、分类头全部冻结；
- 不用 baseline 打分构建 hidden 通道的正负样本，也不以 baseline 分数作为训练标签；
- train/test 均保持 source CSV 原视频顺序，避免 `gt.npy` 错位；
- discovery 使用已有 hidden states，通常只需一次；训练阶段缓存冻结 baseline 一次后，reader 每 epoch 只处理 `[256, 384]` 稀疏通道和 `[256, 7]` 类别概率，远小于重新运行完整 VadCLIP；
- `evaluation/metrics.json` 给出 baseline、纯 hidden 通道和最终重排序的同协议指标，方便确认提升来自 CTNC 而不是评测变化。
