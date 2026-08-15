# XD DNA on VadCLIP

这套代码将 `DSANet_DNA` 的“冻结 CLIP 中间层 + 线性探针 + 激活×梯度×权重”神经元定位迁到本目录的未修改 `VadCLIP` baseline。仅覆盖 XD-Violence。

方法约定：异常视频的 VadCLIP `logits1` 高分片段为伪正样本；负样本严格只来自标签为 `A` 的纯正常视频。12 个 ViT-B/16 CLS 层分别按 triadic 分数选 `--topk-per-layer` 个神经元，默认每层 64 个，即 768D。其 z-score 特征与同时间步官方 512D CLIP 特征拼为 1280D，经过新建的残差支路后再喂给原始 512D VadCLIP。`VadCLIP/` 内没有任何修改。

## 可复用的数据

CLIP 实现相同：`VadCLIP/src/clip/model.py`、`vad_code/DSANet/src/clip/model.py` 的哈希一致。可复用的是 DSANet 已抽取的**数据资产**，而不是 DSANet 代码或模型：

```text
../vad_data/work_xd/
  xd_train_local.csv
  xd_test_local.csv
  clip_hidden_stride16_train_8gpu/manifest.csv
  clip_hidden_stride16_train_8gpu/features/*.npz
  clip_hidden_stride16_test_8gpu/manifest.csv
  clip_hidden_stride16_test_8gpu/features/*.npz
```

每个 manifest 的 `hidden_path` 指向一个 `hidden` 数组，契约为 `[T, 12, 768]`、`token_pool=cls`、`stride=16`。上述位置是 DSANet 已完成隐藏特征导出的标准输出位置；本地当前工作区未包含 `vad_data`，因此正式服务器运行时传入实际挂载后的相对路径即可。若 manifest 保留了旧机器路径，请同时传 `--hidden-prefix-from`、`--hidden-prefix-to`，只改数据路径映射，不会引用任何其他项目代码。

不可复用：DSANet checkpoint、DSANet 伪分数、旧选中神经元、旧派生特征、DSANet 训练权重。新的伪分数由 VadCLIP XD 512D checkpoint 生成。

所有新产物都在同级 `../vadclipDNA_data/xd_normal_negative_top64/`。重复运行默认验证并复用单视频/单分片产物；某阶段需要从头重做时只给该命令加 `--clean`。训练中断后给同一训练命令加 `--resume`。测试逐视频预测会自动续跑。

## XD 正式命令

从 `vadclipDNA_code` 运行。请把 `../vad_data`、`../vadclip_data/model/vadclip_xd.pth` 替换为服务器实际的同级数据挂载路径；命令中所有路径均为相对路径。

```bash
python -m xd_dna.score_pseudo \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --device cuda

python -m xd_dna.build_samples \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadclipDNA_data/xd_normal_negative_top64/pseudo_scores/group_scores.csv \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --validation-fraction 0.20 \
  --top-p 0.10 \
  --min-positive-per-video 3 \
  --max-positive-per-video 32 \
  --seed 234

python -m xd_dna.cache_probe \
  --samples-csv ../vadclipDNA_data/xd_normal_negative_top64/samples/samples.csv \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --videos-per-shard 25

python -m xd_dna.localize \
  --cache ../vadclipDNA_data/xd_normal_negative_top64/cache/probe_cache.npz \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --device cuda \
  --probe-epochs 100 \
  --probe-lr 1e-2 \
  --probe-weight-decay 1e-4 \
  --topk-per-layer 64 \
  --seed 234
```

`localization/selected_neurons.json` 记录每层选择结果、pure-normal 负样本约束、normal mean/std 和 768D+512D 输入契约。把 `--topk-per-layer 64` 改为其他正整数即可控制每层神经元数；总维度为 `12 × topk-per-layer`。

```bash
python -m xd_dna.build_features \
  --split train \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --alignment crop_hidden \
  --allow-missing-hidden

python -m xd_dna.build_features \
  --split test \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --alignment crop_hidden

python -m xd_dna.train \
  --train-list ../vadclipDNA_data/xd_normal_negative_top64/lists/xd_concat_train.csv \
  --test-list ../vadclipDNA_data/xd_normal_negative_top64/lists/xd_concat_test.csv \
  --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --scheduler-milestones 3 6 10 \
  --scheduler-rate 0.1 \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --seed 234 \
  --device cuda

python -m xd_dna.test \
  --test-list ../vadclipDNA_data/xd_normal_negative_top64/lists/xd_concat_test.csv \
  --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
  --model-path ../vadclipDNA_data/xd_normal_negative_top64/training/model_best.pth \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

训练严格沿用 XD VadCLIP 的 `AdamW`、`MultiStepLR(3,6,10; gamma=0.1)`、`CLAS2 + CLASM + 1e-4×prompt-orthogonality` 和每 epoch 验证；最佳模型按官方 XD 的语言分支 `AP2` 保存为 `training/model_best.pth`。评测使用官方的帧级 `AUC1/AP1/AUC2/AP2` 和 `utils/xd_detectionMAP.py` detection mAP，结果在 `evaluation/metrics.json`。

主要产物：

```text
../vadclipDNA_data/xd_normal_negative_top64/
  pseudo_scores/group_scores.csv
  samples/samples.csv
  cache/shards/*.npz
  cache/probe_cache.npz
  localization/selected_neurons.json
  localization/neuron_scores.csv
  features/train/*.npy
  features/test/*.npy
  lists/xd_concat_train.csv
  lists/xd_concat_test.csv
  training/checkpoint_last.pth
  training/model_best.pth
  training/history.csv
  evaluation/predictions/*.npz
  evaluation/metrics.json
```
