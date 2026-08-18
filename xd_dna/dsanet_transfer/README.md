# DSANet 神经元迁移到 VadCLIP（XD-Violence）

这个实验不再让 VadCLIP 的伪分数参与 DNA 定位。神经元身份完全来自 DSANet 的
`fdu_indices.json`，FDU 激活直接复用 DSANet 已导出的 `[T,K]` 特征。VadCLIP 只接收
`[z-score FDU K 维 | 原始 CLIP 512 维]`，并沿用现有残差支路、官方 XD 损失、AP2 最优模型选择和评测。

迁移前会严格检查：两边都是 CLIP ViT-B/16、CLS token、DSANet FDU 指纹一致、完整
chunk 名和标签一致，以及每个 FDU 与 VadCLIP 特征的时间长度 `T` 完全一致。不会裁剪、补零或重新定位。
旧版 DSANet `fdu_indices.json` 若没有 `token_pool` 字段，会按其历史 CLS-only 导出契约兼容，并在新 contract 中记录推断来源。若旧导出目录没有 `export_spec.json`，代码会进入兼容模式：检查每个 FDU 的维度和时间轴，并在摘要中记录该限制；有 `export_spec.json` 时仍执行完整 FDU 指纹校验。

从 `vadclipDNA_code` 目录运行。下面的相对路径对应 DSANet 已完成 XD FDU 导出的标准产物；如服务器的数据挂载目录不同，只替换输入相对路径，不要改代码。

```bash
python -m xd_dna.dsanet_transfer.prepare \
  --dsanet-fdu-json ../DSANet_DNA/outputs/xd/localization/fdu_indices.json \
  --output-root ../vadclipDNA_data/xd_dsanet_neuron_transfer

python -m xd_dna.dsanet_transfer.build_features \
  --split train \
  --source-list ../vad_data/work_xd/xd_train_local.csv \
  --source-path-base . \
  --dsanet-fdu-manifest ../DSANet_DNA/outputs/xd/fdu_features/train/aligned_features.csv \
  --fdu-path-base . \
  --fdu-dir ../DSANet_DNA/outputs/xd/fdu_features/train/features \
  --neuron-contract ../vadclipDNA_data/xd_dsanet_neuron_transfer/contract/dsanet_transfer_neurons.json \
  --output-root ../vadclipDNA_data/xd_dsanet_neuron_transfer \
  --normal-label A \
  --allow-missing-fdu

python -m xd_dna.dsanet_transfer.build_features \
  --split test \
  --source-list ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --dsanet-fdu-manifest ../DSANet_DNA/outputs/xd/fdu_features/test/aligned_features.csv \
  --fdu-path-base . \
  --fdu-dir ../DSANet_DNA/outputs/xd/fdu_features/test/features \
  --neuron-contract ../vadclipDNA_data/xd_dsanet_neuron_transfer/contract/dsanet_transfer_neurons.json \
  --output-root ../vadclipDNA_data/xd_dsanet_neuron_transfer \
  --normal-label A
```

`train` 只用训练集标签为 `A` 的 DSANet FDU 序列计算均值和标准差。统计按视频分片原子保存，数据加载中断后会从已完成的统计分片和已验证的融合特征继续。

```bash
python -m xd_dna.dsanet_transfer.train \
  --train-list ../vadclipDNA_data/xd_dsanet_neuron_transfer/lists/xd_dsanet_transfer_train.csv \
  --test-list ../vadclipDNA_data/xd_dsanet_neuron_transfer/lists/xd_dsanet_transfer_test.csv \
  --neuron-json ../vadclipDNA_data/xd_dsanet_neuron_transfer/contract/dsanet_transfer_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_dsanet_neuron_transfer \
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

python -m xd_dna.dsanet_transfer.test \
  --test-list ../vadclipDNA_data/xd_dsanet_neuron_transfer/lists/xd_dsanet_transfer_test.csv \
  --neuron-json ../vadclipDNA_data/xd_dsanet_neuron_transfer/contract/dsanet_transfer_neurons.json \
  --model-path ../vadclipDNA_data/xd_dsanet_neuron_transfer/training/model_best.pth \
  --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_dsanet_neuron_transfer \
  --gt-path VadCLIP/list/gt.npy \
  --gt-segment-path VadCLIP/list/gt_segment.npy \
  --gt-label-path VadCLIP/list/gt_label.npy \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

训练中断后，在同一训练命令末尾加 `--resume`。融合阶段和评测逐视频自动复用有效产物；想只重做当前阶段，在对应命令末尾加 `--clean`。`--clean` 不会改动 `VadCLIP/`、`DSANet_DNA/` 或其他实验目录。

产物全部位于 `../vadclipDNA_data/xd_dsanet_neuron_transfer/`：

```text
contract/dsanet_transfer_neurons.json
features/normal_stats.npz
features/normal_stat_shards/
features/train/  features/test/
lists/xd_dsanet_transfer_train.csv
lists/xd_dsanet_transfer_test.csv
training/checkpoint_last.pth  training/model_best.pth  training/history.csv
evaluation/predictions/  evaluation/metrics.json
```
