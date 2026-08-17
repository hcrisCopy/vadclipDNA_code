# NeuVAD-Lens（VadCLIP / XD-Violence）

这是 DNA 的低开销文本透镜版本。它不修改 VadCLIP 和 xd_dna：

- 复用 xd_dna 已完成的 selected_neurons.json、纯正常 DNA 统计量、CLIP hidden manifest 与原始 512D 特征；
- 对全部末层 768D CLS hidden 用冻结 CLIP 的 LN-post、projection 与固定类别文本计算软文本证据；
- DNA 残差与文本残差并行相加，训练和评测仍使用 VadCLIP 原来的损失、优化器、学习率、保存规则和指标；
- 不取 DNA/text 硬交集；top-k 仅用于审计展示，不裁剪模型输入。

从 vadclipDNA_code 目录运行。所有新产物写入 ../vadclipDNA_data/xd_neuvad_lens/；默认会复用已有完整产物。要重做某一步，只给该步增加 --clean；训练中断后给同一训练命令增加 --resume。

## 1. 构建冻结透镜资产

    python -m neuvad_lens.build_lens_assets \
      --dataset xd \
      --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
      --source-path-base . \
      --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
      --hidden-path-base . \
      --output-root ../vadclipDNA_data/xd_neuvad_lens \
      --clip-model ViT-B/16 \
      --last-layer-index -1 \
      --normal-subspace-dim 64 \
      --prototype-count 16 \
      --verify-videos 16 \
      --min-projection-cosine 0.995 \
      --seed 234 \
      --device cuda

以下命令使用 bash 的续行符。

输出：../vadclipDNA_data/xd_neuvad_lens/lens/lens_assets.pt。

## 2. 构建三段输入特征

这里复用已有 DNA 定位结果；不需要重新执行 score_pseudo、build_samples、cache_probe、localize。

    python -m neuvad_lens.build_features \
      --dataset xd \
      --split train \
      --source-csv ../vad_data/work_xd/xd_train_local.csv \
      --source-path-base . \
      --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
      --hidden-path-base . \
      --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
      --lens-assets ../vadclipDNA_data/xd_neuvad_lens/lens/lens_assets.pt \
      --output-root ../vadclipDNA_data/xd_neuvad_lens \
      --alignment crop_hidden \
      --allow-missing-hidden

    python -m neuvad_lens.build_features \
      --dataset xd \
      --split test \
      --source-csv ../vad_data/work_xd/xd_test_local.csv \
      --source-path-base . \
      --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
      --hidden-path-base . \
      --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
      --lens-assets ../vadclipDNA_data/xd_neuvad_lens/lens/lens_assets.pt \
      --output-root ../vadclipDNA_data/xd_neuvad_lens \
      --alignment crop_hidden

输出列表：../vadclipDNA_data/xd_neuvad_lens/lists/xd_neuvad_lens_train.csv 和 xd_neuvad_lens_test.csv。

每个特征顺序固定为：

    [DNA z-score selected channels | original 512D CLIP | raw final CLS 768D]

## 3. 正式训练

    python -m neuvad_lens.train \
      --train-list ../vadclipDNA_data/xd_neuvad_lens/lists/xd_neuvad_lens_train.csv \
      --test-list ../vadclipDNA_data/xd_neuvad_lens/lists/xd_neuvad_lens_test.csv \
      --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
      --lens-assets ../vadclipDNA_data/xd_neuvad_lens/lens/lens_assets.pt \
      --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
      --output-root ../vadclipDNA_data/xd_neuvad_lens \
      --gt-path VadCLIP/list/gt.npy \
      --gt-segment-path VadCLIP/list/gt_segment.npy \
      --gt-label-path VadCLIP/list/gt_label.npy \
      --max-epoch 10 \
      --batch-size 96 \
      --lr 1e-5 \
      --scheduler-milestones 3 6 10 \
      --scheduler-rate 0.1 \
      --num-workers 0 \
      --dna-hidden-dim 1024 \
      --dna-depth 3 \
      --text-hidden-dim 512 \
      --text-depth 2 \
      --text-temperature 0.07 \
      --seed 234 \
      --device cuda

输出：../vadclipDNA_data/xd_neuvad_lens/training/。最佳模型仍按 VadCLIP XD 的 AP2 保存为 model_best.pth。

## 4. 正式测试

    python -m neuvad_lens.test \
      --test-list ../vadclipDNA_data/xd_neuvad_lens/lists/xd_neuvad_lens_test.csv \
      --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
      --lens-assets ../vadclipDNA_data/xd_neuvad_lens/lens/lens_assets.pt \
      --model-path ../vadclipDNA_data/xd_neuvad_lens/training/model_best.pth \
      --init-baseline-model ../vadclip_data/model/vadclip_xd.pth \
      --output-root ../vadclipDNA_data/xd_neuvad_lens \
      --gt-path VadCLIP/list/gt.npy \
      --gt-segment-path VadCLIP/list/gt_segment.npy \
      --gt-label-path VadCLIP/list/gt_label.npy \
      --num-workers 0 \
      --dna-hidden-dim 1024 \
      --dna-depth 3 \
      --text-hidden-dim 512 \
      --text-depth 2 \
      --text-temperature 0.07 \
      --device cuda

输出：../vadclipDNA_data/xd_neuvad_lens/evaluation/metrics.json 和可续跑的逐视频预测。

## 5. 导出可解释性证据

    python -m neuvad_lens.audit \
      --feature-list ../vadclipDNA_data/xd_neuvad_lens/lists/xd_neuvad_lens_test.csv \
      --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
      --lens-assets ../vadclipDNA_data/xd_neuvad_lens/lens/lens_assets.pt \
      --output-root ../vadclipDNA_data/xd_neuvad_lens \
      --split-name test \
      --topk 8 \
      --text-temperature 0.07 \
      --device cuda

输出：../vadclipDNA_data/xd_neuvad_lens/lens_audit/test/*.npz。文件含每帧类别软路由、正常性距离和各类别 top-8 正文本贡献；这些 top-8 仅用于展示，模型训练和推理始终使用完整 768D 文本证据。
