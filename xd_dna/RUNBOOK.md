# DNA-on-VadCLIP 运行手册（可复用产物）

从 `vadclipDNA_code` 目录执行全部命令：

```bash
cd /root/autodl-tmp/vadclipDNA_code
```

本包不导入 DSANet 或其他项目代码。已经完成的 512D CLIP `.npy` 与
`[T, 12, 768]` hidden `.npz` 只是**输入数据资产**，通过命令行路径读取；所有
新产物只写到同级 `../vadclipDNA_data/`。

## 产物复用规则

- 默认直接重跑同一条命令会验证并复用已完成的单视频、分片或阶段产物。
- 不要在输入、参数、模型 checkpoint 或神经元选择改变后混用旧产物。
- 只对需要重算的阶段追加 `--clean`；它仅删除该阶段，不会删除整个输出根目录。
- 训练中断后，只有**代码、输入特征和训练参数均未改变**时才追加 `--resume`。
- 测试默认复用 `evaluation/predictions/*.npz`。模型权重变更后，必须给测试加
  `--clean`（或 `--no-resume`），使预测缓存与新模型一致。
- 残差的 padding 清零逻辑已修正。此前用旧 `xd_dna/model.py` 训练的模型不能
  `--resume`；可复用定位和融合特征，但要重新训练并重新测试。

默认的 CSV / hidden manifest 保留了 DSANet 原任务的相对路径约定：其中的相对路径
相对命令启动目录解释，而不是相对 CSV 或 manifest 文件解释。因此从本目录运行时保留
`--source-path-base .` 与 `--hidden-path-base .`。

---

## XD-Violence

输出根目录：

```text
../vadclipDNA_data/xd_normal_negative_top64/
```

### 1. 用冻结 VadCLIP 生成伪分数

```bash
python -m xd_dna.score_pseudo \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --source-path-base . \
  --baseline-model ../vadclip_data/model/vadclip_xd.pth \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --device cuda
```

复用：`pseudo_scores/scores/*.npy` 和 `pseudo_scores/group_scores.csv`。

### 2. 构建 probe 样本

异常视频中 VadCLIP 高伪分片段为正样本；负样本仅来自标签 `A` 的纯正常视频。

```bash
python -m xd_dna.build_samples \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --pseudo-csv ../vadclipDNA_data/xd_normal_negative_top64/pseudo_scores/group_scores.csv \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --validation-fraction 0.20 \
  --top-p 0.10 \
  --min-positive-per-video 3 \
  --max-positive-per-video 32 \
  --seed 234
```

复用：`samples/samples.csv`。样本构建参数改变后，应依次重算 samples、cache、
localization、features、training 与 evaluation。

### 3. 建立 probe cache

```bash
python -m xd_dna.cache_probe \
  --dataset xd \
  --samples-csv ../vadclipDNA_data/xd_normal_negative_top64/samples/samples.csv \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --videos-per-shard 25
```

复用：`cache/shards/*.npz` 与 `cache/probe_cache.npz`。

### 4. 定位每层 top-k 神经元

`--topk-per-layer 64` 表示 12 层共 768D；它可以改为任意正整数。

```bash
python -m xd_dna.localize \
  --dataset xd \
  --cache ../vadclipDNA_data/xd_normal_negative_top64/cache/probe_cache.npz \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --device cuda \
  --probe-epochs 100 \
  --probe-lr 1e-2 \
  --probe-weight-decay 1e-4 \
  --topk-per-layer 64 \
  --seed 234
```

复用：`localization/selected_neurons.json`、normal mean/std、probe 与 neuron 表。
`topk` 或 probe 参数改变后，后续融合特征、训练和测试都必须重做。

### 5. 构建 `[DNA | 512D CLIP]` 融合特征

XD train 隐藏特征存在已知缺项时可加 `--allow-missing-hidden`；测试集不允许跳过。

```bash
python -m xd_dna.build_features \
  --dataset xd \
  --split train \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --source-path-base . \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --alignment crop_hidden \
  --allow-missing-hidden

python -m xd_dna.build_features \
  --dataset xd \
  --split test \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --source-path-base . \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --neuron-json ../vadclipDNA_data/xd_normal_negative_top64/localization/selected_neurons.json \
  --output-root ../vadclipDNA_data/xd_normal_negative_top64 \
  --alignment crop_hidden
```

复用：`features/train/*.npy`、`features/test/*.npy` 与 `lists/xd_concat_{train,test}.csv`。

### 6. 训练残差支路

主干 VadCLIP 参数冻结；只训练 DNA LayerNorm、MLP 和 gate。最佳模型按 XD 官方
语言分支 `AP2` 选择。

```bash
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
```

产物：`training/checkpoint_last.pth`、`training/model_best.pth`、`training/history.csv`。

### 7. 最终测试

```bash
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

输出：`evaluation/metrics.json`，包含 `AUC1/AP1/AUC2/AP2` 与 detection mAP。

---

## UCF-Crime

输出根目录：

```text
../vadclipDNA_data/ucf_normal_negative_top64/
```

以下命令与 XD 的阶段相同；UCF 的纯正常标签是 `Normal`。UCF 不使用
`--allow-missing-hidden`。

### 1–4. 伪分数、样本、cache、神经元定位

```bash
python -m xd_dna.score_pseudo \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --source-path-base . \
  --baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --device cuda

python -m xd_dna.build_samples \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --pseudo-csv ../vadclipDNA_data/ucf_normal_negative_top64/pseudo_scores/group_scores.csv \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --validation-fraction 0.20 \
  --top-p 0.10 \
  --min-positive-per-video 3 \
  --max-positive-per-video 32 \
  --seed 234

python -m xd_dna.cache_probe \
  --dataset ucf \
  --samples-csv ../vadclipDNA_data/ucf_normal_negative_top64/samples/samples.csv \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --videos-per-shard 25

python -m xd_dna.localize \
  --dataset ucf \
  --cache ../vadclipDNA_data/ucf_normal_negative_top64/cache/probe_cache.npz \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --device cuda \
  --probe-epochs 100 \
  --probe-lr 1e-2 \
  --probe-weight-decay 1e-4 \
  --topk-per-layer 64 \
  --seed 234
```

### 5. 构建 UCF 融合特征

```bash
python -m xd_dna.build_features \
  --dataset ucf \
  --split train \
  --source-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --source-path-base . \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --hidden-path-base . \
  --neuron-json ../vadclipDNA_data/ucf_normal_negative_top64/localization/selected_neurons.json \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --alignment crop_hidden

python -m xd_dna.build_features \
  --dataset ucf \
  --split test \
  --source-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --source-path-base . \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --hidden-path-base . \
  --neuron-json ../vadclipDNA_data/ucf_normal_negative_top64/localization/selected_neurons.json \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --alignment crop_hidden
```

### 6. 训练 UCF 残差支路

UCF 每步拼接等量 `Normal` 与异常 batch。训练按原始 VadCLIP UCF 的实际保存规则，
以 `AUC1` 选择最佳模型。

```bash
python -m xd_dna.train_ucf \
  --train-list ../vadclipDNA_data/ucf_normal_negative_top64/lists/ucf_concat_train.csv \
  --test-list ../vadclipDNA_data/ucf_normal_negative_top64/lists/ucf_concat_test.csv \
  --neuron-json ../vadclipDNA_data/ucf_normal_negative_top64/localization/selected_neurons.json \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --gt-path VadCLIP/list/gt_ucf.npy \
  --gt-segment-path VadCLIP/list/gt_segment_ucf.npy \
  --gt-label-path VadCLIP/list/gt_label_ucf.npy \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 2e-5 \
  --scheduler-milestones 4 8 \
  --scheduler-rate 0.1 \
  --eval-interval-samples 1280 \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --seed 234 \
  --device cuda
```

### 7. 最终测试

```bash
python -m xd_dna.test_ucf \
  --test-list ../vadclipDNA_data/ucf_normal_negative_top64/lists/ucf_concat_test.csv \
  --neuron-json ../vadclipDNA_data/ucf_normal_negative_top64/localization/selected_neurons.json \
  --model-path ../vadclipDNA_data/ucf_normal_negative_top64/training/model_best.pth \
  --init-baseline-model ../vadclip_data/model/vadclip_ucf.pth \
  --output-root ../vadclipDNA_data/ucf_normal_negative_top64 \
  --gt-path VadCLIP/list/gt_ucf.npy \
  --gt-segment-path VadCLIP/list/gt_segment_ucf.npy \
  --gt-label-path VadCLIP/list/gt_label_ucf.npy \
  --num-workers 0 \
  --residual-hidden-dim 1024 \
  --residual-depth 3 \
  --device cuda
```

输出：`evaluation/metrics.json`，包含 `AUC1/AP1/AUC2/AP2`、`Ano-AUC1/Ano-AUC2`
和 detection mAP。

---

## 常见的最小重算范围

| 变化 | 可复用 | 必须重算 |
|---|---|---|
| 训练中断，无代码/参数变化 | 所有前置产物 | `train* --resume`，随后按需测试 |
| 残差模型、训练参数或 baseline checkpoint 改变 | pseudo、samples、cache、localization、features | training、evaluation |
| 仅测试模型权重改变 | 全部前置产物 | evaluation（加 `--clean`） |
| `topk-per-layer` 或 probe 参数改变 | pseudo、samples、cache | localization、features、training、evaluation |
| 样本规则、seed、伪分数 checkpoint 改变 | 无下游阶段 | samples 起的所有后续阶段 |
| 512D/hidden 输入或其路径映射改变 | 无法安全假定旧产物可用 | 从 pseudo 或 samples 起按受影响输入重算 |

例如，因 padding 修复而重新训练 UCF 时，只需：

```bash
python -m xd_dna.train_ucf ... --clean
python -m xd_dna.test_ucf ... --clean
```

其中 `...` 替换为上文完整训练或测试参数；不要对已验证的 pseudo、cache、
localization 或 features 添加 `--clean`。
