# CTNC-VAD：冻结 CLIP 内部正常性电路 + 分数级重排

CTNC-VAD 不训练或修改 VadCLIP，也不使用 DNA 的 `top-k baseline score` 伪正样本。它读取已有的 CLIP CLS hidden states `[T,12,768]`，从正常视频、视频级弱标签和冻结 CLIP 文本路由中发现稀疏 hidden circuit；测试时，VadCLIP 先输出原始分数，CTNC 再只在**分数排序层**做解释驱动的重排。

从 `vadclipDNA_code` 目录运行。新产物全部位于同级的 `../vadclipDNA_data/`。本包不导入 `xd_dna`、`neuvad_lens`、`vad_code` 或其他项目代码；`VadCLIP/` 仅以未修改的 baseline 形式加载。

## 输入约定

- source CSV 必须有 `path,label` 两列，`path` 指向原始 512D VadCLIP 特征。XD 训练 CSV 中同键的 `__0` … `__9` 是同一视频的 VadCLIP 训练增强；CTNC 将它们聚成一条 hidden trajectory，并仅以排序后的第一条 feature 做时间长度对齐，不会把十条增强错误拼接成长视频。
- hidden manifest 必须有 `key,hidden_path` 两列；hidden 文件是已有的 `hidden=[T,12,768]` CLS artifact。
- source feature 的视频键和 manifest 的 `key` 按文件名去掉末尾 `__数字` 对齐。XD 同时接受 `A` 和官方分块列表中的 `A-0-0` normal 标签。发现和训练阶段默认取二者交集，缺失的训练 hidden 会写入 `discovery/missing_hidden.csv` 或 `training/missing_train_hidden.csv`；测试阶段保持严格，以免评测样本不完整。需要把缺失视为错误时，分别加 `--strict-hidden-manifest`、`--strict-train-hidden-manifest`。
- 正式实验前应确认 feature 与 hidden 的时间长度一致；现有产物若存在已知截断，可使用下方默认的 `--alignment crop_hidden`。

## XD-Violence

### 1. 发现 normality circuit

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
  --device cuda
```

输出：`../vadclipDNA_data/xd_ctnc_vad/discovery/circuit_assets.pt`、`channel_scores.csv`、`summary.json`。

### 2. 正式训练

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
  --lr 1e-5 \
  --scheduler-milestones 3 6 10 \
  --scheduler-rate 0.1 \
  --num-workers 0 \
  --normal-frame-weight 0.25 \
  --sparsity-weight 0.001 \
  --gate-initial-logit 0.0 \
  --alignment crop_hidden \
  --rank-anchor-fraction 0.125 \
  --rank-margin 0.10 \
  --rank-strength 0.25 \
  --rank-steps 3 \
  --seed 234 \
  --device cuda
```

输出：`../vadclipDNA_data/xd_ctnc_vad/training/model_best.pth`、`checkpoint_last.pth`、`history.csv`。

### 3. 正式测试

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
  --rank-anchor-fraction 0.125 \
  --rank-margin 0.10 \
  --rank-strength 0.25 \
  --rank-steps 3 \
  --device cuda
```

输出：`../vadclipDNA_data/xd_ctnc_vad/evaluation/metrics.json` 和可续跑的 `evaluation/predictions/*.npz`。

### 4. 导出解释证据

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
  --device cuda
```

输出：`../vadclipDNA_data/xd_ctnc_vad/audit/test/`。每个视频 artifact 包含 normal context、state/transition/text evidence、gate、每帧 top hidden dimensions 和对应贡献。

## UCF-Crime

命令完全相同，只替换数据集、输入 artifact、VadCLIP UCF checkpoint、评测 GT 和原训练超参数：

```text
--dataset ucf
--source-train-csv ../vad_data/work_ucf/ucf_train_local.csv
--source-test-csv ../vad_data/work_ucf/ucf_test_local.csv
--train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv
--test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv
--assets ../vadclipDNA_data/ucf_ctnc_vad/discovery/circuit_assets.pt
--init-baseline-model ../vadclip_data/model/vadclip_ucf.pth
--output-root ../vadclipDNA_data/ucf_ctnc_vad
--gt-path VadCLIP/list/gt_ucf.npy
--gt-segment-path VadCLIP/list/gt_segment_ucf.npy
--gt-label-path VadCLIP/list/gt_label_ucf.npy
--batch-size 64
--lr 2e-5
--scheduler-milestones 4 8
```

先用 `ctnc_vad.discover` 创建 UCF 的 `circuit_assets.pt`，再运行 `train`、`test`、`audit`。除上述路径和 VadCLIP 原始 UCF 默认超参数外，不需要写 UCF 专用代码或规则。

## 复用、中断与清理

- `discover`：已有 `discovery/circuit_assets.pt` 时自动复用；加 `--no-resume` 重算，或加 `--clean` 清理 discovery 阶段。
- `train`：每个 epoch 保存 `training/checkpoint_last.pth`；中断后使用原命令加 `--resume`。
- `test`：默认复用 `evaluation/predictions/*.npz`；加 `--no-resume` 重算预测，加 `--clean` 仅清理 evaluation。
- `audit`：默认复用每视频证据；加 `--no-resume` 或 `--clean` 重做当前 audit split。

`--clean` 只删除当前 CTNC 阶段，绝不会删除原始 VadCLIP、hidden manifest、512D 特征或其他项目的产物。
