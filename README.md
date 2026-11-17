# Adaptive Clinical Residual Modulation for Multimodal Skin Lesion Classification

**Author:** Sahar Hassan Alzahrani  
**Institution:** Adelaide University  
**Project:** Computer Vision and Multimodal Machine Learning capstone

This project investigates whether clinical metadata can improve six-class skin lesion classification by adjusting the visual features extracted from smartphone photographs. The proposed method, **Adaptive Clinical Residual Modulation (ACRM)**, learns a metadata-conditioned adjustment and adds it to the original visual representation through a gated residual connection. The experiments compare ACRM with four benchmarks and two ablations under the same five-fold patient-grouped evaluation protocol.

## Dataset

PAD-UFES-20 contains 2,298 smartphone images of 1,641 lesions from 1,373 patients. The six classes are actinic keratosis (ACK), basal cell carcinoma (BCC), melanoma (MEL), melanocytic nevus (NEV), squamous cell carcinoma (SCC), and seborrheic keratosis (SEK). The dataset is imbalanced, with 845 BCC images and 52 MEL images. Macro-F1 is the primary metric because it gives equal weight to each diagnosis. [PAD-UFES-20 dataset, version 1](https://doi.org/10.17632/zr7vgbcyr2.1)

The multimodal experiments use eight clinical variables: age, anatomical region, itching, growth, pain, change, bleeding, and elevation. The project excludes the biopsy indicator and a block of 13 fields whose joint missingness was associated with particular diagnoses. Patient identifiers define the evaluation groups. Identifiers and diagnostic labels are excluded from the predictor inputs.

## Method and benchmarks

All models use ImageNet-pretrained EfficientNet-B0 to extract a pooled, 1,280-dimensional image representation. ACRM encodes the clinical variables with a two-layer multilayer perceptron. Three linear heads produce feature-wise scale, shift, and gate values. If `v` is the visual representation, the fused representation is:

```text
v_fused = v + sigmoid(gate_logits) ⊙ (scale ⊙ v + shift)
```

Here, `⊙` denotes element-wise multiplication. Zero-initialising the scale and shift heads makes the initial fusion output equal to the visual representation. The [ACRM implementation](src/pad_multimodal/models/acrm.py) defines the fusion algorithm directly using PyTorch layers and uses Torchvision for the pretrained backbone.

| Runner argument | Role | Comparison |
|---|---|---|
| `image_only` | Benchmark | EfficientNet-B0 with image input only. Establishes the visual baseline. |
| `concat` | Benchmark | Concatenates image and tabular embeddings. Measures the benefit of simple multimodal fusion. |
| `metablock_se` | Benchmark | Adapts MetaBlock-SE using sentence embeddings and metadata-guided modulation. Provides a published approach to conditioning visual features. |
| `mmfnet` | Benchmark | Adapts MMF-Net using bidirectional attention over compact modality representations. Provides an attention-based fusion comparison. |
| `acrm` | Proposed method | Applies feature-wise gated residual modulation. |
| `acrm_no_gate` | Ablation | Retains the residual pathway and removes adaptive gating. |
| `acrm_no_residual_no_gate` | Ablation | Retains scale and shift modulation while removing the gate and residual pathway. |

### Evaluation protocol

1. Use the repository's fixed outer folds. All images from a patient belong to the same fold.
2. For each outer fold, reserve that fold for held-out evaluation and draw one stratified patient-grouped validation split from the other four folds.
3. Fit tabular preprocessing on the relevant training partition. Resize images to 224 × 224 pixels, apply augmentation during training, and use deterministic transforms for validation and evaluation.
4. Select the training duration by inner-validation macro-F1. Reinitialise the model, train on the full outer training partition for the selected duration using the recorded learning-rate schedule, and evaluate on the held-out fold.
5. Aggregate fold-level metrics and pool held-out predictions for confusion matrices and class-level recall. Model seeds are 42–46, derived from base seed 42 and the outer fold index.

## Reproduce the experiments

### 1. Set up the source and environment

Use the extracted submission folder or clone the repository with a GitHub account that has access. Run the commands below from the repository root. Python 3.10 or later is required. The runner uses CUDA when available and otherwise uses CPU. A CUDA-capable GPU is practical for the full experiment matrix. The first run may download pretrained EfficientNet-B0 weights and, for MetaBlock-SE, the sentence encoder.

```bash
git clone https://github.com/SaharHassanAAlzahrani/PAD_UFES_Capstone.git
cd PAD_UFES_Capstone

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e . numpy pandas scikit-learn scipy pillow
python -m pip check
python scripts/run_model.py --help
```

The activation command above applies to macOS and Linux. On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`. On a CUDA system, install a matching PyTorch and Torchvision build for that system before the editable install. The install command includes runtime libraries used by the scripts and notebook in addition to the dependencies declared in `pyproject.toml`. The [recorded environment](results/statistical_analysis/benchmark_environment.csv) describes the Tesla T4 inference benchmark.

### 2. Prepare and validate the dataset

Download PAD-UFES-20 version 1 and extract all three image archives, `imgs_part_1.zip`, `imgs_part_2.zip`, and `imgs_part_3.zip`, into `data/images/`. Nested image folders are supported. Keep the original PNG filenames. Use the repository's `metadata/pad_ufes_20_with_folds.csv` to preserve the published experiment splits. [Dataset download](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

Expected layout:

```text
PAD_UFES_Capstone/
├── data/
│   └── images/                         # 2,298 extracted PNGs
├── metadata/
│   └── pad_ufes_20_with_folds.csv       # fixed patient-grouped folds
├── notebooks/
├── results/
├── scripts/
│   └── run_model.py
├── src/
│   └── pad_multimodal/
└── pyproject.toml
```

Check the image filenames and fold assignments before training:

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

metadata = pd.read_csv("metadata/pad_ufes_20_with_folds.csv")
images = list(Path("data/images").rglob("*.png"))
image_names = {path.name for path in images}

assert len(metadata) == 2298 and metadata["img_id"].is_unique
assert len(image_names) == len(images), "Duplicate PNG filenames"
missing = set(metadata["img_id"]) - image_names
assert not missing, f"{len(missing)} images missing: {sorted(missing)[:5]}"
assert set(metadata["fold"]) == set(range(5))
assert metadata.groupby("patient_id")["fold"].nunique().eq(1).all()
print(pd.crosstab(metadata["fold"], metadata["diagnostic"]))
PY
```

### 3. Run one fold or the full comparison

The runner skips a fold when its metrics are already saved under `results/<model>/`. To verify fresh training, first move the existing results directory for that model to a backup location. For example, if `results/acrm/` exists, rename or move that directory before running:

```bash
python scripts/run_model.py --model acrm --fold 0
```

Omit `--fold` to run all five folds for a model. To retrain the full comparison, back up the existing per-model output directories first, then run:

```bash
for model in image_only concat metablock_se mmfnet acrm acrm_no_gate acrm_no_residual_no_gate; do
  python scripts/run_model.py --model "$model" || break
done
```

### 4. Reproduce the tables and figures

- [Experiment notebook](notebooks/PAD_UFES_Capstone_Experiments.ipynb) documents model runs, comparison tables, paired fold-level tests, parameter counts, and inference benchmarking.
- To use the notebook in Google Colab as written, select a GPU runtime for inference benchmarking. The clone cell uses a Colab secret named `GITHUB_TOKEN` if the checkout is absent. The data setup cells expect the three archives under `MyDrive/PAD_UFES_Capstone/data/` and a copy of the fixed fold CSV under `MyDrive/PAD_UFES_Capstone/metadata/`. With a prepared local checkout, use the command-line steps above for training.
- Each model writes `fold_metrics.csv` and `predictions.csv` to `results/<model>/`. When all five folds are available, the runner also writes `summary_metrics.csv`, `confusion_matrix.png`, and `per_class_recall.png`.
- [Statistical and computational outputs](results/statistical_analysis/) contain the saved comparison, paired macro-F1 tests, model parameter counts, inference measurements, and benchmark environment. The notebook reads the saved fold metrics for the comparison and statistical cells.

## Results

Values are percentages reported as mean ± sample standard deviation across five held-out folds.

| Model | Macro-F1 | Balanced accuracy | Macro AUROC |
|---|---:|---:|---:|
| Image-only EfficientNet-B0 | 60.42 ± 3.32 | 62.28 ± 3.18 | 89.25 ± 2.11 |
| Feature concatenation | 63.96 ± 3.90 | 65.05 ± 2.95 | 90.51 ± 1.75 |
| Adapted MetaBlock-SE | 60.88 ± 5.17 | 62.30 ± 5.20 | 89.78 ± 2.91 |
| Adapted MMF-Net | 65.64 ± 2.70 | 68.93 ± 4.16 | 92.86 ± 1.53 |
| **ACRM** | **69.70 ± 2.35** | **71.48 ± 2.50** | **93.14 ± 1.49** |
| ACRM without gate | 67.91 ± 3.49 | 69.94 ± 3.03 | 93.04 ± 1.56 |
| ACRM without gate or residual pathway | 67.22 ± 4.12 | 68.83 ± 2.65 | 93.07 ± 0.98 |

ACRM achieved the highest mean macro-F1 and balanced accuracy. Its mean macro-F1 exceeded adapted MMF-Net by 4.06 percentage points. The repository's two-sided Wilcoxon tests use five paired fold scores, and all reported p-values exceed 0.05. The class-level analysis shows uneven performance across diagnoses, including 75.00% pooled recall for melanoma and 29.69% for squamous cell carcinoma. [Model comparison](results/statistical_analysis/model_comparison.csv) · [Paired comparisons](results/statistical_analysis/paired_macro_f1_comparison.csv)

### Computational cost and limitations

Full ACRM contains approximately 4.53 million parameters. For batch size `B`, metadata dimension `d_m`, hidden dimension `h`, and visual dimension `d_v`, the fusion block takes approximately `O(B(d_m h + h² + 3 h d_v + d_v))` operations. Its parameters scale with `O(d_m h + h² + 3 h d_v)`, and its intermediate activations scale with `O(B(h + d_v))`, excluding the visual backbone and final classifier.

In the recorded FP32 forward-pass benchmark on a Tesla T4 at batch size 32, ACRM averaged **34.306 ms per batch** with **365.53 MiB peak GPU memory**. The image-only benchmark averaged **33.04 ms** and **363.55 MiB**. These are model forward-pass measurements. End-to-end latency also depends on image loading, preprocessing, and metadata preparation. [Inference benchmark](results/statistical_analysis/inference_benchmark.csv) · [Parameter comparison](results/statistical_analysis/model_parameter_comparison.csv)

The evaluation uses a single dataset and a restricted set of clinical variables. ACRM operates on pooled visual features, so it cannot directly link a metadata field to a spatial lesion region. The ablations measure the gate's contribution with the residual pathway present and the residual pathway's contribution with the gate disabled. Exact training scores and timings can vary with hardware and library versions.

## Repository guide

| Location | Contents |
|---|---|
| [`src/pad_multimodal/`](src/pad_multimodal/) | Image and metadata preprocessing, model definitions, fold evaluation, training, and metrics |
| [`scripts/run_model.py`](scripts/run_model.py) | Command-line entry point for one fold or a full five-fold run |
| [`metadata/`](metadata/) | Fixed patient-grouped fold assignments |
| [`notebooks/`](notebooks/) | Exploratory and experimental workflows |
| [`results/`](results/) | Held-out predictions, scores, figures, and computational analysis |

## Dataset citation

Pacheco, A. G. C., et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221. [Dataset paper](https://doi.org/10.1016/j.dib.2020.106221) · [Dataset record](https://doi.org/10.17632/zr7vgbcyr2.1)
