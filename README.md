# Adaptive Clinical Residual Modulation for Multimodal Skin Lesion Classification

**Author:** Sahar Hassan Alzahrani  
**Institution:** Adelaide University  
**Project:** Computer Vision and Multimodal Machine Learning capstone

This project evaluates **Adaptive Clinical Residual Modulation (ACRM)**, a method that uses clinical metadata to adjust visual features for six-class skin lesion classification. The study compares ACRM with four benchmarks and two ablations under five-fold patient-grouped cross-validation.

## Project folder

The submission includes the source code, fixed fold assignments, notebooks, and saved results. Use [the experiment runner](scripts/run_model.py) to train and evaluate models, and [the main notebook](notebooks/PAD_UFES_Capstone_Experiments.ipynb) to regenerate the analysis.

```text
PAD_UFES_Capstone/
├── .gitignore                                  # Excludes downloaded data and generated Python files.
├── README.md                                   # This file.
├── pyproject.toml                              # Defines the Python package and its declared dependencies.
├── data/
│   └── images/                                 # Holds the 2,298 PNG images extracted during dataset setup.
├── metadata/
│   └── pad_ufes_20_with_folds.csv               # Stores clinical metadata, diagnosis labels, and fixed patient-grouped folds.
├── notebooks/
│   ├── 01_dataset_audit.ipynb                   # Examines image characteristics, class balance, missingness, and potential leakage.
│   ├── 02_metadata_preprocessing.ipynb          # Develops and verifies clinical preprocessing fitted on training data within each fold.
│   ├── 03_image_baseline.ipynb                  # Documents the original image-only EfficientNet-B0 experiments.
│   └── PAD_UFES_Capstone_Experiments.ipynb       # Documents the full model comparison, statistical tests, and computational benchmarks.
├── results/
│   ├── acrm/
│   ├── acrm_no_gate/
│   ├── acrm_no_residual_no_gate/
│   ├── concat/
│   ├── image_only/
│   ├── metablock_se/
│   ├── mmfnet/
│   └── statistical_analysis/
│       ├── accuracy_efficiency_comparison.csv  # Reports predictive performance and inference cost by model.
│       ├── accuracy_efficiency_tradeoff.png    # Plots macro-F1 against measured inference cost.
│       ├── benchmark_environment.csv           # Records the hardware, software, and settings used for inference benchmarking.
│       ├── inference_benchmark.csv              # Reports forward-pass latency and peak GPU memory by model.
│       ├── model_comparison.csv                 # Summarises predictive metrics across the five held-out folds.
│       ├── model_parameter_comparison.csv       # Compares registered, trainable, and forward-used parameter counts.
│       ├── model_performance_comparison.png     # Visualises performance across the benchmarks, ACRM, and its ablations.
│       └── paired_macro_f1_comparison.csv       # Reports paired fold-level differences, confidence intervals, and Wilcoxon p-values.
├── scripts/
│   └── run_model.py                             # Runs a selected model for one or five folds and exports its results.
├── src/
│   └── pad_multimodal/
│       ├── __init__.py                          # Initialises the experiment package.
│       ├── config.py                            # Defines class labels, input features, fold settings, and hyperparameters.
│       ├── data.py                              # Indexes images and builds datasets, transforms, and data loaders.
│       ├── evaluate.py                          # Computes classification metrics from held-out predictions.
│       ├── experiment.py                        # Coordinates grouped splitting, inner validation, retraining, and held-out evaluation.
│       ├── models/
│       │   ├── __init__.py                      # Initialises the model subpackage.
│       │   ├── acrm.py                          # Implements the gated residual fusion block and its ablation options.
│       │   ├── concat.py                        # Implements feature concatenation for image and metadata embeddings.
│       │   ├── factory.py                       # Constructs the model or ablation selected by the experiment runner.
│       │   ├── image_only.py                    # Implements the image-only EfficientNet-B0 classifier.
│       │   ├── metablock_se.py                  # Implements visual feature modulation using clinical sentence embeddings.
│       │   └── mmfnet.py                        # Implements bidirectional attention between image and metadata representations.
│       ├── preprocessing.py                     # Fits tabular transformations on training data and applies them to evaluation data.
│       ├── reproducibility.py                   # Sets experiment and per-fold random seeds.
│       ├── results.py                           # Exports predictions and metrics and generates evaluation plots.
│       ├── sentence_metadata.py                 # Converts clinical metadata into sentence embeddings for MetaBlock-SE.
│       └── train.py                             # Implements class weighting, single-epoch training, and prediction collection.
└── tests/
    ├── test_data.py                             # Tests data loading and dataset utilities.
    ├── test_evaluate.py                         # Tests classification metric calculations.
    ├── test_experiment.py                       # Tests fold splitting and experiment utilities.
    ├── test_factory.py                          # Tests model construction through the factory.
    ├── test_models.py                           # Tests the model implementations.
    ├── test_preprocessing.py                    # Tests clinical metadata preprocessing.
    ├── test_reproducibility.py                  # Tests random seed utilities.
    ├── test_results.py                          # Tests results export and figure generation.
    └── test_train.py                            # Tests model training and prediction utilities.
```

### Shared model outputs

Each model directory under `results/` uses the following output schema.

| File | Contents |
|---|---|
| `confusion_matrix.png` | Shows the row-normalised confusion matrix from pooled held-out predictions. |
| `fold_metrics.csv` | Records the selected training duration and held-out metrics for each outer fold. |
| `per_class_recall.png` | Shows recall for each diagnosis across pooled held-out predictions. |
| `predictions.csv` | Stores the true label, predicted label, and class probabilities for each held-out image. |
| `summary_metrics.csv` | Reports the mean and standard deviation of selected metrics across the five outer folds. |

## Dataset

PAD-UFES-20 contains 2,298 smartphone images of 1,641 lesions from 1,373 patients. The six classes are actinic keratosis (ACK), basal cell carcinoma (BCC), melanoma (MEL), melanocytic nevus (NEV), squamous cell carcinoma (SCC), and seborrheic keratosis (SEK). The dataset is imbalanced, with 845 BCC images and 52 MEL images. Macro-F1 is the primary metric because it gives equal weight to each diagnosis.

The multimodal experiments use eight clinical variables: age, anatomical region, itching, growth, pain, change, bleeding, and elevation. The project excludes the biopsy indicator and a block of 13 fields whose joint missingness was associated with particular diagnoses. Patient identifiers define the evaluation groups. Identifiers and diagnostic labels are excluded from the predictor inputs.

## Method and benchmarks

All models use ImageNet-pretrained EfficientNet-B0 to extract a pooled, 1,280-dimensional image representation. ACRM encodes the clinical variables with a two-layer multilayer perceptron. Three linear heads produce feature-wise scale, shift, and gate values. If `v` is the visual representation, the fused representation is:

```text
v_fused = v + sigmoid(gate_logits) ⊙ (scale ⊙ v + shift)
```

Here, `⊙` denotes element-wise multiplication. Zero-initialising the scale and shift heads makes the initial fusion output equal to the visual representation. The [ACRM implementation](src/pad_multimodal/models/acrm.py) defines the fusion algorithm directly using PyTorch layers and uses Torchvision for the pretrained backbone.

| Runner argument | Role | Comparison |
|---|---|---|
| `image_only` | Benchmark | Uses images alone to establish the visual baseline. |
| `concat` | Benchmark | Concatenates image and tabular embeddings to measure the benefit of simple fusion. |
| `metablock_se` | Benchmark | Uses clinical sentence embeddings to modulate visual features. |
| `mmfnet` | Benchmark | Uses bidirectional attention over compact image and metadata representations. |
| `acrm` | Proposed method | Applies feature-wise gated residual modulation. |
| `acrm_no_gate` | Ablation | Retains the residual pathway and removes adaptive gating. |
| `acrm_no_residual_no_gate` | Ablation | Retains scale and shift modulation while removing the gate and residual pathway. |

### Evaluation protocol

1. Use the included fixed outer folds. All images from a patient belong to the same fold.
2. For each outer fold, reserve that fold for held-out evaluation and draw one stratified patient-grouped validation split from the other four folds.
3. Fit tabular preprocessing on the relevant training partition. Resize images to 224 × 224 pixels, apply augmentation during training, and use deterministic transforms for validation and evaluation.
4. Select the training duration by inner-validation macro-F1. Reinitialise the model, train on the full outer training partition for the selected duration using the recorded learning-rate schedule, and evaluate on the held-out fold.
5. Aggregate fold-level metrics and pool held-out predictions for confusion matrices and class-level recall. Model seeds are 42–46, derived from base seed 42 and the outer fold index.

## Reproduce the experiments

| Goal | Where to start |
|---|---|
| View saved results | Open the relevant model folder under `results/`. Cross-model tables, statistical tests, and inference benchmarks are in `results/statistical_analysis/`. |
| Regenerate an individual model’s results | Set up the environment and dataset in Steps 1–2, then retrain the selected model in Step 3. |
| Regenerate the cross-model analysis | Follow Step 4 using the included five-fold results or outputs from a fresh run. |

### 1. Set up the Python environment

Open a terminal in the submitted project folder that contains `pyproject.toml`. If the folder was delivered as an archive, extract it first. Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e . numpy pandas scikit-learn scipy pillow
python -m pip check
python scripts/run_model.py --help
```

For Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. For GPU training, install compatible CUDA builds of PyTorch and Torchvision before the editable install. The runner uses CUDA when available and otherwise uses the CPU. The first run may download the pretrained EfficientNet-B0 weights and the sentence encoder used by MetaBlock-SE.

### 2. Prepare and validate the dataset

For training, verify that `data/images/` contains the PAD-UFES-20 PNGs. If the images are absent, download version 1 of PAD-UFES-20 and extract all three archives (`imgs_part_1.zip`, `imgs_part_2.zip`, and `imgs_part_3.zip`) into that directory. Keep the original PNG filenames. Use the included `metadata/pad_ufes_20_with_folds.csv` to reproduce the fixed splits. [Dataset download](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

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

### 3. Regenerate model results

For a fresh run, move the selected model’s existing output folder outside `results/`. For example, back up `results/acrm/` before retraining ACRM. The runner skips folds with saved metrics.

From the submitted project root, run all five folds for one model:

```bash
python scripts/run_model.py --model acrm
```

Replace `acrm` with any runner argument listed under [Method and benchmarks](#method-and-benchmarks). To run just fold 0, use:

```bash
python scripts/run_model.py --model acrm --fold 0
```

The runner writes fold metrics and held-out predictions to `results/<model>/`. Once all five folds are available, it also generates the summary metrics, confusion matrix, and per-class recall plot.

To regenerate results for all seven configurations, back up their existing output folders first, then run:

```bash
for model in image_only concat metablock_se mmfnet acrm acrm_no_gate acrm_no_residual_no_gate; do
  python scripts/run_model.py --model "$model" || break
done
```

### 4. Regenerate the cross-model analysis

Open the [experiment notebook](notebooks/PAD_UFES_Capstone_Experiments.ipynb) in Google Colab. Upload and extract the submitted project folder to `/content/PAD_UFES_Capstone/` so that `pyproject.toml` and `results/` sit directly inside it.

Run the setup cell below, then start at **Section 14. Statistical Analysis and Model Comparison**. The comparison tables and paired statistical tests use the five-fold outputs and can run on CPU. Select a GPU runtime to rerun the GPU inference benchmark. Cross-model outputs are saved under [`results/statistical_analysis/`](results/statistical_analysis/).

```python
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/content/PAD_UFES_Capstone")
assert (REPO_DIR / "pyproject.toml").is_file(), "Extract the submitted folder first."
os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-e", ".",
        "torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy", "pillow",
    ],
    check=True,
)
sys.path.insert(0, str(REPO_DIR / "src"))
```

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

ACRM exceeded the strongest benchmark, adapted MMF-Net, by 4.06 percentage points in mean macro-F1. The two-sided Wilcoxon tests use five paired fold scores, and all reported p-values exceed 0.05. The class-level analysis shows uneven performance across diagnoses, including 75.00% pooled recall for melanoma and 29.69% for squamous cell carcinoma. [Model comparison](results/statistical_analysis/model_comparison.csv) · [Paired comparisons](results/statistical_analysis/paired_macro_f1_comparison.csv)

### Computational cost and limitations

Full ACRM contains approximately 4.53 million parameters. For batch size `B`, metadata dimension `d_m`, hidden dimension `h`, and visual dimension `d_v`, the fusion block takes approximately `O(B(d_m h + h² + 3 h d_v + d_v))` operations. Its parameters scale with `O(d_m h + h² + 3 h d_v)`, and its intermediate activations scale with `O(B(h + d_v))`, excluding the visual backbone and final classifier.

In the recorded FP32 forward-pass benchmark on a Tesla T4 at batch size 32, ACRM averaged **34.306 ms per batch** with **365.53 MiB peak GPU memory**. The image-only benchmark averaged **33.04 ms** and **363.55 MiB**. These are model forward-pass measurements. End-to-end latency also depends on image loading, preprocessing, and metadata preparation. [Inference benchmark](results/statistical_analysis/inference_benchmark.csv) · [Benchmark environment](results/statistical_analysis/benchmark_environment.csv) · [Parameter comparison](results/statistical_analysis/model_parameter_comparison.csv)

The evaluation uses a single dataset and a restricted set of clinical variables. ACRM operates on pooled visual features, so it cannot directly link a metadata field to a spatial lesion region. The ablations measure the gate's contribution with the residual pathway present and the residual pathway's contribution with the gate disabled. Exact training scores and timings can vary with hardware and library versions.

## Dataset citation

Pacheco, A. G. C., et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221. [Dataset paper](https://doi.org/10.1016/j.dib.2020.106221) · [Dataset record](https://doi.org/10.17632/zr7vgbcyr2.1)