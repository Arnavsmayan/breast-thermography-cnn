# Breast Thermography Image Classification with CNNs

End-to-end deep-learning pipeline that classifies thermal (infrared) breast images as **Healthy** vs **Sick** using transfer learning on a pretrained CNN backbone. Built on the public DMR (Database for Mastology Research) dataset from Visual Lab, UFF, Brazil.

The project goes beyond a vanilla classifier: it handles a small, imbalanced medical dataset with **patient-grouped data loading**, **stratified splits**, **class-weighted training**, **early stopping**, and a full evaluation suite (AUROC, F1, sensitivity, specificity, ROC, PR, and lift curves).

---

## Why this project

Thermography is a non-invasive, radiation-free imaging modality that can surface vascular and metabolic anomalies associated with breast pathology. The challenge is that public thermal datasets are **small, noisy, and imbalanced** — the kind of real-world constraints that production ML has to deal with. This repo is a clean baseline that demonstrates how to:

- Build a reliable image-classification pipeline on a small medical dataset
- Avoid the most common leakage trap: mixing patients across train/test
- Handle class imbalance properly with weights instead of resampling tricks
- Report results that a reviewer would actually trust (per-class metrics, not just accuracy)

---

## Results

Single training run, VGG16 backbone, NVIDIA Tesla T4:

| Metric                      | Train  | Test   |
| --------------------------- | ------ | ------ |
| Accuracy                    | 0.893  | 0.763  |
| AUROC                       | 0.970  | 0.876  |
| F1                          | —      | 0.825  |
| Precision                   | —      | 0.786  |
| Sensitivity (recall)        | —      | 0.868  |
| Specificity                 | —      | 0.571  |

**Read with care:** in the current code, the positive class is *Healthy*, so "sensitivity" is the model's ability to correctly identify healthy patients. For a clinical screening setup you'd flip the labels so the positive class is *Sick* and report sensitivity for disease detection — easy to do, just relabel and rerun. AUROC (0.88) is invariant to that choice and is the headline number.

---

## Dataset

- **Source:** [DMR — Database for Mastology Research](http://visual.ic.uff.br/dmi/), Visual Lab, UFF, Niterói, Brazil
- **Classes:** Healthy (label `1`) vs Sick (label `0`)
- **Modality:** Thermal IR images, both *static* and *dynamic* protocols
- **After filtering:** 389 images across 238 patient folders (~5 images per usable patient)
- **Imbalance:** ~65% Healthy / 35% Sick (class weights ≈ `{0: 1.43, 1: 0.77}`)

The dataset is **not redistributed** in this repo — request access from the DMR project page.

### Loading strategy

For each patient, the loader prefers the static thermography protocol; if static images are unavailable, it falls back to dynamic frames, deduplicated by frame prefix so that we don't oversample a single patient's video. A short list of patients is excluded for data-quality reasons. Patient `p247` has a hand-tuned exception because of a non-standard filename layout.

### Preprocessing

- Resize to 640×480, then center-pad to 640×640 to preserve aspect ratio
- Convert to RGB so ImageNet-pretrained backbones can ingest the tensors
- No augmentation in the baseline (kept off intentionally to make ablations interpretable)

---

## Modeling

**Backbone:** VGG16, ImageNet weights, frozen.

```
VGG16 (frozen)
  → GlobalAveragePooling2D
  → Dense(128, relu, L2=1e-3)
  → Dropout(0.5)
  → Dense(1, sigmoid, L2=1e-3)
```

**Training**

- Loss: binary cross-entropy
- Optimizer: Adam (default LR)
- Batch size: 32
- Max epochs: 200, with `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)`
- Class weights computed with `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`
- Splits: 70 / 15 / 15 train / val / test, stratified by label, fixed seed (`random_state=42`)

The notebook also imports MobileNetV2 — wired for an easy backbone swap if you want a faster baseline.

---

## Evaluation

`evaluate_model` produces, for every model trained:

- Accuracy and AUROC on train and test
- F1, precision, sensitivity (recall), specificity
- Confusion matrix (test)
- Training/validation **loss** and **accuracy** curves
- ROC curve vs. no-skill baseline
- Precision–recall curve
- Lift / cumulative-gain chart

Per-image predictions are dumped to `vgg16_predictions.csv` and the full training history to `vgg16_training_history.csv` for offline analysis.

---

## Repo structure

```
.
├── thermal-image-classification-pipeline-using-cnns.ipynb   # full pipeline
├── LICENSE
└── README.md
```

A single self-contained notebook on purpose — easy to read top to bottom, easy to fork on Kaggle/Colab.

---

## How to reproduce

### Option A — Kaggle (path of least resistance)

The notebook expects the DMR dataset mounted at:
```
/kaggle/input/dmr-data/DMR - Database For Mastology Research - Visual Lab, UFF, Niterói, Brazil/
    Healthy/
    Sick/
```
Add the dataset to a Kaggle notebook, attach a GPU (T4 is enough), run all cells.

### Option B — Local

```bash
git clone https://github.com/Arnavsmayan/breast-thermography-cnn.git
cd breast-thermography-cnn
python -m venv .venv && source .venv/bin/activate
pip install tensorflow scikit-learn numpy pillow tqdm pandas matplotlib seaborn jupyter
jupyter lab
```

Then update `healthy_base_directory` and `sick_base_directory` in cell 1 to point to your local copy of DMR.

> A GPU is strongly recommended — VGG16 at 640×640 is heavy. Drop input size to 224×224 (and adjust the model's `input_shape`) for fast CPU iteration.

---

## Honest limitations

- **N is small.** 389 images / 238 patients is not enough to declare clinical victory. Treat the AUROC as a proof-of-concept signal, not a deployable threshold.
- **Single train/val/test split.** Cross-validation (patient-grouped K-fold) would give tighter confidence intervals — high-priority follow-up.
- **No augmentation in the baseline.** Adds a ceiling that data augmentation would lift.
- **Frozen backbone.** Fine-tuning the top VGG16 blocks at a low learning rate is the obvious next experiment.
- **Class-positive labeling is inverted** for clinical reporting (positive = healthy in the current code). Flip before quoting sensitivity for screening.

---

## Roadmap

- Patient-grouped K-fold cross-validation
- Backbone bake-off: VGG16 vs MobileNetV2 vs EfficientNet-B0/B3 vs ConvNeXt-Tiny
- Two-stage fine-tuning (head only → unfreeze top blocks at LR/10)
- SHAP / Grad-CAM explainability (scaffolding already present, commented out at the bottom of the notebook)
- Augmentation suite tuned for thermal: mild rotations, horizontal flip, intensity jitter
- Threshold tuning against a clinically meaningful operating point (high sensitivity for the *Sick* class)

---

## Tech stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `Pillow` · `NumPy` · `pandas` · `Matplotlib` · `Seaborn` · `tqdm`

---

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Arnav** — building data-science and ML projects across vision, NLP, and applied analytics.
GitHub: [@Arnavsmayan](https://github.com/Arnavsmayan)
