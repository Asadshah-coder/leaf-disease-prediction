# Leaf Disease Prediction Using Deep Learning

Final Year Project — Plant Disease Recognition System, rebuilt to
research standard for scholarship applications.

- **Author:** Muhammad Asad Bukhari, BS IT, University of Education Lahore (DG Khan Campus), 2022–2026
- **Stack:** TensorFlow / Keras, scikit-learn, Streamlit, Python 3.12
- **Classes:** 38 (healthy + diseased, across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)

## Abstract

Plant diseases cause large crop losses, and early identification from a
leaf photograph can help farmers act before the disease spreads. This
project trains a convolutional neural network (CNN) to classify leaf
images into 38 healthy/diseased categories and serves the trained model
through a Streamlit web app. The original 10-epoch run recorded
**98.18% training accuracy and 96.06% validation accuracy** (see
Results). The model was **retrained on 7 Nov 2026** on Google Colab
(T4 GPU) with the reproducible script in this repo (data augmentation
on, 10 epochs, 70,295 train / 17,572 validation images): final
**95.20% training accuracy, 94.21% validation accuracy, 0.9417
macro-average F1** — the honest, reproducible numbers, saved as
`trained_plant_disease_model.keras` (86 MB). The improved version adds a reproducible training script,
proper per-class evaluation, Grad-CAM explainability, and a
transfer-learning comparison (custom CNN vs. MobileNetV2 and
EfficientNetB0) so the work can be judged as research, not only as a
demo. A short research summary is in `docs/research-summary.md`.

## Problem

A farmer who notices a sick leaf usually has no quick, cheap way to
know which disease it is. Laboratory diagnosis is slow and costly, and
misidentification leads to the wrong treatment. The goal of this project
is a tool that takes one leaf photo and returns the most likely disease
with a confidence score — fast enough to use in the field.

## Dataset

The images come from the public **PlantVillage** dataset (Hughes &
Salathé, 2015), the standard open dataset for plant-disease image
classification.

| Split | Images |
|---|---|
| Train | 70,295 |
| Validation | 17,572 |
| Test (sample images in the original zip) | 33 |
| **Total** | **87,900** |

The full dataset (~88,000 images, 1.7 GB) is **not** in this repo —
GitHub does not host files that large. Download the full project zip
(code + dataset) from Google Drive:

- **[Plant Disease FYP.zip (1.7 GB) — Google Drive](https://drive.google.com/file/d/13_IbE48q2Nv3whIDxVf8OV60JzZGVySx/view?usp=sharing)**

After unzipping, the dataset folders are `train/`, `valid/`, and
`test/`, organised by class folder (e.g. `Tomato___Early_blight`).

## Methodology

### Architecture

Custom CNN (Sequential), reproduced exactly from the original training
notebook — including its quirk that the second convolution in the
128-filter block uses 32 filters, kept unchanged so results stay
comparable with the original run:

| Block | Layers |
|---|---|
| 1 | Conv2D 32, Conv2D 32, MaxPool 2×2 |
| 2 | Conv2D 64, Conv2D 64, MaxPool 2×2 |
| 3 | Conv2D 128, Conv2D 32, MaxPool 2×2 |
| 4 | Conv2D 256, Conv2D 256, MaxPool 2×2 |
| 5 | Conv2D 512, Conv2D 512, MaxPool 2×2 |
| Head | Dropout 0.25, Flatten, Dense 1500 (ReLU), Dropout 0.4, Dense 38 (softmax) |

### Training setup

- Input: 128×128 RGB images, batch size 32
- Optimiser: Adam, learning rate 0.0001
- Loss: categorical cross-entropy; metric: accuracy
- Epochs: 10 (original run)
- Augmentation (new in this version; the original had none): random horizontal flip, rotation (±10%), zoom (±10%) on the training set only
- Seed: 42 (see `src/train.py`)

### Evaluation and explainability

- `src/evaluate.py` — accuracy + per-class precision/recall/F1 and a confusion matrix on any labelled folder.
- `src/gradcam.py` — Grad-CAM heatmaps showing which part of the leaf drove a prediction.
- `src/compare_models.py` — same data pipeline, three models: custom CNN, MobileNetV2 (ImageNet), EfficientNetB0 (ImageNet); results in `results/model_comparison.csv`.

## Results

### Retraining run #2 — 7 Nov 2026 (Google Colab, T4 GPU)

The model was retrained a second time the same day (the first
retrain's model file was lost when its Colab runtime disconnected
before the file was saved). Run #1 reached 94.86% training / 93.08%
validation accuracy; its files are preserved as
`results/*_retrain1.*`. Run #2 used the same reproducible script
(`python src/train.py --data-dir data --epochs 10 --batch-size 32`),
with data augmentation (random flip / rotation / zoom) on the training
set only. Full per-epoch history in `results/training_history.json`,
per-class metrics in `results/evaluation.json` and
`results/classification_report.txt` (17,572 validation images),
confusion matrix in `results/confusion_matrix.png`.

| Metric | Epoch 1 | Final (epoch 10) |
|---|---|---|
| Training accuracy | 40.11% | **95.20%** |
| Validation accuracy | 65.47% | **94.21%** |
| Training loss | 2.049 | 0.146 |
| Validation loss | 1.109 | 0.167 |

Validation accuracy peaked at **94.32% (epoch 9)**; macro-average F1
**0.9417**, weighted-average F1 **0.9416**. The trained model
(`trained_plant_disease_model.keras`, 86 MB) is kept off GitHub (see
`.gitignore`) — it is hosted on Google Drive (public, anyone with the
link) in the `FYP-Leaf-Disease` folder, and `main.py` downloads it
automatically on first run if no local model file exists.

![Training vs. validation accuracy](results/figures/accuracy.png)
![Training vs. validation loss](results/figures/loss.png)

### Original run (for comparison)

These are the **verified numbers from the original run**
(`results/training_history_original.json`, 10 epochs, no augmentation).
No other accuracy figure is claimed anywhere in this project.

| Metric | Epoch 1 | Final (epoch 10) |
|---|---|---|
| Training accuracy | 61.50% | **98.18%** |
| Validation accuracy | 83.68% | **96.06%** |
| Training loss | 1.293 | 0.057 |
| Validation loss | 0.523 | 0.138 |

The retrained run scores lower than the original run, which is
expected: augmentation makes training harder (the model sees a
different augmented image every epoch), so training accuracy rises
more slowly — but the retrained numbers are the reproducible ones,
measured end-to-end by `src/train.py` on the same validation set,
with the trained model file actually saved. Do not quote test-set
numbers: the original zip's `test/` folder holds only 33 sample
images, not a proper held-out test set.

## Reproducibility

Full training needs a GPU. The steps are in
[`docs/COLAB_TRAINING.md`](docs/COLAB_TRAINING.md) (Google Colab, free
GPU). In short:

```bash
git clone https://github.com/Asadshah-coder/leaf-disease-prediction.git
cd leaf-disease-prediction
# put the dataset in data/ (train/, valid/, test/)
pip install -r requirements.txt
python src/train.py --data-dir data --epochs 10
streamlit run main.py
```

Plot the training curves from an existing history file:

```bash
python src/plot_results.py
```

Evaluate a trained model and explain one prediction:

```bash
python src/evaluate.py --model trained_plant_disease_model.keras --data-dir data/valid
python src/gradcam.py --model trained_plant_disease_model.keras --image path/to/leaf.jpg --output results/gradcam.png
```

## Live demo

**https://leaf-disease-prediction-app.streamlit.app/** (Streamlit
Community Cloud, free tier). The app downloads the trained model from
Google Drive on first run (see the model note under Results).

The app has five sections (sidebar navigation):

- **Diagnose** — upload up to 3 leaf photos (or take one with the
  camera), add the farm/city location (manual or browser detection),
  and get the top-3 predictions with confidence bars, disease
  information and local weather (Open-Meteo).
- **Dashboard** — session statistics, diagnoses by disease and by day,
  and the model's verified performance.
- **History & Calendar** — every diagnosis saved with date, time and
  location; browse past diagnoses on a month calendar, read the
  seasonal disease guide, and export history as CSV.
- **Disease Library** — all 38 classes with symptoms and general
  management advice (searchable).
- **About** — the project, the model and its verified results.

## How to run the app

```bash
pip install -r requirements.txt
streamlit run main.py
```

Then upload up to 3 leaf images on the Diagnose page and press
Diagnose. The top 3 predictions are shown with confidence bars; a
prediction below **80%** confidence is marked uncertain (kept from the
original project's 80% rule, which used to hide such rows entirely).
If no model file is present, the app explains how to get one instead
of crashing. History is kept for the session only — a database is
needed for permanent history.

## Known limitations

1. **Trained model file missing in the original zip.** `main.py` and
   the Test notebook load `trained_plant_disease_model.keras`, but that
   file was not in the original project zip, and the original training
   notebook saved the model as `trained_model.keras` (name mismatch).
   Fixed in this version: `src/train.py` saves with the correct name,
   and `main.py` loads the model once (cached) and falls back to
   `trained_model.keras` if the preferred name is absent. The model
   was retrained on 7 Nov 2026 (see Results) — place
   `trained_plant_disease_model.keras` next to `main.py` to run
   predictions; until then the app explains how to get one instead of
   crashing. The 86 MB model file is not stored in this repo
   (`.gitignore`); the app downloads it from Google Drive on first
   run, or you can retrain with the Colab notebook to reproduce it.
2. **Test set is only 33 images.** The original zip's `test/` folder
   is a small sample set, not a proper held-out test set. Test-set
   numbers must come from a proper split in the retraining run; the
   validation set was also (incorrectly) used as the test set in the
   old Test notebook.
3. **No data augmentation in the original run.** Added in `src/train.py`.
4. **Per-class metrics recorded on retraining.** `src/evaluate.py`
   produces them; `results/evaluation.json` and
   `results/classification_report.txt` now hold the 7 Nov 2026
   retraining run (macro-average F1 0.9417, 17,572 validation images).
   Note: two dataset folder names differ slightly from the names in
   `src/model.py` (`Tomato___Spider_mites Two-spotted_spider_mite`
   uses a space; one Corn folder name also differs) — `src/train.py`
   is unaffected (TensorFlow sorts folders, indices align) and
   `src/evaluate.py` now tolerates such spelling variants.
5. **Single train/validation split, no cross-validation.** Noted as future work.
6. **Grad-CAM fixed 7 Nov 2026.** It failed twice on Keras 3
   (`AttributeError: ... has never been called`, then a `KeyError`
   in `Functional.call()`), and now works via a pure eager
   layer-by-layer forward pass inside `GradientTape`
   (`results/gradcam.png`, verified on the retrained model).

## Future work

- Fix Grad-CAM for Keras 3 models loaded from disk and publish `results/gradcam.png`.
- Publish the verified test-set classification report.
- Complete the MobileNetV2 / EfficientNetB0 comparison and report it in `results/model_comparison.csv`.
- Add cross-validation and a proper held-out test split.
- Try a larger input resolution (e.g. 224×224) for the transfer-learning models.
- Deploy the app (Streamlit Cloud / Hugging Face Spaces) with the trained model.
- Test on real field photographs (different lighting/backgrounds) to check robustness outside the PlantVillage distribution.

## Project structure

```
.
├── main.py                     # Streamlit app (model loaded once, cached)
├── home_page.png.png           # Banner image used by the app
├── index.html, disease_recognition.html   # Static HTML pages (original)
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py                # Custom CNN definition + class names
│   ├── train.py                # Reproducible training + evaluation
│   ├── evaluate.py             # Standalone model evaluation
│   ├── gradcam.py              # Grad-CAM explainability
│   ├── compare_models.py       # Custom CNN vs. MobileNetV2 vs. EfficientNetB0
│   └── plot_results.py         # Accuracy/loss figures
├── notebooks/
│   ├── Train_plant_disease.ipynb   # Original training notebook (unchanged)
│   └── Test_Plant_Disease.ipynb    # Original test notebook (unchanged)
├── results/
│   ├── training_history.json   # Verified 10-epoch history (original run)
│   └── figures/                # accuracy.png, loss.png (from plot_results.py)
└── docs/
    ├── COLAB_TRAINING.md       # Step-by-step Colab retraining guide
    └── research-summary.md     # Short research summary for applications
```
