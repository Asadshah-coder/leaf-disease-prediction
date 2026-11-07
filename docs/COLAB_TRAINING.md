# Training on Google Colab (GPU) — step by step

Full training on ~70,000 images needs a GPU. This machine (the one
that built this improved version) has no GPU and no TensorFlow
installed, so training is done on Google Colab's free GPU instead.
Follow these steps exactly.

## 1. Get the code and the data

1. Open [Google Colab](https://colab.research.google.com/) and sign in
   with a Google account.
2. Create a new notebook (`File → New notebook`).
3. Enable a GPU: `Runtime → Change runtime type → Hardware accelerator
   → T4 GPU → Save`. Check it worked by running this cell:

   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices("GPU"))
   ```

   You should see at least one GPU listed. If the list is empty, the
   runtime did not switch — try `Runtime → Restart runtime` and check
   again.

## 2. Download the project and the dataset

Run this cell in Colab (replace the repo URL with the real one if the
improved code has been pushed to GitHub):

```bash
!git clone https://github.com/Asadshah-coder/leaf-disease-prediction.git
%cd leaf-disease-prediction
!pip install -q -r requirements.txt
```

Get the dataset (the full 1.7 GB project zip) from Google Drive:

- [Plant Disease FYP.zip (1.7 GB) — Google Drive](https://drive.google.com/file/d/13_IbE48q2Nv3whIDxVf8OV60JzZGVySx/view?usp=sharing)

The simplest way in Colab is to mount your own Google Drive and copy
the file there, or download it in the browser and upload it to Colab
(`Files` panel → upload). Then unzip it in Colab:

```bash
!unzip -q "Plant Disease FYP.zip"
!ls "Plant Disease FYP"
```

You need the folders `train/` and `valid/` (each with 38 class
subfolders). Put them in a `data/` folder:

```bash
!mkdir -p data
!mv "Plant Disease FYP/train" data/train
!mv "Plant Disease FYP/valid" data/valid
!ls data/train | head
```

Check the counts match the README (train 70,295 / valid 17,572 images).

## 3. Train the custom CNN

```bash
!python src/train.py --data-dir data --epochs 10 --batch-size 32 \
    --model-out trained_plant_disease_model.keras
```

What this does:

- trains the custom CNN (see `src/model.py`) with augmentation;
- saves the model as `trained_plant_disease_model.keras` — the exact
  name `main.py` looks for;
- writes `results/training_history.json`,
  `results/evaluation.json`, `results/classification_report.txt` and
  `results/confusion_matrix.png`.

Training 10 epochs on a T4 GPU takes roughly 1–2 hours for this
dataset. Keep the Colab tab open; if the runtime disconnects, training
stops and must be restarted (the free tier has session time limits, so
do not close the tab).

## 4. Evaluate and explain

```bash
!python src/evaluate.py --model trained_plant_disease_model.keras \
    --data-dir data/valid
!python src/plot_results.py
# Grad-CAM on one sample image:
!python src/gradcam.py --model trained_plant_disease_model.keras \
    --image "data/valid/Tomato___Early_blight/<a-sample-image>.jpg" \
    --output results/gradcam.png
```

(Replace `<a-sample-image>.jpg` with a real filename from that
folder — list it with `!ls "data/valid/Tomato___Early_blight" | head`.)

Copy the numbers from `results/evaluation.json` and
`results/classification_report.txt` into the README results table.
Only quote numbers you actually measured in this run.

## 5. (Optional) Run the transfer-learning comparison

```bash
!python src/compare_models.py --data-dir data --epochs 5
```

This trains the custom CNN, MobileNetV2 and EfficientNetB0 (5 epochs
each, as a first comparison) and writes
`results/model_comparison.csv`. Use more epochs for final numbers.

## 6. Download the trained model and run the app

1. In the Colab `Files` panel, download
   `trained_plant_disease_model.keras` to your own computer.
2. Place it next to `main.py` (in the project folder).
3. Run the app:

   ```bash
   pip install -r requirements.txt
   streamlit run main.py
   ```

4. Open the Streamlit URL, upload a leaf photo, and press Predict.
   A prediction is shown only when confidence is ≥ 80%.

> **If the Drive-save cell fails:** the `drive.mount()` step needs
> you to complete Google's authorisation prompt inside the notebook
> output — if it reports `mount failed`, the model was NOT copied to
> Drive and exists only in this Colab runtime, which disappears when
> the runtime ends. In that case download
> `trained_plant_disease_model.keras` (87 MB) and the `results/`
> folder from the Colab `Files` panel immediately, before closing
> the tab, and upload the model to your Drive manually.

## 7. Save your results back to GitHub

After a successful run, commit the new artefacts (but **never** the
model file or the dataset — both are in `.gitignore`):

```bash
git add results/ README.md
git commit -m "Retrain on Colab: evaluation + comparison results"
git push
```

Then download the `.keras` file separately if you need it elsewhere
(Google Drive, or a release on GitHub — not as a normal repo file).
