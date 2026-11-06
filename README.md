# Leaf Disease Prediction Using Deep Learning

Final Year Project — Plant Disease Recognition System. A CNN trained to identify plant diseases from leaf images, with a Streamlit app for predictions.

- **Author:** Muhammad Asad Bukhari (Asadshah-coder), BS IT, University of Education Lahore (DG Khan Campus), 2022–2026
- **Stack:** TensorFlow / Keras, OpenCV, Streamlit, Python
- **Classes:** 38 (healthy + diseased, across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)

## Results (from `training_hist.json`, 10 epochs)

| | Final |
|---|---|
| Training accuracy | 98.18% |
| Validation accuracy | 96.06% |
| Training loss | 0.057 |
| Validation loss | 0.138 |

Accuracy rose from 61.5% (epoch 1) to 98.2% (epoch 10). These are the recorded training-history values, not a separately re-run evaluation.

## Model

- CNN (Sequential): Conv2D 32 → 64 → 128 → 256 → 512 blocks with MaxPooling, Dense 1500 (ReLU), output Dense 38 (softmax)
- Input: 128×128 RGB images, batch size 32, Adam (lr 0.0001), categorical cross-entropy, 10 epochs
- Training notebook: `Train_plant_disease.ipynb` (saves the model as `trained_model.keras`)
- Test notebook: `Test_Plant_Disease.ipynb`

## Dataset

- **Train:** 70,295 images · **Validation:** 17,572 images · **Test:** 33 sample images
- The full dataset (~88,000 images, 1.7 GB) is **not** in this repo — GitHub does not host files that large. Download the full project zip (code + dataset) from Google Drive:
  - **[Plant Disease FYP.zip (1.7 GB) — Google Drive](https://drive.google.com/file/d/13_IbE48q2Nv3whIDxVf8OV60JzZGVySx/view?usp=sharing)**
- The dataset folders in the zip are `train/`, `valid/`, and `test/`, organised by class folder (e.g. `Tomato___Early_blight`).

## Run the app

```bash
pip install -r requirements.txt
streamlit run main.py
```

Then upload up to 3 leaf images in the Disease Recognition section and press Predict. Predictions are shown only when confidence is ≥ 80%.

## ⚠️ Known issue — trained model file is missing

`main.py` and `Test_Plant_Disease.ipynb` load `trained_plant_disease_model.keras`, but **that file is not in the original project zip**, and the training notebook saves the model under a different name (`trained_model.keras`). So predictions will fail with a file-not-found error until one of these is done:

1. Retrain with `Train_plant_disease.ipynb` and rename the saved file to `trained_plant_disease_model.keras`, or
2. Place an existing `trained_plant_disease_model.keras` next to `main.py`.

This is the first thing to fix in the improvements round.

## Files

| File | What it is |
|---|---|
| `main.py` | Streamlit app (home + disease recognition) |
| `Train_plant_disease.ipynb` | Data prep, CNN build, training, evaluation, confusion matrix |
| `Test_Plant_Disease.ipynb` | Load the trained model and test on single images |
| `training_hist.json` | Training/validation accuracy & loss per epoch |
| `index.html`, `disease_recognition.html` | Static HTML pages for the project |
| `home_page.png.png` | Banner image used by the Streamlit app |
| `requirements.txt` | Python dependencies |
