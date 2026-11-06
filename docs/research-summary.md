# Leaf Disease Prediction Using Deep Learning: A Convolutional Neural Network Approach to Plant Disease Recognition

**Author:** Muhammad Asad Bukhari
**Affiliation:** BS Information Technology, University of Education Lahore (DG Khan Campus), Pakistan
**Contact:** GitHub — Asadshah-coder

---

## Abstract

Plant diseases are a major cause of crop loss worldwide, and early
identification is critical for effective treatment. This work presents a
convolutional neural network (CNN) that classifies leaf images into 38
healthy and diseased categories, served through a Streamlit web
application for practical use. The model was trained on 70,295 images
and validated on 17,572 images from the public PlantVillage dataset.
Over 10 training epochs, training accuracy rose from 61.50% to 98.18%
and validation accuracy from 83.68% to 96.06%. The project further
provides a reproducible training pipeline, per-class evaluation,
Grad-CAM visual explanations, and a transfer-learning comparison
against MobileNetV2 and EfficientNetB0. The trained model file from
the original project was unavailable, so the current results are the
verified training-history values and a full test-set evaluation is
planned on retraining. The system is intended as a low-cost decision
support tool for farmers and as a basis for further research in
agricultural computer vision.

**Keywords:** plant disease detection; convolutional neural network;
deep learning; PlantVillage; Grad-CAM; precision agriculture

---

## 1. Introduction

Agriculture underpins food security, yet plant diseases destroy a
large share of global crop production every year. Traditional diagnosis
relies on visual inspection by experts or laboratory testing, both of
which are slow, costly, and often unavailable to smallholder farmers.
A farmer who photographs a sick leaf with a phone could, in principle,
receive an immediate preliminary diagnosis — if a reliable model
exists.

Deep learning, and convolutional neural networks in particular, has
proven effective for image classification, including plant-disease
recognition on the PlantVillage dataset. This project applies a custom
CNN to that task and packages it as a usable web application, with the
aim of producing work that is reproducible and evaluable, not only a
demonstration.

## 2. Related Work and Dataset

The dataset is the public PlantVillage dataset (Hughes & Salathé,
2015), a widely used open collection of leaf images labelled by crop
and disease. It contains 38 classes covering 14 crops, including
healthy and diseased leaves. The split used here is 70,295 training
images, 17,572 validation images, and a 33-image sample test set
supplied with the original project files. The full dataset (about
88,000 images, 1.7 GB) is available from the project's Google Drive
link in the repository README.

## 3. Methodology

### 3.1 Model architecture

A custom Sequential CNN was built with five convolutional blocks
(32, 64, 128, 256 and 512 filters; 3×3 kernels, ReLU activation,
2×2 max-pooling), followed by Dropout (0.25), a Flatten layer, a
Dense layer of 1,500 units (ReLU), Dropout (0.4), and a 38-unit
softmax output layer. Input images are 128×128 RGB.

### 3.2 Training

The model was compiled with the Adam optimiser (learning rate
0.0001) and categorical cross-entropy loss, and trained for 10
epochs with a batch size of 32. The improved training script
(`src/train.py`) adds data augmentation — random horizontal flips,
rotations and zoom — applied to the training set only, and uses a
fixed seed (42) for reproducibility.

### 3.3 Evaluation and explainability

Performance is reported as training and validation accuracy and loss
per epoch. The improved pipeline adds per-class precision, recall and
F1-score, a confusion matrix (`src/evaluate.py`), Grad-CAM heatmaps
showing which leaf regions influenced a prediction
(`src/gradcam.py`), and a comparison with ImageNet-pretrained
MobileNetV2 and EfficientNetB0 baselines
(`src/compare_models.py`).

## 4. Results

The verified results of the original 10-epoch training run are:

| Metric | Epoch 1 | Epoch 10 |
|---|---|---|
| Training accuracy | 61.50% | **98.18%** |
| Validation accuracy | 83.68% | **96.06%** |
| Training loss | 1.293 | 0.057 |
| Validation loss | 0.523 | 0.138 |

Accuracy improved steadily across epochs, and the final validation
accuracy of 96.06% indicates the model generalises well to unseen
images from the same distribution. The training and validation curves
are shown in `results/figures/accuracy.png` and
`results/figures/loss.png`. Per-class metrics and a held-out test-set
evaluation are pending retraining, as described in Section 5.

## 5. Discussion and Limitations

The headline accuracy is encouraging, but it must be read with its
limitations. First, the trained model file from the original project
was not available in the project archive, and the training notebook
saved the model under a different filename than the application
expected; this is fixed in the improved code, but predictions cannot
run until the model is retrained. Second, the supplied test set
contains only 33 images, so no reliable test-set figure can be
reported yet; the validation set was also used as the test set in the
original test notebook, which is methodologically incorrect. Third,
the original run used no data augmentation and no per-class metrics.
Finally, PlantVillage images were photographed under controlled
conditions, so performance on real field photographs — different
lighting, backgrounds and leaf orientations — remains untested.

## 6. Future Work

Retraining on Google Colab with the corrected pipeline is the
immediate next step, followed by a proper held-out test evaluation and
the MobileNetV2/EfficientNetB0 comparison. Further work includes
cross-validation, a larger input resolution for the transfer-learning
models, field testing on real photographs, and deployment of the
application with the trained model.

## 7. Conclusion

This project demonstrates that a custom CNN can learn to distinguish
38 plant-disease classes from leaf images with high validation
accuracy, and it provides the reproducible code, evaluation and
explanation tooling needed to develop that result into research-grade
work. The remaining steps — retraining, test-set evaluation and
baseline comparison — are defined and ready to run.

---

## References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of
   images on plant health to enable the development of mobile disease
   diagnostics. *arXiv:1511.08060*. (PlantVillage dataset source.)
2. TensorFlow documentation. https://www.tensorflow.org/ (Keras API
   and training tools used in this project.)
3. Streamlit documentation. https://docs.streamlit.io/ (web
   application framework used in this project.)

*Note: only the results quoted in Section 4 are measured values from
the recorded training history. No other accuracy or evaluation figure
is claimed in this summary.*
