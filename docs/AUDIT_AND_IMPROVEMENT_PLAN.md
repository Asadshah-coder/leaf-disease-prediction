# Plant Disease Recognition — Technical Audit & Improvement Plan

**Date:** 7 November 2026
**Repository:** `Asadshah-coder/leaf-disease-prediction` (branch `main`)
**Model under audit:** `trained_plant_disease_model.keras` (retrain #2, 86 MB)
**Live app:** https://leaf-disease-prediction-app.streamlit.app/
**Author:** Muse (audit conducted for Muhammad Asad Bukhari)
**Status:** Audit complete. Implementation NOT started — awaiting approval of this plan.

> **Rule of this document:** every number in Sections 1–4 was measured,
> read from a file, or observed live on the deployed app. Numbers for
> future experiments are marked as *targets*, not results. Nothing here
> is invented.

---

## 1. Executive summary

**The headline issue is real, and it is not a bug in the trained
weights.** When a human photograph was uploaded, the app answered
"Tomato — Late Blight, 41.1%". That happened because the system is a
*closed-set* softmax classifier over 38 leaf classes: it has no way to
say "this is not a leaf", so it must always pick one of the 38
diseases, even for a face, a car, or a blank image. The 41.1%
confidence is the model being *uncertain* — and the app already shows
a "low confidence, treat as uncertain" warning — but an uncertain
disease label is still a disease label, and it must never be presented
as a diagnosis for a non-leaf image.

**One genuine bug was found and fixed during this audit** (see F2):
the app fed raw 0–255 pixel values to a model trained on 0–1 inputs.
Before the fix, all 6 test photos returned the same class at 100%;
after the fix, 4 of 6 were correct. The model itself was never broken.

**The model is good on its own validation set and weaker in the real
world.** Validation accuracy is 94.21% (n = 17,572), but the
validation images are clean PlantVillage lab photos with uniform
backgrounds. On 6 real PlantVillage photos served through the live
app, 4 were correct and 2 fell below the 80% confidence rule. The two
failures were exactly the classes that are weakest in the per-class
report (Tomato Early Blight, Tomato Bacterial Spot), so the weakness
is measurable and specific, not a vague "bad model".

**The plan:** (Phase 1) an input-validation gate that rejects
non-leaf, blank, corrupt and blurry images *before* the disease model
ever sees them, plus confidence calibration; (Phase 2) OOD rejection
measured on real out-of-distribution sets; (Phase 3) model
improvements compared on the same splits, including the transfer
learning comparison that Colab's GPU limit blocked. No model is
replaced without a before/after comparison. All experiments are logged
in the format of Section 7 so they can be written up for the paper.

---

## 2. Evidence base — what was actually inspected

| Item | Source | Result |
|---|---|---|
| Model architecture | `src/model.py` | Custom CNN, exact reproduction of the original notebook (see §3, F3) |
| Training pipeline | `src/train.py` | `image_dataset_from_directory`, Rescaling(1/255) in the data pipeline, augmentation: RandomFlip, RandomRotation 0.1, RandomZoom 0.1; Adam lr 0.0001, 10 epochs, batch 32 |
| Evaluation pipeline | `src/evaluate.py` | Same Rescaling(1/255); validates folder names against `CLASS_NAMES` |
| Grad-CAM | `src/gradcam.py` | Divides by 255.0 before inference |
| App inference (before fix) | `main.py` (commit `dc27c40`) | **No division by 255** — raw 0–255 pixels |
| App inference (after fix) | `main.py` (commit `9b14a76c`) | `img_to_array(image) / 255.0` |
| Class mapping | `main.py` vs `src/model.py` | 38 classes, identical order; `train.py`/`evaluate.py` import from `model.py` (single source of truth) |
| Retrain #2 results | `results/evaluation.json`, `results/classification_report.txt` | Accuracy 0.9421, macro F1 0.9417, n = 17,572 (full per-class table in §3, F6) |
| Training history | `results/training_history.json` | Final train acc 95.20%, val acc 94.21% |
| Live test, pre-fix | Browser, 7 Nov 15:21–15:24 | 6 images → all Tomato Mosaic Virus, 100% (0/6) |
| Live test, post-fix | Browser, 7 Nov 15:31–15:34 | 4/6 correct (see §3, F2) |
| Live test, professional app | Browser, 7 Nov 16:47–16:49 | 3 images: healthy 99.11% ✓; early blight 46.66% correct class, uncertain; bacterial spot → spider mite 64.16% ✗, uncertain |
| Human-photo test | User's camera photo, 7 Nov 16:51 | Face → Late Blight 41.1%, Cedar Apple Rust 27.1%, Leaf Scorch 14.0% (top 3) |
| Dataset | `Plant Disease FYP.zip` (damaged) | 87,996 entries: train 70,295 / valid 17,572 / test 33; per-class *train* counts not verifiable from the damaged zip (limitation L3) |
| Backup | `~/workspace/fyp/backups/improved-2026-11-07-pre-audit.tar.gz` | 39 files, code snapshot before any change (model file excluded — it is preserved on Google Drive and in the working tree) |
| Repo state | GitHub `main` | Head `34254077` at time of audit |

---

## 3. Findings

### F1 — The model cannot say "not a leaf" (root cause of the human-photo issue)

**Severity: high. Evidence: the user's selfie test.**

The network ends in a 38-way softmax. A softmax always sums to 1
across the 38 classes, so *every* input — a face, a car, a blank
image — receives a plant-disease label. There is no rejection class,
no out-of-distribution (OOD) check, and no input-quality check
anywhere in the pipeline. The 41.1% top confidence on the selfie is
the model being uncertain, but the app still displayed a disease name
as the headline result. The "low confidence" warning mitigates this;
it does not fix it.

**Implication:** any claim that the app "diagnoses" real-world photos
is unsafe until a gate exists. This is the single most important
finding for the paper: it is a clean, demonstrable limitation of
closed-set classifiers, and the fix (a plant/non-plant gate) is a
genuine contribution.

### F2 — Preprocessing mismatch in the app: found and fixed

**Severity: high (was). Evidence: live 0/6 → 4/6.**

| Stage | Normalisation | Where |
|---|---|---|
| Training | ÷ 255 (`Rescaling(1.0/255)`) | `src/train.py`, data pipeline |
| Evaluation | ÷ 255 | `src/evaluate.py`, data pipeline |
| Grad-CAM | ÷ 255 | `src/gradcam.py:170` |
| App, before 7 Nov 15:24 | **none (raw 0–255)** | `main.py` `predict_image` |
| App, after 7 Nov 15:24 | ÷ 255 | `main.py`, commit `9b14a76c` |

Before the fix, every image — healthy tomato, healthy apple, three
different tomato diseases — returned Tomato Mosaic Virus at 100%.
After the fix, the same 6 images gave 4 correct predictions with
varied confidences (99.11%, 99%, 99.84%, 98.22%) and 2 below-threshold
results. The model was never at fault; the app was feeding it inputs
255× larger than the ones it was trained on, which saturates the
network into a single class.

**Deeper issue:** the `Rescaling` layer lives in the *data pipeline*,
not inside the saved model. Any future consumer of the `.keras` file
(a new app, a notebook, a paper reproduction) will silently make the
same mistake. **Recommendation:** bake the rescaling into the model
itself (a `Rescaling` layer as the first model layer) when the model
is next retrained, and verify the saved model accepts raw 0–255 input.

### F3 — Architecture: faithful reproduction, with known quirks

**Severity: low (documentation). Evidence: `src/model.py`.**

Conv blocks 32/32, 64/64, 128/**32** (the original notebook's quirk —
the second convolution in the 128-filter block uses 32 filters, kept
deliberately for comparability), 256/256, 512/512, Dropout 0.25 →
Flatten → Dense 1500 → Dropout 0.4 → Dense 38 softmax. Adam
lr 0.0001, categorical cross-entropy, 128×128 input, batch 32.
No batch normalisation, no residual connections, no attention. The
architecture is simple and dated compared with modern transfer
learning — which is exactly why the comparison experiment (E3) matters
for the paper.

### F4 — Class mapping is consistent

**Severity: none. Evidence: programmatic check.**

`main.py`, `src/model.py` and the training/evaluation code all use the
same 38-class list in the same order, imported from one place
(`model.py`). The dataset folder names match the list (verified by
`evaluate.py`'s own folder-name check). The one known naming quirk —
the dataset folder `Tomato___Spider_mites Two-spotted_spider_mite`
uses a space where the code uses an underscore — does not change the
index order and does not affect predictions.

### F5 — Overfitting: no gross overfitting, but validation overstates real-world performance

**Severity: medium. Evidence: training history + live tests.**

Final train accuracy 95.20% vs validation accuracy 94.21%: a gap of
0.99 percentage points. That is *not* severe overfitting. The problem
is different: the validation set comes from the same clean,
lab-photographed PlantVillage distribution as the training set
(single leaf, uniform background, controlled light). Real-world photos
— field backgrounds, shadows, multiple leaves, phone cameras — are a
different distribution, and the model's own numbers say nothing about
them. Live evidence: 4/6 correct on PlantVillage photos through the
app, and 2 of the 6 below the 80% confidence rule. A proper
real-world test set does not exist yet (see E4 and limitation L2).

### F6 — Weakest classes are specific and measurable

**Severity: medium. Evidence: `results/classification_report.txt`,
retrain #2, n = 17,572.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Tomato Septoria leaf spot | 0.9714 | **0.7018** | **0.8149** | 436 |
| Soybean healthy | 0.9945 | **0.7208** | **0.8358** | 505 |
| Tomato Target Spot | 0.8959 | 0.8096 | 0.8506 | 457 |
| Potato healthy | **0.7726** | 0.9759 | 0.8624 | 456 |
| Tomato Early Blight | 0.8680 | 0.9042 | 0.8857 | 480 |
| Tomato Spider Mites | 0.8401 | 0.9540 | 0.8934 | 435 |
| Tomato Leaf Mold | 0.8339 | 0.9936 | 0.9068 | 470 |
| Tomato Late Blight | 0.9023 | 0.9179 | 0.9101 | 463 |
| Pepper bell healthy | 0.8561 | 0.9819 | 0.9147 | 497 |
| Peach Bacterial Spot | 0.9533 | 0.8889 | 0.9200 | 459 |

Overall: accuracy 0.9421, macro precision 0.9460, macro recall
0.9424, macro F1 0.9417. Validation support per class is 410–505, so
there is **no severe class imbalance in the validation split**; the
weak classes are genuinely hard (Septoria and Target Spot are
visually similar to other tomato spots; Soybean healthy is confused
with other healthy leaves; Potato healthy has the lowest precision,
meaning other classes are often mislabelled as potato healthy).

Notably, the two live failures were exactly Tomato Early Blight and
Tomato Bacterial Spot — both in the weaker half of the table. The
live evidence and the per-class report agree.

### F7 — Confidence is uncalibrated

**Severity: medium. Evidence: live tests.**

Softmax outputs are routinely treated as probabilities, but they are
not calibrated: before the /255 fix the model reported 100% confidence
on wrong answers; after the fix it still reports 99%+ on some correct
answers and 41–64% on wrong ones, with no measured relationship
between confidence and correctness. No calibration has ever been
measured (no ECE, no reliability diagram, no temperature scaling).
A threshold (80%) was inherited from the original project without any
measurement behind it. **Recommendation:** fit temperature scaling on
the validation logits, report ECE before/after, and choose the
rejection threshold from an ROC curve on ID vs OOD data — not from
tradition.

### F8 — No input-quality checks

**Severity: medium. Evidence: code inspection.**

The app accepts any file PIL can open: blank images, tiny images,
extremely blurry photos, screenshots, and non-leaf photos all reach
the model. There is no check for corrupt data, blank/near-blank
frames, blur, resolution, or aspect ratio. These checks are cheap
(pixel statistics, Laplacian variance) and belong in the gate
(Phase 1).

### F9 — App-level observations (minor)

- Library search works but is slow: typing "tomato" shows the filtered
  "10 of 38" only after several seconds (verified 16:51 PKT). Not a
  bug; a Streamlit rerun timing artefact. Consider debouncing or a
  note — low priority.
- The sidebar diagnosis count lags one rerun after diagnosing.
  Cosmetic; fix with a placeholder when convenient.
- `use_container_width` deprecation warnings appear in the logs.
  Harmless; clean up in a later pass.

### F10 — Dataset and evaluation limits

- The original `test/` folder has only 33 images. It is a sample set,
  not a held-out test set. All headline numbers are *validation*
  numbers.
- No real-world (field) test set exists yet.
- No OOD set exists yet.
- Per-class *training* counts could not be verified: the dataset zip
  has a missing central directory and the repaired copy is incomplete,
  so folder counts from it are unreliable. Validation support
  (410–505/class) is the only verified balance evidence.

---

## 4. Root-cause summary (the human-photo question, answered)

1. The model is a closed-set 38-way softmax classifier. It *cannot*
   output "not a leaf". (F1)
2. It was trained only on leaf photos, so its feature space has no
   representation of faces, animals, vehicles or buildings. A face
   maps to whichever leaf class is nearest in that space — Late Blight
   at 41.1% in this case. (F1)
3. The 41.1% confidence is low, and the app correctly flagged it
   uncertain — but a flagged wrong answer is still a wrong answer
   shown to the user. (F7)
4. This is **not** overfitting and **not** a broken model: the
   train/val gap is under 1 point, and the same model scores 94.21%
   on validation. It is a missing rejection mechanism plus an
   uncalibrated confidence score. (F5, F7)

---

## 5. Improvement plan

### Phase 0 — Safeguards (done, 7 Nov 2026)

- [x] Backup of the working tree:
  `~/workspace/fyp/backups/improved-2026-11-07-pre-audit.tar.gz`
  (39 files; model excluded, preserved on Drive + working tree).
- [x] Repo head recorded: `34254077`.
- [ ] Rule for all later phases: no change lands without a
  before/after comparison on the same test data, and no model file is
  replaced without beating the baseline on the agreed metrics.

### Phase 1 — Input validation gate (no retraining; deployable first)

**1a. Quality gates (deterministic, runs first, ~milliseconds).**
Reject with a clear message when the upload is: unreadable/corrupt;
blank or near-blank (pixel standard deviation below a threshold set
from data); extremely blurry (Laplacian variance below a threshold
set from data); or below a minimum resolution. Warn (not reject) for
moderate blur. Thresholds are *measured* on the validation set, not
guessed — see experiment E4.

**1b. Plant-vs-non-plant gate (the core fix).**
Recommended approach: a lightweight binary classifier built on a
frozen ImageNet-pretrained MobileNetV2 feature extractor plus a small
classification head, trained on:
- *Positive class:* leaf images from the existing PlantVillage
  training split (already on hand, 70,295 images).
- *Negative class:* non-plant images from public sets — people,
  animals, vehicles, buildings, indoor scenes (e.g. a subset of
  ImageNet/COCO/Caltech-101; exact source to be recorded in the
  experiment log).
Target: a model under ~20 MB that runs on CPU inside the Streamlit
app, so the live demo needs no GPU. Training needs one Colab GPU
session (see §6).

*Alternatives considered and why they are not the primary approach:*
- Softmax-confidence threshold alone — **rejected as the sole gate**:
  the model has reported 100% confidence on garbage inputs (pre-fix)
  and 41–64% on wrong answers; confidence does not separate
  in-distribution from out-of-distribution reliably.
- Energy score / Mahalanobis distance on the disease model's features
  — useful as a *second* signal (Phase 2), but weaker on its own than
  a dedicated gate trained with real non-plant examples.
- CLIP zero-shot ("a photo of a plant leaf" vs "a photo of a person")
  — accurate, but the model is large for the free Streamlit tier and
  adds a heavy dependency; keep as a fallback if the MobileNetV2 gate
  underperforms.

**1c. UX contract.** The gate runs *before* the disease model. A
rejected image shows: what was detected ("This does not look like a
plant leaf"), why, and what to do (upload a clear photo of a single
leaf). No disease prediction is shown for rejected images. Accepted
images proceed to the existing top-3 display with the calibrated
confidence (Phase 2).

### Phase 2 — Calibration & OOD rejection (no retraining)

**Status (7 Nov 2026): COMPLETE + WIRED IN (experiment E5).**
Temperature T = **1.0616**, ECE 0.0077 → 0.0042, accuracy unchanged
(94.21%). The model was already well calibrated — the honest finding
is that calibration was never the main problem. Energy beat
max-softmax as the OOD signal (AUC 0.8902 vs 0.8097); threshold
6.0440 at 95% TPR flags 4.96% of real leaves and catches 55.1% of
non-leaf images — a second layer only, the gate (99.87%) remains the
primary defence. Old 80% rule: flagged 9.95% as uncertain, 61% of
those were actually correct. The shipped uncertainty rule is the
energy threshold OR calibrated top confidence below 50% (union ≤ 7.5%
of validation leaves). Full log: `docs/experiments/E5.md`.
Wired into `main.py` (calibrated top-3 + energy uncertainty rule,
80% rule removed) — live verification pending.

- **2a. Temperature scaling:** fit a single temperature T on the
  validation logits; report Expected Calibration Error (ECE) and a
  reliability diagram before/after. Ship the fitted T in the app.
- **2b. OOD scoring:** compute max-softmax and energy scores for
  validation images vs the OOD set; choose the rejection threshold
  from the ROC curve (report AUROC, and FPR at 95% TPR). Combine with
  the Phase 1 gate: reject if *either* signal fires.
- **2c. Reporting:** top-3 predictions shown with calibrated
  confidences; the 80% rule is replaced by the measured threshold,
  and the change is documented with before/after numbers.

### Phase 3 — Model improvements (needs GPU; compared, never assumed)

- **E2 — Preprocessing & augmentation:** bake `Rescaling` into the
  model; stronger augmentation (brightness/contrast, RandAugment or
  Cutout, in addition to the existing flip/rotation/zoom); class
  weighting for the weak classes of F6; train 10 epochs on the same
  split so the comparison with E1 is fair.
- **E3 — Transfer learning:** MobileNetV2 vs EfficientNetB0
  (ImageNet-pretrained, frozen base + new 38-class head), same
  pipeline, 5 epochs each. **Status (8 Nov 2026): COMPLETE, 5-epoch
  run on Colab T4** — custom CNN **0.9239**, MobileNetV2 0.8884,
  EfficientNetB0 **0.0273 = failed run** (random chance, loss ≈
  ln 38; likely preprocessing mismatch + frozen BatchNorm, not a
  real architecture result — must NOT be quoted as EfficientNet's
  accuracy; needs a focused re-run with per-base preprocess_input).
  The custom CNN beats the valid baseline; the comparison is
  two-model + one documented failure until EfficientNet is re-run.
  Full log: `docs/experiments/E3.md`.
- **Winner rule:** a new model replaces the baseline only if it beats
  it on validation accuracy *and* macro F1 *and* the real-world set
  (E4), with the comparison published in the experiment log.

### Phase 4 — Real-world validation protocol (the honest number for the paper)

Build two small, labelled sets (target 100–200 images each; collection
is partly the user's job — phone photos of real leaves):
- **RW-Field:** real leaves photographed in the field/garden — varied
  backgrounds, lighting, angles, multiple leaves, phone camera.
- **OOD:** people, animals, vehicles, buildings, indoor scenes, blank
  images, blurry photos.
Report per-set accuracy for every model version. This is the number
that belongs in the paper next to the 94.21% validation figure, with
the gap discussed explicitly.

**Status (8 Nov 2026): Batch 1 done (E6 log).** 3 houseplant photos
(fiddle-leaf fig, monstera, philodendron — all outside the 38
classes, so no accuracy number): gate rejected 2/3 real leaves (7.5%,
38.1% — busy backgrounds; lab-measured recall 99.74% does not hold
on real-world photos), and the philodendron was confidently labelled
"Potato — Healthy" 85.1% with no warning — gate + energy catch
non-leaf images, not unseen leaf species. Crop photos with known
labels still needed for a scored RW-Field accuracy.

### Phase 5 — Paper documentation

- Experiment log for E1–E5 in the format of §7, kept in
  `docs/experiments/` in the repo.
- Paper outline: abstract, introduction, related work (PlantVillage
  baselines, OOD detection, calibration), method (gate + calibration),
  experiments, results, limitations, conclusion. Target audience: AI /
  Computer Vision professors in China — the paper should lead with the
  rejection-gate contribution, because it is the novel, demonstrable
  part; the 94.21% classifier alone is a standard PlantVillage result.

---

## 6. Experiments E1–E5

| # | Experiment | Hypothesis | Dataset | Config | Metrics | Status |
|---|---|---|---|---|---|---|
| E1 | Baseline (existing model) | — (reference point) | valid 17,572; live 6-image set; human photo | Current `.keras`, /255 fix in app | Acc 0.9421, macro F1 0.9417, per-class P/R/F1 (see F6); live 4/6; selfie → Late Blight 41.1% | **Done** |
| E2 | Preprocessing + augmentation | Baking rescaling into the model and stronger augmentation improves real-world accuracy without hurting validation | Same train/valid split; RW-Field | Rescaling in model; +brightness/contrast, RandAugment; class weights; 10 epochs, Adam 1e-4 | Val acc/macro F1; RW-Field acc; per-class F1 for weak classes | Planned — needs GPU |
| E3 | Transfer learning | ImageNet-pretrained MobileNetV2 / EfficientNetB0 beat the custom CNN on the same split | Same train/valid split; RW-Field | Frozen base + new head; 5 epochs each | Val acc/macro F1; RW-Field acc; training time | **Blocked** — Colab GPU usage limit, 7 Nov 14:36; retry after reset or paid units |
| E4 | Plant/non-plant gate + quality checks | A lightweight binary gate rejects non-leaf images with ≥95% accuracy while keeping ≥99% of true leaves | Leaf: PlantVillage train split; non-leaf: public non-plant set (to be recorded); quality thresholds from valid set | MobileNetV2 frozen + small head; quality gates (blank/blur/resolution) | Gate precision/recall/F1; leaf acceptance rate; non-leaf rejection rate; end-to-end: selfie must be rejected, 6 leaf photos must pass | Planned — needs GPU for gate training |
| E5 | Calibration + OOD rejection | Temperature scaling + an OOD threshold reduce confident wrong answers on OOD data | Valid logits; OOD set | Temperature fit on valid; ROC-based threshold | ECE before/after; AUROC (ID vs OOD); FPR at 95% TPR; % of OOD images rejected | Planned — no GPU needed (logits only) |

**Before/after comparison rule:** every experiment reports the same
table (accuracy, macro F1, per-class F1 for the 10 weakest classes,
RW-Field accuracy, OOD rejection rate) so the paper can show exactly
what changed and what did not.

---

## 7. Experiment log template (one file per experiment, `docs/experiments/`)

```markdown
# E# — <name>
- Date:
- Hypothesis:
- Dataset (exact source, size, split):
- Configuration (architecture, preprocessing, hyper-parameters, seed):
- Results (accuracy, precision, recall, F1, confusion matrix, per-class table):
- Before/after comparison (vs E1):
- Limitations:
- Conclusions:
- Artefacts (model file, log, figures):
```

---

## 8. Risks, limitations & compute plan

- **Compute:** Colab's free GPU quota is exhausted for today
  (7 Nov, usage limits after a full day of GPU work). E2–E4 need GPU.
  Options: (a) retry in the evening after the quota resets (free);
  (b) Colab Pay As You Go (user's payment decision); (c) train the
  small gate on CPU — slow but possible for a tiny head. E5 needs no
  GPU. The live Streamlit app is CPU-only, so everything shipped
  there must be small.
- **Data:** no real-world or OOD set exists yet; collecting one is
  required before any generalisation claim. The user can contribute
  phone photos of real leaves (healthy and diseased).
- **Dataset provenance:** the source zip is damaged; per-class
  *training* counts are unverifiable. Any paper must state this.
- **Scope control:** the gate and calibration (Phases 1–2) are
  shippable without retraining the disease model. Model replacement
  (Phase 3) is a separate decision with its own comparison.

---

## 9. Decisions needed from the user

1. **Approve Phase 1** (quality gates + plant/non-plant gate + app
   wiring) so implementation can start. No model replacement is
   involved.
2. **Compute for Phases 1b/3:** retry Colab GPU in the evening (free)
   or pay for Colab compute units? The gate training is short
   (one session); the transfer-learning comparison is the longer job.
3. **Real-world photos:** can you photograph ~20–50 real leaves with
   your phone (healthy + visibly diseased, outdoors, normal
   background) for the RW-Field set? This is the most valuable data
   for the paper.
4. **Paper target:** any specific universities or professors in China,
   or a general shortlist? (Optional — affects how the paper is
   framed, not the experiments.)

---

## Appendix A — Live-test evidence (7 Nov 2026)

| Test | Image | Expected | Got | Confidence | Verdict |
|---|---|---|---|---|---|
| Pre-fix (6) | 2× healthy tomato, early blight, late blight, bacterial spot, healthy apple | varied | Tomato Mosaic Virus (all 6) | 100% (all) | 0/6 — app bug, not model |
| Post-fix (6) | same 6 | varied | 4 correct | 98–99.8% | 4/6; early blight + bacterial spot below 80% |
| Pro app (3) | healthy tomato, early blight, bacterial spot | varied | Healthy 99.11% ✓; Early Blight 46.66% ✓ (uncertain); Spider Mite 64.16% ✗ (uncertain) | — | Gate + calibration would change the last two rows' handling |
| Selfie (1) | user's face | *reject* | Late Blight 41.1% (top 3: 41.1 / 27.1 / 14.0) | 41.1% | Must be rejected by the gate |

## Appendix B — Files changed during this audit

- None. This audit only read files. The backup
  (`improved-2026-11-07-pre-audit.tar.gz`) was created before any
  future change.
- Earlier fix referenced here: `main.py` `/255` normalisation,
  commit `9b14a76c` (7 Nov 15:24 PKT), verified live 4/6.
