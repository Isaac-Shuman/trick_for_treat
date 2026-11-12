# TrickForTreat: Posture Pipeline Feasibility (Claude Code Handoff)

## Your task
Write **pseudocode only** (no working implementation yet) for the five scripts below. Create one file per script under `scripts/`, using Python-flavored pseudocode in comments and docstrings, plus function signatures, inputs/outputs, and TODOs. Also write `README.md` with run order and a `configs/default.yaml` listing every tunable. When done, summarize open questions for me. Do not install packages or download data yet.

## Project context
TrickForTreat is an IoT dog trainer (college embedded systems class). An ESP32-CAM streams MJPEG to a laptop, and the laptop runs the whole ML pipeline. Final pipeline: YOLO dog detector -> posture classifier on dog crop -> temporal gate -> frontier LLM verifier. Commands in scope: "sit" and "down" (also "stand" as a non-target class).

**Current goal:** prove the detector -> classifier pipeline is feasible on a public dataset. The temporal gate, ESP32, MQTT, and LLM verifier are out of scope for now. Do not write them.

Stack: Python on a laptop, PyTorch, Ultralytics YOLO, torchvision or timm, Hugging Face `datasets`, scikit-learn.

## Dataset
DogPoseCV on Hugging Face: `stockeh/dog-pose-cv` (Apache-2.0).
- 20,578 images, 120 breeds, adapted from Stanford Dogs.
- Classes: standing (4143), sitting (3038), lying down (7090), undefined (6307, mostly close-up portraits).
- Class imbalance: lying down is about 2x sitting.
- Repo and paper: github.com/stockeh/canine-embedded-ml, arxiv.org/abs/2101.02380.

## Scripts to pseudocode

### 1. `01_download.py`
- Load the dataset via `datasets.load_dataset`, using the labels that ship with it.
- Print the actual label names from the dataset's `ClassLabel` feature (verify exact strings; do not assume).
- Save images to `data/raw/<label>/` plus a `manifest.csv` (image_id, path, original_label, width, height).
- Print class counts as a sanity check.

### 2. `02_filter_labels.py`
- Keep only sitting, lying down, standing.
- Map to project labels: sitting -> `sit`, lying down -> `down`, standing -> `stand`.
- Drop `undefined` by default, but make this a config flag so it can instead become an `other` class later (it matters for the close-up camera concern).
- Output `data/filtered_manifest.csv`.

### 3. `03_split.py`
- Carve out a held-out test set (config: 15%, stratified, fixed seed). Never touch it during cross-validation.

### 4. `04_crop_dogs.py`
- Run an Ultralytics YOLO (nano or small) model on each filtered image, keeping only COCO class `dog` (id 16).
- Pick the highest-confidence dog box. Add configurable padding (e.g., 10%), clamp to image bounds, and crop.
- Save crops to `data/crops/<label>/` and write `crops_manifest.csv`.
- If no dog is detected, skip, log the image, and report the no-detection rate overall and per class. Open question to flag: whether to fall back to the full image for those cases.
- Make the confidence threshold configurable.

### 5. `05_train_classifier.py`
- Fine-tune a small pretrained classifier on the crops (config: MobileNetV3 or EfficientNet-B0), with `sit`/`down`/`stand` heads.
- Training loop with K-cross fold validation.
- Handle class imbalance with class-weighted loss (or a weighted sampler), configurable.
- Augmentations should target the real deployment: color jitter and brightness variation (the room has a window, so lighting changes), horizontal flip, light random crop and scale, blur.
- Metrics per fold and on the held-out test set: per-class precision and recall, F1, confusion matrix. Emphasize **precision on the reward decision** (a wrongly given treat is worse than a missed one).
- Log per-stage inference latency (crop to prediction) to inform the later embedded/latency discussion.
- Save the best checkpoint plus a `results/metrics.json`.

## Conventions
- Fixed random seeds everywhere, all paths and hyperparameters in `configs/default.yaml`.
- Each script runnable standalone, reading the previous script's outputs.
- Keep it modular so later pipeline stages (temporal gate, live ESP32 MJPEG input) can reuse the detector and classifier as importable modules (`trickforTreat/detector.py`, `trickforTreat/classifier.py`), so please note where the shared code should live.
- Add clear debugging instructions at the end of every module so that I can clearly understand what the intermediates are.
- Add sanity checks (e.g., asserts) whenever possible at the end of modules to make sure we don't go off the rails in the actual code.

## Out of scope for now
Temporal gate, frontier LLM verification, bark detection, MQTT/Nano, ESP32 firmware, state machine.