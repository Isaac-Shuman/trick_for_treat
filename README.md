# TrickForTreat: Posture Pipeline

A **YOLO dog detector → square dog crop → posture classifier (sit / down / stand)** pipeline, trained and evaluated on the public DogPoseCV dataset (`stockeh/dog-pose-cv`). Every script is implemented and runs on a CPU-only laptop.

Out of scope for now: the temporal gate, the LLM verifier, bark detection, MQTT/Nano, ESP32 firmware, and the state machine.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

Model weights download on first use: YOLO into `models/`, timm through the Hugging Face cache.

## Run order

| Step | Command | Time on a 4-core CPU |
|---|---|---|
| 1. Download + join labels | `.venv/bin/python scripts/01_download.py` | ~15 s (plus a one-time 780 MB download) |
| 2. Map labels | `.venv/bin/python scripts/02_filter_labels.py` | seconds |
| 3. Dedup + test split | `.venv/bin/python scripts/03_split.py` | ~15 s |
| 4. Detect + crop | `.venv/bin/python scripts/04_crop_dogs.py` | ~30 min (YOLO11s) |
| 5. Train + evaluate | `.venv/bin/python scripts/05_train_classifier.py` | ~2.5 h (5-fold CV + final retrain); add `--reuse-folds` to resume after a crash using saved fold checkpoints |
| 6. Run the pipeline | `.venv/bin/python scripts/06_run_pipeline.py --source data/split_manifest.csv --split test` | minutes |

Every script accepts `--limit N` for a fast smoke test. Script 05 also takes `--epochs N --folds 0,1`.

Each script prints numbered sections, finishes with `[ok]` sanity checks (an `assert` stops it with a clear message), and writes debug images to `results/debug/`. The **DEBUGGING GUIDE** at the bottom of each script explains what to look for.

| If you change...                          | Re-run     |
|-------------------------------------------|------------|
| `labels.*`                                | 02 → 05    |
| `dedup.*`, `split.*`, or `seed`           | 03 → 05    |
| `detector.*` or `crop.*`                  | 04 → 05    |
| `classifier.*`, `train.*`, `eval.*`, `augment.*` | 05  |

## Layout and extension points

```
configs/default.yaml     every path and tunable
scripts/01-06            thin drivers (dataset, CSV, CV and reporting logic)
trickforTreat/           importable code shared by the scripts and the future live system
  sources.py             ImageSource: folder, image file, manifest CSV, video file, webcam, MJPEG URL
  detector.py            Detector protocol, YoloDogDetector, select_and_crop (the one crop function)
  transforms.py          eval transform (runtime) + training augmentations
  classifier.py          PostureModel protocol, build_model (any timm backbone), PostureClassifier (checkpoint I/O,
                         predict, should_reward)
  pipeline.py            PosturePipeline: source -> detector -> crop -> classifier -> reward decision
  metrics.py             classification metrics, reward curve, threshold choice, plots
  common.py              config, paths, seeding, CSV I/O, logging, contact sheets, charts
```

- **New image source** (ESP32-CAM, your own videos): `open_source("http://<esp32-ip>:81/stream")` already works through OpenCV. Any other source just subclasses `ImageSource` and yields `Frame`s. Neither the pipeline nor the models change.
- **Different classifier backbone**: set `classifier.backbone` to any timm model name (e.g. `efficientnet_b0`, `mobilenetv3_small_100`) and re-run 05. A non-timm model only needs to implement the `PostureModel` protocol (`classes`, `predict_batch`, `should_reward`).
- **Different detector**: implement `Detector.detect(images) -> list[FrameDetections]`.
- **Checkpoint**: `checkpoints/final.pt` holds weights, class order, preprocessing, and reward thresholds, so `PostureClassifier.load()` is all the runtime needs.

## Key decisions

| Topic | Decision |
|---|---|
| Labels | From the per-breed CSVs in `labels.tar.gz`. The HF loader doesn't attach them, and the dataset card's standing/lying counts appear swapped |
| Unlabeled images | Skipped: Standard Schnauzer (no CSV), 2 Leonberg images, the `validation/` folder |
| `undefined` | Always dropped |
| Near-duplicates | Perceptual hash, ≤ 4 bits apart; one image per group kept at random, before the split |
| Detector | YOLO11s, COCO `dog` only, conf ≥ 0.25. YOLO11n misread too many dogs as bears/teddy bears |
| No dog / several dogs / box < 32 px | Image dropped and logged |
| Crop | Square around the box padded 10%, gray fill past the image edge, resized with no center crop |
| Classifier | timm MobileNetV3-Large, ImageNet-pretrained, 160 px input (CPU budget), head-only first epoch |
| Imbalance | None: plain cross-entropy |
| Reward rule | Argmax plus threshold; threshold = lowest value reaching 0.85 precision on pooled out-of-fold CV predictions |
| Final model | Retrained on all trainval for the mean best epoch + 1 |

## Dataset findings

- The HF repo is two tarballs. Labels are `standing / sitting / lying / undefined` (not "lying down").
- 20,423 labeled images, 119 breeds. After dropping `undefined`: 7,047 stand, 4,110 down, 3,008 sit.
- 116 near-duplicate images were removed. Some are the same photo filed under two breeds.

## Results

Full numbers: `results/metrics.json`, plots and image grids in `results/debug/05_*`.
Model: `checkpoints/final.pt` (mobilenetv3_large_100, 160px, 13 epochs on all 10,953 trainval crops).

**Posture classifier, held-out test crops (1,926)**

| class | precision | recall | F1 |
|---|---|---|---|
| sit | 0.771 | 0.771 | 0.771 |
| down | 0.850 | 0.883 | 0.866 |
| stand | 0.928 | 0.906 | 0.917 |

Macro F1 0.851 (5-fold CV: 0.851 ± 0.007), accuracy 0.870. sit→down 11.4%, down→sit 8.3%.

**Reward decision (threshold chosen on out-of-fold CV predictions, applied to test)**

| command | threshold | precision | recall | false-reward rate |
|---|---|---|---|---|
| sit | 0.81 | 0.841 | 0.650 | 3.4% |
| down | 0.30 | 0.850 | 0.883 | 6.5% |

Feasibility bar (precision ≥ 0.85 for both targets, sit↔down < 10%): **narrowly not met**: sit precision
0.841, and sit→down is 11.4%. Looking at `05_test_false_rewards.jpg`, a good share of the "false" sit
rewards are images labeled `standing` in DogPoseCV that clearly show a sitting dog, so measured precision
understates the real precision somewhat. The real test is the owner's own dog from the actual camera angle.

**Latency (CPU, i7-11370H, batch 1):** classifier 7.2 ms/crop. Pipeline end to end (`06_run_pipeline.py`,
300 test photos): detect 65 ms + crop 0.4 ms + classify 8 ms ≈ 74 ms/frame (13 fps). 91% of photos
yield exactly one dog; accuracy on those 0.84.

**Possible improvements, roughly in order of payoff:** fine-tune on frames of the actual dog from the
mounted camera; train at 224px (≈2× slower training, likely better sit/down separation); more epochs
(4 of 5 folds were still improving at epoch 12); clean the mislabeled sit/stand images.
