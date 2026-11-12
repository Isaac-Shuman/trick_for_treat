# TrickForTreat: Posture Pipeline Feasibility

**Status: pseudocode only.** Nothing here runs yet. The files define structure, inputs and outputs, sanity checks, and TODOs.

The goal is to show that a **YOLO dog detector, followed by a posture classifier on the dog crop**, can tell `sit` / `down` / `stand` apart on the public DogPoseCV dataset (`stockeh/dog-pose-cv`, Apache-2.0). Precision on the reward decision is the metric that matters most.

The temporal gate, LLM verifier, ESP32, MQTT, bark detection, and state machine are out of scope for this stage. A second test on recordings of the real dog from the real camera comes after this one works.

## Layout

```
configs/default.yaml        every path + tunable (seed, labels, dedup, detector, crop, model, train, eval, aug)
scripts/
  01_download.py            HF dataset -> data/raw/<label>/*.jpg + manifest.csv; prints exact label names
  02_filter_labels.py       map to sit/down/stand, drop undefined -> filtered_manifest.csv, classes.json
  03_split.py               drop near-duplicates (phash), then stratified 15% test split -> split_manifest.csv
  04_crop_dogs.py           YOLO (COCO dog=16); drop no-dog / multi-dog / tiny; square crop -> data/crops/
  05_train_classifier.py    K-fold CV -> epoch count + reward thresholds; retrain; test eval; latency
trickforTreat/              shared, importable code (reused later by the live ESP32 pipeline)
  detector.py               DogDetector, square_crop_box, crop_with_fill
  classifier.py             build_model, build_eval_transform, PostureClassifier (load/save/predict/should_reward)
  transforms.py             (planned) training augmentations only
  common.py                 config, seeding, CSV I/O, device pick, debug contact sheets
```

**Where the shared code lives:** anything the live loop will need goes in `trickforTreat/`: detection, square cropping, eval preprocessing, model construction, checkpoint format, and the per-frame reward check. The scripts are thin drivers. They hold the dataset, CSV, dedup, CV, and reporting logic, which the live system never imports.

## Key decisions

| Topic | Decision |
|---|---|
| Dataset version | Latest from Hugging Face, not pinned |
| `undefined` label | Always dropped (script 02) |
| Near-duplicates | Dropped before the split; one image per group kept at random, even across labels (script 03) |
| No dog detected | Image skipped and logged (script 04) |
| Multiple dogs | Image dropped and logged (script 04) |
| Crop shape | Square around the padded box, gray fill past the image edge, resized to 224 with no center crop |
| Class imbalance | No weighting, no sampler. Plain cross-entropy |
| Reward rule | Argmax plus threshold: reward `t` only if `argmax == t` and `p(t) ≥ threshold[t]` |
| Threshold choice | Lowest threshold reaching 0.85 precision on pooled out-of-fold CV predictions |
| Final model | Retrain on all trainval for the mean best epoch count across folds |
| Feasibility bar | Test precision ≥ 0.85 for sit and down; sit↔down confusion < 10% |

## Run order

```bash
python scripts/01_download.py        --config configs/default.yaml
#   STOP: copy the printed ClassLabel names into configs/default.yaml -> labels.mapping / dropped_original_labels
python scripts/02_filter_labels.py   --config configs/default.yaml
python scripts/03_split.py           --config configs/default.yaml
#   STOP: open results/debug/03_duplicate_groups.jpg; tune dedup.max_hamming_distance if needed
python scripts/04_crop_dogs.py       --config configs/default.yaml
#   STOP: read results/detection_report.json (per-class drop rates) and the 04 debug grids
python scripts/05_train_classifier.py --config configs/default.yaml
```

Smoke test first. Pass `--limit 50` to every script, and add `--folds 0` for 05 with `train.epochs: 1`. Every output file should appear in a few minutes on CPU.

Each script reads only the previous script's outputs, so you can re-run from any step:

| If you change...                          | Re-run     |
|-------------------------------------------|------------|
| `labels.*`                                | 02 → 05    |
| `dedup.*`, `split.*`, or `seed`           | 03 → 05    |
| `detector.*` or `crop.*`                  | 04 → 05    |
| `classifier.*`, `train.*`, `eval.*`, `augment.*` | 05  |

## Data flow

```
manifest.csv ─02→ filtered_manifest.csv + classes.json
             ─03→ duplicates_dropped.csv + split_manifest.csv + test_ids.txt
             ─04→ crops/ + crops_manifest.csv + crop_drops.csv + detection_report.json
             ─05→ checkpoints/final.pt + results/metrics.json
```

The test split is chosen at the image level in 03, after dedup and before cropping. 04 then drops some images from both splits (no dog, multiple dogs, tiny box), so the final test fraction lands slightly under 15%. 05 loads `split == "test"` rows in exactly one function (`final_eval`), after the epoch count and thresholds are fixed.

## Debugging

Every script ends with a **DEBUGGING GUIDE** block. It lists each intermediate file, how to check it, and common failures. Visual checks go to `results/debug/`:

- `01_samples_by_class.jpg`: what each original label looks like
- `03_duplicate_groups.jpg`: what counted as a near-duplicate
- `04_boxes_overlay.jpg`, `04_crops_<label>.jpg`, `04_drops_<reason>.jpg`
- `05_train_aug_samples.jpg`, `05_test_errors.jpg`

## Planned dependencies (not installed yet)

`torch`, `torchvision`, `timm`, `ultralytics`, `datasets`, `scikit-learn`, `imagehash`, `pillow`, `numpy`, `pyyaml`, `matplotlib`, `tqdm`
