"""
05_train_classifier.py  (PSEUDOCODE, not runnable yet)

Purpose
-------
Fine-tune a small pretrained classifier (MobileNetV3 or EfficientNet-B0 via timm) on the dog
crops to predict sit / down / stand. Uses stratified K-fold CV on the trainval split to choose
the epoch count and the reward thresholds, retrains on all of trainval, then evaluates ONCE on
the held-out test split.

The metric we care about most is PRECISION ON THE REWARD DECISION: when the model says "sit"
(and the command was sit), how often is the dog actually sitting? A wrongly given treat is worse
than a missed one.

Reward rule = ARGMAX PLUS THRESHOLD: reward for commanded class t only if
    argmax(probs) == t  and  probs[t] >= threshold[t]
threshold[t] is the lowest value that reaches eval.reward_precision_target (0.85) on pooled
out-of-fold CV predictions. There is NO class weighting and NO weighted sampler: the model is
trained with plain cross-entropy, and precision is controlled only by the threshold.

Model construction, preprocessing, and predict() live in trickforTreat/classifier.py, so the
live pipeline will load the exact same model + transforms. This script owns training only.

Inputs
------
- data/crops_manifest.csv  (from 04)
- data/classes.json        (from 02; class order)
- configs/default.yaml     (classifier.*, train.*, augment.*, eval.*)

Outputs
-------
- checkpoints/fold{k}_best.pt                     best epoch per fold (by train.selection_metric)
- checkpoints/final.pt                            the model evaluated on test
      contents: state_dict, classes, backbone, input_size, normalize stats, reward thresholds,
                config snapshot, n_epochs used for the retrain
- results/metrics.json                            see build_metrics_json()
- results/confusion_fold{k}.png, results/confusion_test.png
- results/pr_vs_threshold_cv.csv                  pooled out-of-fold curve; thresholds are chosen from this
- results/pr_vs_threshold_test.csv                same curve on test (reported only, never used to choose)
      columns: class, threshold, precision, recall, false_reward_rate, n_fired
- results/train_log.csv                           fold, epoch, train_loss, val_loss, val metrics, lr

Run
---
    python scripts/05_train_classifier.py --config configs/default.yaml [--folds 0,1] [--limit N]
"""

# import argparse, json, time, copy
# import numpy as np, torch, torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
# from trickforTreat.common import load_config, set_seed, read_csv, seed_worker
# from trickforTreat.classifier import build_model, build_eval_transform, PostureClassifier
# from trickforTreat.transforms import build_train_transform


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CropDataset:  # (torch.utils.data.Dataset)
    """
    __init__(rows, transform): store rows (crop_path, label_id)
    __len__: len(rows)
    __getitem__(i): img = PIL.open(crop_path).convert("RGB"); return transform(img), label_id, image_id
    """


def build_train_transform(cfg):
    """
    (Lives in trickforTreat/transforms.py; listed here for review.)
    Compose([
      RandomResizedCrop(input_size, scale=augment.random_resized_crop_scale,
                        ratio=augment.random_resized_crop_ratio),      # light crop/scale jitter,
                                                                       # mimics detector box jitter
      RandomHorizontalFlip(p=augment.hflip_p),                         # posture is L/R symmetric
                                                                       # NO vertical flip / big rotation:
                                                                       # a flipped "down" isn't a real pose
      RandomApply([ColorJitter(brightness, contrast, saturation, hue)], p=augment.color_jitter_p),
                                                                       # window lighting changes
      RandomGrayscale(p=augment.grayscale_p),
      RandomApply([GaussianBlur(kernel, sigma)], p=augment.gaussian_blur_p),  # motion / ESP32 focus
      # TODO optional: random JPEG compression (augment.jpeg_compression_p) for MJPEG artifacts
      ToTensor(), Normalize(mean, std),
    ])
    Eval transform (in classifier.py): Resize((input_size, input_size)), ToTensor, Normalize.
    Crops from 04 are already square with head and tail inside, so there is no center crop.
    The RandomResizedCrop scale floor (0.80) is mild on purpose so it rarely cuts off the dog.
    """


def make_loaders(train_rows, val_rows, cfg, classes):
    """
    train_ds = CropDataset(train_rows, build_train_transform(cfg))
    val_ds   = CropDataset(val_rows,   build_eval_transform(cfg))
    counts = bincount(label_ids of train_rows, minlength=len(classes))     # logged only
    train_loader = DataLoader(train_ds, shuffle=True, generator=seeded, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, shuffle=False, ...)    # val_rows may be None for the final retrain
    return train_loader, val_loader, counts
    """


def make_criterion(cfg):
    """
    return CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)    # unweighted, by design
    """


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device) -> float:
    """
    model.train(); for x, y, _ in loader: forward (autocast if amp), loss, backward, step,
    scheduler.step() if per-iteration. Return mean loss.
    assert not torch.isnan(loss), "NaN loss: check lr, AMP, or a corrupt image"
    """


@torch_no_grad
def predict_all(model, loader, device) -> tuple["np.ndarray", "np.ndarray", list[str]]:
    """Returns probs [N, C] (softmax), y_true [N], image_ids. model.eval()."""


def compute_metrics(probs, y_true, classes, target_classes) -> dict:
    """
    Plain argmax classification metrics (used per epoch for checkpoint selection):
    y_pred = probs.argmax(1)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(C), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(C))
    out = {"per_class": {c: {"precision", "recall", "f1", "support"}}, "macro_f1", "accuracy",
           "confusion_matrix": cm.tolist(),
           "sit_as_down": cm[sit, down] / cm[sit].sum(), "down_as_sit": cm[down, sit] / cm[down].sum()}
    return out
    """


def reward_curve(probs, y_true, t: int, thresholds) -> list[dict]:
    """
    Argmax-plus-threshold reward decision for commanded class t ("sit" or "down"):
      y_pred = probs.argmax(1)
      for thr in thresholds:                       # np.arange(start, stop + step/2, step) from eval.threshold_sweep
          fire = (y_pred == t) & (probs[:, t] >= thr)
          precision = mean(y_true[fire] == t)            (NaN if nothing fires)
          recall    = sum(fire & (y_true == t)) / sum(y_true == t)
          false_reward_rate = sum(fire & (y_true != t)) / sum(y_true != t)
                                                   # fraction of non-t images that would earn a treat
          n_fired = sum(fire)
          rows.append({thr, precision, recall, false_reward_rate, n_fired})
    return rows
    """


def choose_threshold(curve, precision_target) -> dict:
    """
    Lowest threshold whose precision >= target (lowest = keeps the most recall).
    Require n_fired >= 30 at that threshold so precision isn't estimated from a handful of images.
    If no threshold reaches the target: return the threshold with the highest precision, plus
    reached_target = False. That's a feasibility red flag to report, not a crash.
    return {"threshold", "precision", "recall", "false_reward_rate", "reached_target"}

    NOTE: these are per-FRAME numbers. The temporal gate (out of scope) should raise effective
          precision further, so per-frame recall can be traded away somewhat.
    """


def run_fold(fold, train_rows, val_rows, cfg, classes, device) -> dict:
    """
    set_seed(cfg.seed + fold)
    model = build_model(cfg.classifier, num_classes=len(classes)).to(device)
    loaders, counts = make_loaders(...); criterion = make_criterion(cfg)
    optimizer = AdamW([{"params": head, "lr": lr_head}, {"params": backbone, "lr": lr_backbone}],
                      weight_decay=...)
    scheduler = cosine with warmup
    best, best_epoch, patience = -inf, -1, 0
    for epoch in range(train.epochs):
        set backbone.requires_grad = (epoch >= classifier.freeze_backbone_epochs)
        tr_loss = train_one_epoch(...)
        probs, y, _ = predict_all(model, val_loader, device)
        m = compute_metrics(...); val_loss = criterion(probs/logits, y)
        append a row to train_log.csv
        score = m[train.selection_metric]          # macro_f1
        if score > best: best = score; best_epoch = epoch; save checkpoints/fold{fold}_best.pt; patience = 0
        else: patience += 1; if patience >= early_stopping_patience: break
    reload best checkpoint, recompute val probs + metrics, save confusion_fold{fold}.png
    return {"fold": fold, "best_epoch": best_epoch, "val_metrics": m, "val_probs": probs, "val_y": y,
            "n_train", "n_val", "class_counts"}
    """


def cross_validate(trainval_rows, cfg, classes, device) -> list[dict]:
    """
    skf = StratifiedKFold(n_splits=train.k_folds, shuffle=True, random_state=cfg.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(rows, labels)):
        if --folds given and fold not in it: continue
        assert set(tr ids).isdisjoint(va ids)
        assert no row has split == "test"               # the test set NEVER enters CV
        results.append(run_fold(...))
    Summarize: mean +/- std across folds of each per-class precision/recall/F1 and sit<->down confusion.
    Reward thresholds (the key output of CV):
        oof_probs = concat(r["val_probs"] for r in results); oof_y = concat(r["val_y"])
        assert len(oof_y) == len(trainval_rows)           # every trainval image predicted exactly once
        for t in target_classes:
            curve = reward_curve(oof_probs, oof_y, classes.index(t), sweep)
            thresholds[t] = choose_threshold(curve, eval.reward_precision_target)
        save results/pr_vs_threshold_cv.csv
    Pooled out-of-fold predictions give a threshold choice that never touches test.
    return results, thresholds
    """


def train_final_model(trainval_rows, cfg, classes, device, cv_results) -> "nn.Module":
    """
    Retrain from the pretrained backbone on ALL trainval rows:
        epochs = round(mean(best_epoch over folds)) + 1     # best_epoch is 0-indexed
        same optimizer/scheduler/freeze schedule as run_fold, with scheduler length = epochs
        no validation, no early stopping
    return model
    NOTE: the CV thresholds were fit on fold models, not this one. Retraining on ~25% more data
          shifts probabilities slightly; test results at those thresholds tell us if that matters.
    """


def final_eval(model, test_rows, cfg, classes, thresholds_from_cv, device) -> dict:
    """
    THE ONLY PLACE test rows are loaded.
    probs, y, ids = predict_all(model, test_loader, device)
    m = compute_metrics(...)
    for t in target_classes:
        curve = reward_curve(probs, y, t, sweep)                     # full curve, for the writeup
        m["reward"][t] = row of curve at thresholds_from_cv[t]       # NOT re-tuned on test
    save confusion_test.png, pr_vs_threshold_test.csv,
         results/test_errors.csv (image_id, true, pred, prob) for the misclassification grid
    return m
    """


def measure_latency(model, cfg, device) -> dict:
    """
    Per-stage inference latency from CROP to PREDICTION, batch size 1 (as in the live loop):
      for dev in eval.latency_devices (+ cuda/mps if available):
        clf = PostureClassifier(model, transform, classes, device=dev)
        take ~50 real test crops as PIL images
        warmup eval.latency_warmup_iters
        for eval.latency_timed_iters:
            t0 = perf_counter()
            x = transform(img)              # stage: preprocess (resize/normalize)
            t1 = perf_counter()
            logits = model(x[None].to(dev)); sync if cuda   # stage: forward
            t2 = perf_counter()
            probs = softmax(logits).cpu()   # stage: postprocess
            t3 = perf_counter()
        report mean / p50 / p95 per stage + total, in ms. Also param count and checkpoint MB.
    The detector's latency comes from 04's detection_report.json. Combine both in metrics.json
    under "latency" for a first frame->prediction estimate (minus network/MJPEG decode).
    """


def build_metrics_json(cfg, classes, cv_results, test_metrics, latency, extra) -> dict:
    """
    {
      "config": full config snapshot, "classes": classes, "seed": seed,
      "data": {"n_trainval", "n_test", "class_counts",
               "duplicates_dropped": from 03, "crop_drops": from detection_report.json},
      "cv": {"folds": [...], "mean": {...}, "std": {...}, "reward_thresholds": {...}},
      "test": test_metrics,
      "latency": {"classifier": latency, "detector": from detection_report.json},
      "checkpoint": "checkpoints/final.pt",
      "timestamp", "torch_version", "timm_version", "device"
    }
    """


def sanity_checks(cv_results, test_metrics, classes, cfg) -> None:
    """
    assert len(cv_results) == k_folds (unless --folds subset given)
    assert every fold's val confusion matrix sums to n_val; test cm sums to n_test
    assert test metrics computed on rows with split == "test" only, all of whose ids are in data/test_ids.txt
    assert final.pt stores the CV thresholds, and should_reward() matches reward_curve's `fire` on test
    assert final.pt reloads via PostureClassifier.load() and reproduces test probs on 5 images
           (atol 1e-5): catches transform/class-order mismatches between training and runtime
    assert classes in checkpoint == classes.json
    warn if test macro F1 is far below CV mean (> 0.05): possible leakage or drift
    warn if any fold's metric is far from the others: unstable or bad split
    feasibility bar (warn, don't fail): sit and down test precision >= eval.reward_precision_target
        at the CV thresholds, and sit_as_down, down_as_sit < eval.max_sit_down_confusion.
        If not, flag it in the report.
    """


def main():
    """
    args, cfg = ...; set_seed(cfg.seed); torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False; device = pick_device(cfg)
    classes = json.load("data/classes.json")["classes"]
    rows = read_csv(cfg.paths.crops_manifest)
    trainval = [r for r in rows if r["split"] == "trainval"]; test = [... == "test"]
    cv, thresholds = cross_validate(trainval, cfg, classes, device)
    model = train_final_model(trainval, cfg, classes, device, cv)
    save checkpoints/final.pt (via PostureClassifier.save, including thresholds)
    test_m = final_eval(model, test, cfg, classes, thresholds, device)
    lat = measure_latency(model, cfg, device)
    json.dump(build_metrics_json(...), open(cfg.paths.metrics_json, "w"), indent=2)
    print a short summary table (per-class P/R/F1 test, reward precision, latency p50)
    sanity_checks(...)
    debug grid: results/debug/05_test_errors.jpg (worst 32 confident mistakes, captioned
        "true -> pred (prob)"), results/debug/05_train_aug_samples.jpg (augmented train batch)
    """


# if __name__ == "__main__":
#     main()


# ---------------------------------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------------------------------
# Intermediates:
#   results/train_log.csv          Loss/metrics per epoch per fold. Plot train vs val loss for over/underfitting.
#   checkpoints/fold{k}_best.pt    Per-fold best weights (by selection_metric).
#   checkpoints/final.pt           Model + classes + transforms config. Load with PostureClassifier.load().
#   results/confusion_*.png        Look at the sit<->down cells first. That's the feasibility question.
#   results/pr_vs_threshold_cv.csv     How much recall you give up to reach the precision target.
#                                      The chosen threshold per class is the first row with precision >= 0.85.
#   results/pr_vs_threshold_test.csv   Same on test. If test precision at the CV threshold is well below
#                                      0.85, the threshold didn't transfer (see train_final_model NOTE).
#   results/test_errors.csv + results/debug/05_test_errors.jpg   Which images fool the model.
#   results/debug/05_train_aug_samples.jpg   Make sure augmentations look like plausible room frames,
#                                            not destroyed images.
#   results/metrics.json           Everything for the writeup.
#
# Workflow:
#   1. Smoke test: --limit 200 --folds 0 with epochs 1. Should finish in minutes on CPU and produce
#      every output file. Then run the full thing.
#   2. If val accuracy is stuck near the majority-class rate (~50% = "down"), the backbone is frozen
#      too long, the LR is wrong, or labels are shuffled. Check 05_train_aug_samples.jpg captions.
#   3. If train loss -> 0 and val loss climbs: overfitting. More aug, fewer epochs, or a smaller model.
#   4. If test << CV: suspect near-duplicate leakage within trainval folds (inflated CV) or a split bug.
#   5. Confident mistakes on "sit" that are actually ambiguous/mislabeled images are worth counting.
#      DogPoseCV labels are not perfect, which caps achievable precision.
