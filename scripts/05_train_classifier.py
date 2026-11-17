#!/usr/bin/env python3
"""
05_train_classifier.py

Fine-tune a pretrained timm classifier (default MobileNetV3-Large) on the dog crops to predict
sit / down / stand.

  1. Stratified K-fold CV on trainval. Each fold trains with early stopping on val macro F1 and
     gives (a) its best epoch count and (b) out-of-fold (OOF) probabilities for its val images.
  2. Reward thresholds from the pooled OOF predictions. Reward rule = argmax plus threshold:
     reward commanded class t only if argmax == t and p(t) >= threshold[t]. threshold[t] is the
     lowest value reaching eval.reward_precision_target. No class weighting anywhere.
  3. Final model: retrain on ALL trainval for round(mean best epoch) + 1 epochs.
  4. Save checkpoints/final.pt (model + classes + preprocessing + thresholds), reload it through
     PostureClassifier (the runtime path), and evaluate ONCE on the held-out test crops.
  5. Per-stage latency (crop -> prediction) at batch size 1 on CPU.

Inputs:  data/crops_manifest.csv (04), data/classes.json (02), configs/default.yaml
Outputs:
  checkpoints/fold{k}_best.pt, checkpoints/final.pt
  results/metrics.json                 everything for the writeup
  results/train_log.csv                one row per (fold, epoch); updated live while training
  results/pr_vs_threshold_{cv,test}.csv
  results/test_predictions.csv         every test crop: true, predicted, probabilities
  results/debug/05_*                   training curves, confusion matrices, reward curves,
                                       augmented samples, confident mistakes, false rewards

Run:
  python scripts/05_train_classifier.py                       # full run (CPU: a few hours)
  python scripts/05_train_classifier.py --limit 400 --epochs 1 --folds 0,1   # smoke test, ~2 min
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from trickforTreat.classifier import (  # noqa: E402
    PostureClassifier, backbone_parameters, build_model, head_parameters, set_backbone_trainable,
)
from trickforTreat.common import (  # noqa: E402
    add_common_args, header, load_config, pick_device, print_table, read_csv, rel, resolve, save_contact_sheet,
    save_json, seed_worker, set_seed, setup_logging, write_csv,
)
from trickforTreat.metrics import (  # noqa: E402
    choose_threshold, classification_metrics, plot_confusion, plot_reward_curves, plot_training_curves, reward_curve,
    row_at, threshold_grid,
)
from trickforTreat.transforms import build_eval_transform, build_train_transform, denormalize  # noqa: E402

LOG_FIELDS = ["fold", "epoch", "frozen", "train_loss", "val_loss", "val_macro_f1", "val_accuracy",
              "val_sit_as_down", "val_down_as_sit", "lr_head", "lr_backbone", "seconds"]
CURVE_FIELDS = ["class", "threshold", "precision", "recall", "false_reward_rate", "n_fired"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cache(rows, size: int) -> np.ndarray:
    """Every crop decoded once and resized to size x size (crops are square). ~1 GB for ~12k crops."""
    arr = np.empty((len(rows), size, size, 3), dtype=np.uint8)
    for i, r in enumerate(tqdm(rows, desc=f"cache crops @{size}px", unit="img")):
        with Image.open(resolve(r["crop_path"])) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
    return arr


class CropDataset(Dataset):
    def __init__(self, indices, labels: np.ndarray, transform, cache: np.ndarray | None, paths: list[str]):
        self.indices, self.labels, self.transform, self.cache, self.paths = list(indices), labels, transform, cache, paths

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = self.indices[k]
        if self.cache is not None:
            img = Image.fromarray(self.cache[i])
        else:
            with Image.open(resolve(self.paths[i])) as im:
                img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def train_model(cfg, classes, data, train_idx, val_idx, device, epochs: int, tag: str, log_rows: list,
                log) -> dict:
    """Train one model. With val_idx: early stopping on val macro F1, best weights restored.
    Returns {"model", "best_epoch", "val_probs", "val_y"}."""
    tc, cc = cfg.train, cfg.classifier
    set_seed(cfg.seed + (int(tag) if str(tag).isdigit() else 1000))
    model = build_model(cc.backbone, len(classes), bool(cc.pretrained), float(cc.dropout)).to(device)
    mean, std, size = cc.normalize_mean, cc.normalize_std, int(cc.input_size)
    train_tf = build_train_transform(cfg.augment, size, mean, std)
    eval_tf = build_eval_transform(size, mean, std)

    g = torch.Generator().manual_seed(cfg.seed)
    nw = int(tc.num_workers)
    train_loader = DataLoader(CropDataset(train_idx, data["labels"], train_tf, data["cache"], data["paths"]),
                              batch_size=int(tc.batch_size), shuffle=True, drop_last=len(train_idx) > int(tc.batch_size),
                              num_workers=nw, worker_init_fn=seed_worker, generator=g, persistent_workers=nw > 0)
    val_loader = None
    if val_idx is not None:
        val_loader = DataLoader(CropDataset(val_idx, data["labels"], eval_tf, data["cache"], data["paths"]),
                                batch_size=128, shuffle=False, num_workers=nw, persistent_workers=nw > 0)

    opt = torch.optim.AdamW([{"params": head_parameters(model), "lr": float(tc.lr_head)},
                             {"params": backbone_parameters(model), "lr": float(tc.lr_backbone)}],
                            weight_decay=float(tc.weight_decay))
    steps = len(train_loader)
    total, warm = epochs * steps, max(1, int(float(tc.warmup_epochs) * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    freeze = int(cc.freeze_backbone_epochs)

    best = {"score": -1.0, "epoch": -1, "state": None, "probs": None, "y": None}
    patience = 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        frozen = epoch < freeze
        set_backbone_trainable(model, not frozen)
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"[{tag}] epoch {epoch + 1}/{epochs}{' (head only)' if frozen else ''}",
                         leave=False, unit="batch"):
            loss = F.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=float(tc.label_smoothing))
            assert torch.isfinite(loss), "non-finite loss: check the learning rate or a corrupt image"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        row = {"fold": tag, "epoch": epoch + 1, "frozen": frozen, "train_loss": round(float(np.mean(losses)), 4),
               "val_loss": None, "val_macro_f1": None, "val_accuracy": None, "val_sit_as_down": None,
               "val_down_as_sit": None, "lr_head": opt.param_groups[0]["lr"], "lr_backbone": opt.param_groups[1]["lr"]}
        msg = f"[{tag}] epoch {epoch + 1:2d}/{epochs}  train_loss {row['train_loss']:.4f}"
        if val_loader is not None:
            logits, yv = predict_logits(model, val_loader, device)
            probs = torch.softmax(logits, 1).numpy()
            m = classification_metrics(probs, yv.numpy(), classes)
            row.update(val_loss=round(F.cross_entropy(logits, yv).item(), 4), val_macro_f1=round(m["macro_f1"], 4),
                       val_accuracy=round(m["accuracy"], 4), val_sit_as_down=round(m.get("sit_as_down", 0), 4),
                       val_down_as_sit=round(m.get("down_as_sit", 0), 4))
            msg += (f"  val_loss {row['val_loss']:.4f}  macro_f1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}"
                    f"  sit->down {m.get('sit_as_down', 0):.3f}  down->sit {m.get('down_as_sit', 0):.3f}")
            if m["macro_f1"] > best["score"]:
                best.update(score=m["macro_f1"], epoch=epoch + 1, state=copy.deepcopy(model.state_dict()),
                            probs=probs, y=yv.numpy())
                patience = 0
                msg += "  *best*"
            else:
                patience += 1
        row["seconds"] = round(time.perf_counter() - t0, 1)
        log_rows.append(row)
        write_csv(log_rows, cfg.paths.results_dir + "/train_log.csv", LOG_FIELDS)
        log.info("%s  (%.0fs)", msg, row["seconds"])
        if val_loader is not None and patience >= int(tc.early_stopping_patience):
            log.info("[%s] early stop: no val improvement for %d epochs", tag, patience)
            break
    if val_loader is not None:
        model.load_state_dict(best["state"])
    return {"model": model.eval(), "best_epoch": best["epoch"] if val_loader is not None else epochs,
            "val_probs": best["probs"], "val_y": best["y"]}


def make_classifier(model, cfg, classes, thresholds=None, device="cpu") -> PostureClassifier:
    cc = cfg.classifier
    return PostureClassifier(model, classes, cc.backbone, int(cc.input_size), cc.normalize_mean, cc.normalize_std,
                             thresholds, device)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def measure_latency(clf: PostureClassifier, crops: list[Image.Image], cfg) -> dict:
    for c in crops[: int(cfg.eval.latency_warmup_iters)]:
        clf.predict(c)
    stages = {"pre": [], "fwd": [], "post": []}
    for i in range(int(cfg.eval.latency_timed_iters)):
        p = clf.predict(crops[i % len(crops)])
        for k in stages:
            stages[k].append(p.latency_ms[k])
    total = np.sum([stages[k] for k in stages], axis=0)
    out = {k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
           for k, v in {**stages, "total": total}.items()}
    out["device"], out["threads"] = clf.device, torch.get_num_threads()
    out["params_millions"] = sum(p.numel() for p in clf.model.parameters()) / 1e6
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="K-fold CV, thresholds, final retrain, test evaluation.")
    add_common_args(p)
    p.add_argument("--folds", default=None, help="comma-separated fold indices to run (default: all)")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests)")
    p.add_argument("--reuse-folds", action="store_true",
                   help="load existing checkpoints/fold{k}_best.pt instead of retraining that fold (resume after a crash)")
    args = p.parse_args()
    cfg = load_config(args.config, limit=args.limit)
    log = setup_logging("05_train", cfg.debug.log_level)
    set_seed(cfg.seed)
    torch.set_num_threads(int(cfg.train.torch_threads))
    device = pick_device(cfg.detector.device)
    epochs = args.epochs or int(cfg.train.epochs)
    limited = bool(cfg.debug.limit_images)
    dbg, res_dir = resolve(cfg.debug.debug_dir), resolve(cfg.paths.results_dir)
    ckpt_dir = resolve(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------
    header("1. Data")
    import json
    with open(resolve(cfg.paths.classes_json)) as f:
        cj = json.load(f)
    classes, targets = cj["classes"], cj["target_classes"]
    rows = read_csv(cfg.paths.crops_manifest)
    if limited:
        rows = random.Random(cfg.seed).sample(rows, min(cfg.debug.limit_images, len(rows)))
    rows.sort(key=lambda r: r["image_id"])
    for r in rows:
        assert classes[int(r["label_id"])] == r["label"], f"label_id/classes.json mismatch for {r['image_id']}"
    trainval = [r for r in rows if r["split"] == "trainval"]
    test = [r for r in rows if r["split"] == "test"]
    test_ids_file = set(resolve(cfg.paths.test_ids).read_text().split())
    assert {r["image_id"] for r in test} <= test_ids_file, "a test crop is not in data/test_ids.txt"
    assert not ({r["image_id"] for r in trainval} & test_ids_file), "a trainval crop is in the test set"
    tvc, tec = Counter(r["label"] for r in trainval), Counter(r["label"] for r in test)
    print_table([(c, tvc[c], tec[c]) for c in classes], ["class", "trainval", "test"])
    log.info("device %s, %d torch threads, input %dpx, backbone %s, %d epochs max",
             device, torch.get_num_threads(), int(cfg.classifier.input_size), cfg.classifier.backbone, epochs)

    labels = np.array([int(r["label_id"]) for r in trainval])
    data = {"labels": labels, "paths": [r["crop_path"] for r in trainval],
            "cache": load_cache(trainval, int(cfg.train.cache_size)) if cfg.train.cache_images else None}

    if cfg.debug.save_debug_grid:
        tf = build_train_transform(cfg.augment, int(cfg.classifier.input_size), cfg.classifier.normalize_mean,
                                   cfg.classifier.normalize_std)
        ds = CropDataset(range(len(trainval)), labels, tf, data["cache"], data["paths"])
        items = []
        for i in random.Random(cfg.seed).sample(range(len(ds)), min(32, len(ds))):
            x, y = ds[i]
            items.append((denormalize(x, cfg.classifier.normalize_mean, cfg.classifier.normalize_std),
                          f"{classes[y]}\n{trainval[i]['image_id']}"))
        log.info("wrote %s", rel(save_contact_sheet(items, dbg / "05_train_aug_samples.jpg", cols=8,
                                                    title="Augmented training inputs, exactly as the model sees them")))

    # ---- cross-validation --------------------------------------------------
    header("2. Stratified K-fold cross-validation")
    k = int(cfg.train.k_folds)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(k))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    oof = np.full((len(trainval), len(classes)), np.nan, dtype=np.float32)
    log_rows, cv = [], []
    if args.reuse_folds and (res_dir / "train_log.csv").exists():
        num = lambda v: None if v in ("", "None") else (v == "True" if v in ("True", "False") else float(v))
        log_rows = [{k: (v if k == "fold" else num(v)) for k, v in r.items()}
                    for r in read_csv(res_dir / "train_log.csv") if r["fold"].isdigit()]
    t_cv = time.perf_counter()
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if fold not in folds:
            continue
        assert set(tr).isdisjoint(va)
        log.info("fold %d/%d: %d train, %d val", fold, k, len(tr), len(va))
        ckpt_path = ckpt_dir / f"fold{fold}_best.pt"
        if args.reuse_folds and ckpt_path.exists():
            prev = PostureClassifier.load(ckpt_path, device)
            assert prev.classes == classes, f"{ckpt_path} has classes {prev.classes}"
            vl = DataLoader(CropDataset(va, labels, build_eval_transform(int(cfg.classifier.input_size),
                            cfg.classifier.normalize_mean, cfg.classifier.normalize_std), data["cache"], data["paths"]),
                            batch_size=128, shuffle=False, num_workers=int(cfg.train.num_workers))
            logits, yv = predict_logits(prev.model, vl, device)
            out = {"model": prev.model, "best_epoch": int(prev.extra["best_epoch"]),
                   "val_probs": torch.softmax(logits, 1).numpy(), "val_y": yv.numpy()}
            log.info("fold %d: reused %s (best epoch %d)", fold, rel(ckpt_path), out["best_epoch"])
        else:
            out = train_model(cfg, classes, data, tr, va, device, epochs, str(fold), log_rows, log)
        assert (out["val_y"] == labels[va]).all(), "val order changed"
        oof[va] = out["val_probs"]
        m = classification_metrics(out["val_probs"], out["val_y"], classes)
        make_classifier(out["model"], cfg, classes, device=device).save(ckpt_dir / f"fold{fold}_best.pt",
                                                                          {"fold": fold, "best_epoch": out["best_epoch"]})
        plot_confusion(m["confusion_matrix"], classes, dbg / f"05_confusion_fold{fold}.png",
                       f"Fold {fold} validation (best epoch {out['best_epoch']})")
        plot_training_curves(log_rows, dbg / "05_training_curves.png", "Cross-validation training curves")
        cv.append({"fold": fold, "best_epoch": out["best_epoch"], "n_train": len(tr), "n_val": len(va), "metrics": m})
        log.info("fold %d done: best epoch %d, macro F1 %.4f (%.0f min elapsed)", fold, out["best_epoch"],
                 m["macro_f1"], (time.perf_counter() - t_cv) / 60)

    header("3. CV summary")
    def ms(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
    table = []
    for c in classes:
        table.append([c] + [ms([f["metrics"]["per_class"][c][x] for f in cv]) for x in ("precision", "recall", "f1")])
    print_table(table, ["class", "precision", "recall", "f1"])
    print(f"\n  macro F1 {ms([f['metrics']['macro_f1'] for f in cv])}   accuracy {ms([f['metrics']['accuracy'] for f in cv])}"
          f"   sit->down {ms([f['metrics']['sit_as_down'] for f in cv])}   down->sit {ms([f['metrics']['down_as_sit'] for f in cv])}")
    print(f"  best epochs per fold: {[f['best_epoch'] for f in cv]}")

    header("4. Reward thresholds from pooled out-of-fold predictions")
    have = ~np.isnan(oof).any(1)
    oof_p, oof_y = oof[have], labels[have]
    if len(folds) == k:
        assert have.all(), "not every trainval image got an out-of-fold prediction"
    grid = threshold_grid(cfg.eval.threshold_sweep)
    target = float(cfg.eval.reward_precision_target)
    min_fired = int(cfg.eval.min_fired) if not limited else 3
    cv_curves, thresholds, chosen = {}, {}, {}
    for t in targets:
        cv_curves[t] = reward_curve(oof_p, oof_y, classes.index(t), grid)
        chosen[t] = choose_threshold(cv_curves[t], target, min_fired)
        thresholds[t] = chosen[t]["threshold"]
    print_table([(t, chosen[t]["threshold"], chosen[t]["precision"], chosen[t]["recall"],
                  chosen[t]["false_reward_rate"], "yes" if chosen[t]["reached_target"] else "NO") for t in targets],
                ["commanded", "threshold", "precision", "recall", "false reward rate", f"reached {target}"])
    write_csv([{"class": t, **r} for t in targets for r in cv_curves[t]], res_dir / "pr_vs_threshold_cv.csv", CURVE_FIELDS)
    plot_reward_curves(cv_curves, chosen, dbg / "05_reward_curves_cv.png",
                       "Reward precision/recall vs. threshold (pooled out-of-fold, trainval)", target)
    for t in targets:
        if not chosen[t]["reached_target"]:
            log.warning("'%s' never reaches %.2f precision on CV; using its most precise threshold %.2f",
                        t, target, thresholds[t])

    # ---- final model -------------------------------------------------------
    header("5. Final model: retrain on all trainval")
    final_epochs = int(round(np.mean([f["best_epoch"] for f in cv]))) + 1
    log.info("retraining for %d epochs (mean best epoch %.2f + 1) on %d images",
             final_epochs, np.mean([f["best_epoch"] for f in cv]), len(trainval))
    final = train_model(cfg, classes, data, np.arange(len(trainval)), None, device, final_epochs, "final", log_rows, log)
    final_path = make_classifier(final["model"], cfg, classes, thresholds, device).save(
        ckpt_dir / "final.pt", {"final_epochs": final_epochs, "config": cfg.to_dict(), "cv_thresholds": chosen})
    log.info("saved %s (%.1f MB)", rel(final_path), final_path.stat().st_size / 1e6)
    data["cache"] = None  # free ~1 GB before test evaluation

    # ---- test --------------------------------------------------------------
    header("6. Held-out test evaluation (runtime path: PostureClassifier.load -> native crops)")
    clf = PostureClassifier.load(final_path, device)
    assert clf.classes == classes and clf.reward_thresholds == thresholds
    test_imgs = []
    for r in tqdm(test, desc="load test crops", unit="img"):
        with Image.open(resolve(r["crop_path"])) as im:
            test_imgs.append(im.convert("RGB"))
    y_test = np.array([int(r["label_id"]) for r in test])
    probs = clf.predict_probs(test_imgs)
    tm = classification_metrics(probs, y_test, classes)
    print_table([(c, tm["per_class"][c]["precision"], tm["per_class"][c]["recall"], tm["per_class"][c]["f1"],
                  tm["per_class"][c]["support"]) for c in classes], ["class", "precision", "recall", "f1", "support"])
    print(f"\n  macro F1 {tm['macro_f1']:.4f}   accuracy {tm['accuracy']:.4f}   "
          f"sit->down {tm['sit_as_down']:.3f}   down->sit {tm['down_as_sit']:.3f}")
    test_curves, test_reward = {}, {}
    for t in targets:
        test_curves[t] = reward_curve(probs, y_test, classes.index(t), grid)
        test_reward[t] = row_at(test_curves[t], thresholds[t])
    print()
    print_table([(t, thresholds[t], test_reward[t]["precision"], test_reward[t]["recall"],
                  test_reward[t]["false_reward_rate"], test_reward[t]["n_fired"]) for t in targets],
                ["commanded", "CV threshold", "test precision", "test recall", "false reward rate", "rewards"])
    tm["reward_at_cv_threshold"] = test_reward
    write_csv([{"class": t, **r} for t in targets for r in test_curves[t]], res_dir / "pr_vs_threshold_test.csv",
              CURVE_FIELDS)
    plot_confusion(tm["confusion_matrix"], classes, dbg / "05_confusion_test.png", "Held-out test set")
    plot_reward_curves(test_curves, {t: test_reward[t] for t in targets}, dbg / "05_reward_curves_test.png",
                       "Test reward precision/recall (dashed line = threshold chosen on CV)", target)
    pred = probs.argmax(1)
    pred_rows = [{"image_id": r["image_id"], "crop_path": r["crop_path"], "true": r["label"], "pred": classes[pred[i]],
                  **{f"p_{c}": round(float(probs[i, j]), 4) for j, c in enumerate(classes)}} for i, r in enumerate(test)]
    write_csv(pred_rows, res_dir / "test_predictions.csv",
              ["image_id", "crop_path", "true", "pred"] + [f"p_{c}" for c in classes])

    if cfg.debug.save_debug_grid:
        wrong = np.nonzero(pred != y_test)[0]
        wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:32]
        if len(wrong):
            save_contact_sheet([(test_imgs[i], f"true {classes[y_test[i]]}\npred {classes[pred[i]]} "
                                               f"{probs[i, pred[i]]:.2f}\n{test[i]['image_id']}") for i in wrong],
                               dbg / "05_test_confident_mistakes.jpg", cols=8,
                               title="Most confident test mistakes (dataset label, model prediction + probability)")
        fr = []
        for t in targets:
            ti = classes.index(t)
            fire = (pred == ti) & (probs[:, ti] >= thresholds[t]) & (y_test != ti)
            fr += [(test_imgs[i], f"cmd {t}: REWARDED\nlabel {classes[y_test[i]]}, model {t} {probs[i, ti]:.2f}\n{test[i]['image_id']}")
                   for i in np.nonzero(fire)[0]]
        if fr:
            save_contact_sheet(fr[:48], dbg / "05_test_false_rewards.jpg", cols=8,
                               title=f"Test images that would WRONGLY earn a treat at the CV thresholds ({len(fr)} total)")

    # ---- latency -----------------------------------------------------------
    header("7. Latency: crop -> prediction, batch 1")
    lat = measure_latency(clf, test_imgs[:50], cfg)
    print_table([(s, lat[s]["mean"], lat[s]["p50"], lat[s]["p95"]) for s in ("pre", "fwd", "post", "total")],
                ["stage (ms)", "mean", "p50", "p95"])
    det_report_path = res_dir / "detection_report.json"
    det_report = json.loads(det_report_path.read_text()) if det_report_path.exists() else {}
    det_ms = det_report.get("latency_ms_per_image", {})
    yolo_ms = sum(det_ms.get(f"yolo_{k}", 0) for k in ("preprocess", "inference", "postprocess"))
    print(f"\n  detector (from 04, batched): ~{yolo_ms:.1f} ms/image  ->  frame-to-prediction estimate "
          f"~{yolo_ms + lat['total']['mean']:.0f} ms on this CPU (excluding MJPEG decode and network)")

    # ---- metrics.json ------------------------------------------------------
    metrics = {
        "classes": classes, "target_classes": targets, "seed": cfg.seed,
        "data": {"n_trainval": len(trainval), "n_test": len(test), "trainval_counts": dict(tvc),
                 "test_counts": dict(tec), "detector_drop_rate": det_report.get("drop_rate"), "limited": limited},
        "cv": {"k": k, "folds_run": folds, "folds": cv,
               "mean": {"macro_f1": float(np.mean([f["metrics"]["macro_f1"] for f in cv])),
                        "accuracy": float(np.mean([f["metrics"]["accuracy"] for f in cv]))},
               "std": {"macro_f1": float(np.std([f["metrics"]["macro_f1"] for f in cv]))},
               "reward_thresholds": chosen},
        "final": {"epochs": final_epochs, "checkpoint": rel(final_path)},
        "test": tm,
        "latency": {"classifier_batch1_ms": lat, "detector_ms": det_ms},
        "config": cfg.to_dict(),
        "versions": {"torch": torch.__version__, "timm": __import__("timm").__version__},
    }
    save_json(metrics, cfg.paths.metrics_json)
    log.info("wrote %s", cfg.paths.metrics_json)

    # ---- sanity ------------------------------------------------------------
    header("8. Sanity checks")
    for f in cv:
        assert np.sum(f["metrics"]["confusion_matrix"]) == f["n_val"]
    assert np.sum(tm["confusion_matrix"]) == len(test)
    print("  [ok] every confusion matrix sums to its split size")
    clf2 = PostureClassifier.load(final_path, "cpu")
    p2 = clf2.predict_probs(test_imgs[:8])
    assert np.allclose(p2, probs[:8], atol=1e-4), "reloaded checkpoint does not reproduce test probabilities"
    print("  [ok] final.pt reloads and reproduces test probabilities")
    for t in targets:
        ti = classes.index(t)
        fire = (pred == ti) & (probs[:, ti] >= thresholds[t])
        rewards = [clf.should_reward(clf._to_prediction(pp), t) for pp in probs]
        assert np.array_equal(fire, np.array(rewards)), f"should_reward disagrees with the metric for {t}"
    print("  [ok] PostureClassifier.should_reward matches the reward metric exactly")
    cv_f1 = metrics["cv"]["mean"]["macro_f1"]
    if tm["macro_f1"] < cv_f1 - 0.05:
        log.warning("test macro F1 %.3f is well below CV %.3f: leakage or drift?", tm["macro_f1"], cv_f1)
    else:
        print(f"  [ok] test macro F1 {tm['macro_f1']:.3f} is consistent with CV {cv_f1:.3f}")
    bar_ok = all(test_reward[t]["precision"] >= target for t in targets) and \
        tm["sit_as_down"] < float(cfg.eval.max_sit_down_confusion) and tm["down_as_sit"] < float(cfg.eval.max_sit_down_confusion)
    print(f"\n  FEASIBILITY BAR (test precision >= {target} for {targets} at CV thresholds, sit<->down "
          f"< {cfg.eval.max_sit_down_confusion}): {'PASSED' if bar_ok else 'NOT MET'}")
    print("\n  Done. Try the full pipeline: python scripts/06_run_pipeline.py --source data/split_manifest.csv --split test")


if __name__ == "__main__":
    main()

# DEBUGGING GUIDE
#   results/train_log.csv / 05_training_curves.png  live during training. Val loss rising while train
#       loss falls = overfitting; macro F1 stuck near 0.33 = labels or LR broken.
#   05_train_aug_samples.jpg   augmentations should look like plausible room frames, not destroyed images.
#   05_confusion_*.png         read the sit/down cells first: that's the feasibility question.
#   05_reward_curves_*.png     how much recall it costs to reach the precision target.
#   05_test_false_rewards.jpg  every test image that would wrongly get a treat. Mislabeled ones cap precision.
#   checkpoints/final.pt       load with trickforTreat.classifier.PostureClassifier.load(path).
