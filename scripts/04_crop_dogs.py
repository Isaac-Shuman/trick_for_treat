#!/usr/bin/env python3
"""
04_crop_dogs.py

Run the YOLO dog detector on every image in the split manifest and turn each into a square dog
crop, exactly as the live pipeline will (trickforTreat.detector.select_and_crop):
  - no dog box >= conf_threshold   -> dropped ("no_dog", or "below_conf" if YOLO saw a weaker dog)
  - more than one dog box           -> dropped ("multi_dog"): the label is ambiguous
  - dog box under min_box_side_px   -> dropped ("too_small")
  - otherwise: square crop around the padded box, gray fill past the image edge

Inputs:  data/split_manifest.csv (03), configs/default.yaml (detector.*, crop.*)
Outputs:
  data/crops/<label>/<image_id>.jpg   square crops at native resolution
  data/crops_manifest.csv             one row per crop (boxes, confidence, split, label)
  data/crop_drops.csv                 one row per dropped image, with the reason
  results/detection_report.json       drop rates (overall / per class / per split / per reason), latency
  results/debug/04_*                  overlays, crop grids, drop grids, charts

Run:  python scripts/04_crop_dogs.py   [--limit N]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from trickforTreat.common import (  # noqa: E402
    add_common_args, header, histogram, load_config, print_table, read_csv, rel, resolve, safe_filename,
    save_contact_sheet, save_json, set_seed, setup_logging, write_csv,
)
from trickforTreat.detector import YoloDogDetector, draw_detections, select_and_crop  # noqa: E402
from trickforTreat.metrics import plot_hbar  # noqa: E402

CROP_FIELDS = ["image_id", "crop_path", "src_path", "label", "label_id", "split", "breed", "det_conf", "det_class", "n_dogs",
               "bx1", "by1", "bx2", "by2", "sx1", "sy1", "sx2", "sy2", "crop_side", "fill_fraction"]
DROP_FIELDS = ["image_id", "src_path", "label", "split", "reason", "n_dogs", "max_conf"]
REASONS = ["no_dog", "below_conf", "multi_dog", "too_small", "read_error"]


def main() -> None:
    p = argparse.ArgumentParser(description="Detect dogs and save square crops.")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, limit=args.limit)
    log = setup_logging("04_crop", cfg.debug.log_level)
    set_seed(cfg.seed)
    dbg = resolve(cfg.debug.debug_dir)

    rows = read_csv(cfg.paths.split_manifest)
    if cfg.debug.limit_images:
        rows = random.Random(cfg.seed).sample(rows, min(cfg.debug.limit_images, len(rows)))
    rows.sort(key=lambda r: r["image_id"])
    crops_dir = resolve(cfg.paths.crops_dir)
    for c in {r["label"] for r in rows}:
        (crops_dir / safe_filename(c)).mkdir(parents=True, exist_ok=True)

    header("1. Load detector")
    det = YoloDogDetector.from_config(cfg.detector)
    det.warmup()
    log.info("weights %s on %s, conf >= %.2f, imgsz %d", rel(det.weights_path), det.device, det.conf, det.imgsz)

    header("2. Detect + crop")
    crop_rows, drop_rows = [], []
    drop_dets, ok_dets = {}, {}
    per_img_ms, model_ms = [], defaultdict(list)
    bs = int(cfg.detector.batch_size)
    for start in tqdm(range(0, len(rows), bs), desc="detect", unit="batch"):
        batch = rows[start:start + bs]
        imgs, good = [], []
        for r in batch:
            try:
                with Image.open(resolve(r["path"])) as im:
                    imgs.append(im.convert("RGB"))
                good.append(r)
            except Exception as e:  # noqa: BLE001
                drop_rows.append({"image_id": r["image_id"], "src_path": r["path"], "label": r["label"],
                                  "split": r["split"], "reason": "read_error", "n_dogs": 0, "max_conf": 0.0})
                log.warning("could not read %s: %s", r["path"], e)
        t0 = time.perf_counter()
        dets = det.detect(imgs)
        for k, v in det.last_speed_ms.items():
            model_ms[k].append(v)
        for r, im, d in zip(good, imgs, dets):
            res = select_and_crop(im, d, cfg.crop, multi_dog="drop",
                                  second_dog_min_conf=float(cfg.detector.second_dog_min_conf))
            if res.status != "ok":
                drop_rows.append({"image_id": r["image_id"], "src_path": r["path"], "label": r["label"],
                                  "split": r["split"], "reason": res.status, "n_dogs": len(d.boxes),
                                  "max_conf": round(max([b.conf for b in d.boxes] + [d.max_conf_below]), 4)})
                drop_dets[r["image_id"]] = (r["path"], d, res.status)
                continue
            out = crops_dir / safe_filename(r["label"]) / f"{r['image_id']}.jpg"
            res.crop.save(out, quality=95)
            b, s = res.box.xyxy, res.square
            crop_rows.append({"image_id": r["image_id"], "crop_path": rel(out), "src_path": r["path"],
                              "label": r["label"], "label_id": r["label_id"], "split": r["split"],
                              "breed": r["breed"], "det_conf": round(res.box.conf, 4), "det_class": res.box.cls_name,
                              "n_dogs": len(d.boxes),
                              "bx1": round(b[0], 1), "by1": round(b[1], 1), "bx2": round(b[2], 1),
                              "by2": round(b[3], 1), "sx1": s[0], "sy1": s[1], "sx2": s[2], "sy2": s[3],
                              "crop_side": s[2] - s[0], "fill_fraction": round(res.fill_fraction, 4)})
            ok_dets[r["image_id"]] = (r["path"], d, res.square)
        per_img_ms.append(1e3 * (time.perf_counter() - t0) / max(1, len(imgs)))

    write_csv(crop_rows, cfg.paths.crops_manifest, CROP_FIELDS)
    write_csv(drop_rows, cfg.paths.crop_drops_log, DROP_FIELDS)

    # ---- report ------------------------------------------------------------
    header("3. Report")
    n = len(rows)
    classes = sorted({r["label"] for r in rows})
    total_by = Counter(r["label"] for r in rows)
    drops_by = defaultdict(Counter)
    for d in drop_rows:
        drops_by[d["label"]][d["reason"]] += 1
    table = []
    for c in classes:
        dc = drops_by[c]
        table.append([c, total_by[c]] + [dc[x] for x in REASONS[:4]] + [f"{100 * sum(dc.values()) / total_by[c]:.1f}%"])
    allr = Counter(d["reason"] for d in drop_rows)
    table.append(["ALL", n] + [allr[x] for x in REASONS[:4]] + [f"{100 * len(drop_rows) / n:.1f}%"])
    print_table(table, ["class", "images"] + REASONS[:4] + ["dropped"])
    split_by = Counter((c["split"], c["label"]) for c in crop_rows)
    print()
    print_table([(c, split_by[("trainval", c)], split_by[("test", c)]) for c in classes],
                ["crops by class", "trainval", "test"])
    n_test = sum(1 for c in crop_rows if c["split"] == "test")
    print(f"\n  final test fraction: {n_test / max(1, len(crop_rows)):.3f}")
    sides = np.array([c["crop_side"] for c in crop_rows])
    confs = np.array([c["det_conf"] for c in crop_rows])
    fills = np.array([c["fill_fraction"] for c in crop_rows])
    ins = int(cfg.classifier.input_size)
    print(f"  crop side px: median {np.median(sides):.0f}, {100 * (sides < ins).mean():.1f}% below input_size {ins}")
    print(f"  det conf: median {np.median(confs):.2f}, p05 {np.percentile(confs, 5):.2f}")
    print("  YOLO class of kept boxes: " + ", ".join(f"{k} {v}" for k, v in Counter(c["det_class"] for c in crop_rows).most_common()))
    print(f"  crops with gray fill: {100 * (fills > 0).mean():.1f}% (mean fill {fills.mean():.3f})")
    print(f"  latency per image (decode excluded, detect + crop + save): mean {np.mean(per_img_ms):.1f} ms")
    print(f"  YOLO model time per image: " + ", ".join(f"{k} {np.mean(v):.1f} ms" for k, v in model_ms.items()))

    rates = {c: sum(drops_by[c].values()) / total_by[c] for c in classes}
    overall = len(drop_rows) / n
    biased = [c for c in classes if rates[c] > 2 * overall and rates[c] > 0.02]
    if biased:
        log.warning("drop rate for %s is more than 2x the overall %.1f%%: the detector may be biased "
                    "against that posture", biased, 100 * overall)
    report = {
        "total": n, "cropped": len(crop_rows), "dropped": len(drop_rows), "drop_rate": overall,
        "drop_reasons": dict(allr),
        "per_class": {c: {"total": total_by[c], "dropped": sum(drops_by[c].values()), "rate": rates[c],
                          "reasons": dict(drops_by[c])} for c in classes},
        "crops_per_split": {s: sum(1 for c in crop_rows if c["split"] == s) for s in ("trainval", "test")},
        "det_conf": {"median": float(np.median(confs)), "p05": float(np.percentile(confs, 5))},
        "crop_side": {"median": float(np.median(sides)), "frac_below_input": float((sides < ins).mean())},
        "fill": {"frac_with_fill": float((fills > 0).mean()), "mean": float(fills.mean())},
        "latency_ms_per_image": {"pipeline_mean": float(np.mean(per_img_ms)),
                                 **{f"yolo_{k}": float(np.mean(v)) for k, v in model_ms.items()},
                                 "device": det.device, "imgsz": det.imgsz, "batch_size": bs},
        "detector": {"weights": cfg.detector.weights, "conf_threshold": det.conf, "iou": det.iou},
        "crop": cfg.crop.to_dict(),
    }
    save_json(report, "results/detection_report.json")

    # ---- visuals -----------------------------------------------------------
    if cfg.debug.save_debug_grid:
        header("4. Debug visuals")
        rng = random.Random(cfg.seed)
        out = [plot_hbar(classes + ["ALL"], [rates[c] for c in classes] + [overall], dbg / "04_drop_rate_by_class.png",
                         title="Images dropped by the detector step, by class", xlabel="fraction dropped"),
               histogram(sides.tolist(), dbg / "04_crop_side.png", title="Square crop side",
                         xlabel="pixels", vlines={f"input_size {ins}": ins}),
               histogram(confs.tolist(), dbg / "04_det_conf.png", title="Detector confidence of kept dogs",
                         xlabel="confidence", clip_pct=None)]
        ov = []
        for iid in rng.sample(sorted(ok_dets), min(16, len(ok_dets))):
            path, d, sq = ok_dets[iid]
            with Image.open(resolve(path)) as im:
                ov.append((draw_detections(im, d, sq), iid))
        out.append(save_contact_sheet(ov, dbg / "04_boxes_overlay.jpg", cols=8, thumb=200,
                                      title="Blue = dog box (conf), orange = square crop"))
        for c in classes:
            pool = [r for r in crop_rows if r["label"] == c]
            items = [(r["crop_path"], f"{c}\n{r['image_id']}\nconf {r['det_conf']:.2f} side {r['crop_side']}")
                     for r in rng.sample(pool, min(32, len(pool)))]
            out.append(save_contact_sheet(items, dbg / f"04_crops_{c}.jpg", cols=8, title=f"32 random '{c}' crops"))
        for reason in sorted({v[2] for v in drop_dets.values()}):
            ids = [k for k, v in drop_dets.items() if v[2] == reason]
            items = []
            for iid in rng.sample(sorted(ids), min(24, len(ids))):
                path, d, _ = drop_dets[iid]
                with Image.open(resolve(path)) as im:
                    items.append((draw_detections(im, d), f"{reason}\n{iid}"))
            out.append(save_contact_sheet(items, dbg / f"04_drops_{reason}.jpg", cols=8, thumb=180,
                                          title=f"Dropped as '{reason}' ({len(ids)} images), boxes >= threshold drawn"))
        for o in out:
            log.info("wrote %s", rel(o))

    # ---- sanity ------------------------------------------------------------
    header("5. Sanity checks")
    cid, did = [c["image_id"] for c in crop_rows], [d["image_id"] for d in drop_rows]
    assert len(cid) + len(did) == n and len(set(cid)) == len(cid) and set(cid).isdisjoint(did)
    print("  [ok] every image is either cropped or dropped, exactly once")
    for c in crop_rows:
        assert c["sx2"] - c["sx1"] == c["sy2"] - c["sy1"] == c["crop_side"], f"not square: {c['image_id']}"
        assert c["sx1"] <= c["bx1"] and c["sy1"] <= c["by1"] and c["sx2"] >= c["bx2"] and c["sy2"] >= c["by2"]
        assert c["det_conf"] >= det.conf and c["n_dogs"] >= 1 and 0 <= c["fill_fraction"] < 1
    print("  [ok] all crops square, contain the whole dog box, conf >= threshold")
    for c in random.Random(cfg.seed).sample(crop_rows, min(200, len(crop_rows))):
        with Image.open(resolve(c["crop_path"])) as im:
            assert im.size == (c["crop_side"], c["crop_side"]) and im.mode == "RGB", c["crop_path"]
    print("  [ok] 200 random crop files reopen at the recorded size")
    src = {r["image_id"]: r for r in rows}
    assert all(c["label"] == src[c["image_id"]]["label"] and c["split"] == src[c["image_id"]]["split"]
               for c in crop_rows), "label/split changed"
    assert all(d["n_dogs"] >= 2 for d in drop_rows if d["reason"] == "multi_dog")
    print("  [ok] labels and splits carried over unchanged; multi_dog drops have >= 2 boxes")
    if overall > 0.15:
        log.warning("drop rate %.1f%% > 15%%: feasibility red flag", 100 * overall)
    if not cfg.debug.limit_images:
        for c in classes:
            assert split_by[("test", c)] >= 30, f"too few test crops for {c}"
        print("  [ok] every class has >= 30 test crops")
    print("\n  All sanity checks passed. Next: python scripts/05_train_classifier.py")


if __name__ == "__main__":
    main()

# DEBUGGING GUIDE
#   04_boxes_overlay.jpg   blue box should hug the dog, orange square should contain it with a margin.
#   04_crops_<class>.jpg   head and tail visible; gray bars only where the dog touches the image edge.
#   04_drops_<reason>.jpg  multi_dog should really show several dogs; no_dog should be odd images.
#   04_drop_rate_by_class.png  similar rates = fine; one posture much higher = detector bias.
#   Changing detector.* or crop.* requires re-running 04 and 05.
