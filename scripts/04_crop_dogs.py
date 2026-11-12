"""
04_crop_dogs.py  (PSEUDOCODE, not runnable yet)

Purpose
-------
Run an Ultralytics YOLO (nano/small) detector on every image in the split manifest, keeping only
COCO class `dog` (id 16). For each image:
  - no dog box >= conf_threshold     -> skip and log ("no_dog" or "below_conf")
  - more than one dog box >= thresh  -> drop and log ("multi_dog"); the label is ambiguous
  - dog box smaller than min side    -> drop and log ("too_small")
  - otherwise: pad the box, expand it to a SQUARE around the dog, crop; where the square runs past
    the image edge, fill with gray. The square is always big enough that head and tail stay in frame.

This mirrors what the live pipeline will do on ESP32 frames, so the classifier trains on the same
kind of input it will see at runtime.

All detection + crop logic lives in trickforTreat/detector.py (DogDetector, square_crop_box,
crop_with_fill). This script is only a batch driver around it.

Inputs
------
- data/split_manifest.csv  (from 03)
- configs/default.yaml     (detector.*, crop.*, paths.*)

Outputs
-------
- data/crops/<label>/<image_id>.jpg   square crops at native resolution (not resized)
- data/crops_manifest.csv with columns:
      image_id, crop_path, src_path, label, label_id, split, det_conf,
      bx1, by1, bx2, by2   (raw detector box, pixel coords in source image)
      sx1, sy1, sx2, sy2   (square crop box, pixel coords; MAY extend past the image -> gray fill)
      crop_side, fill_fraction (fraction of the crop that is gray fill)
- data/crop_drops.csv: image_id, src_path, label, split, reason, n_dogs, max_conf
      reason in {"no_dog", "below_conf", "multi_dog", "too_small", "read_error"}
- results/detection_report.json: overall + per-class + per-split drop rates by reason, latency
- stdout: the same report

Run
---
    python scripts/04_crop_dogs.py --config configs/default.yaml [--limit N]
"""

# import argparse, json, time, logging
# from PIL import Image
# from trickforTreat.common import load_config, set_seed, read_csv, write_csv
# from trickforTreat.detector import DogDetector, square_crop_box, crop_with_fill


def process_batch(detector, batch_rows, cfg) -> tuple[list[dict], list[dict]]:
    """
    imgs = [Image.open(r["path"]).convert("RGB") for r in batch_rows]    # catch errors -> "read_error"
    dets = detector.detect_dogs(imgs)       # list[FrameDetections]: all dog boxes >= conf, sorted by conf,
                                            # plus max_conf of anything below threshold (see detector.py)
    crop_rows, drop_rows = [], []
    for r, img, d in zip(batch_rows, imgs, dets):
        if len(d.boxes) == 0:
            reason = "below_conf" if d.max_conf_below_threshold > 0 else "no_dog"
            drop_rows.append(...); continue
        if len(d.boxes) > 1:
            drop_rows.append(reason="multi_dog", n_dogs=len(d.boxes)); continue
        box = d.boxes[0]
        if min(box width, box height) < cfg.crop.min_box_side_px:
            drop_rows.append(reason="too_small"); continue
        sq = square_crop_box(box.xyxy, cfg.crop.pad_fraction)        # may extend past image bounds
        c = crop_with_fill(img, sq, cfg.crop.fill_rgb)               # square PIL image
        save c to data/crops/<label>/<image_id>.jpg
        crop_rows.append({..., det_conf=box.conf, b*=box.xyxy, s*=sq, crop_side, fill_fraction})
    return crop_rows, drop_rows

    NOTE: do NOT resize crops here. Keep native resolution; 05's transforms resize to 224x224.
    NOTE: multi_dog counts any second dog box >= conf_threshold, including a false positive.
          If multi_dog drops look like single-dog images in the debug grid, raise conf_threshold
          or ignore second boxes much smaller than the first (TODO if needed).
    """
    # TODO


def build_report(crop_rows, drop_rows, all_rows, latencies) -> dict:
    """
    {
      "total": N, "cropped": len(crop_rows), "dropped": len(drop_rows),
      "drop_rate": dropped / total,
      "drop_reasons": Counter(reason),
      "per_class": {label: {"total", "dropped", "rate", "reasons": Counter}},
      "per_split": {"trainval": {...}, "test": {...}},      # includes final test fraction after drops
      "det_conf": {"mean", "p05", "p50", "p95"},
      "fill_fraction": {"mean", "p95", "frac_crops_with_fill"},
      "latency_ms_per_image": {"mean", "p50", "p95", "device", "imgsz", "batch_size"},
      "detector": {"weights", "conf_threshold", "iou_threshold"}, "crop": {...}
    }
    Highlight if any class's no-detection rate is > 2x the overall rate. That means the detector is
    biased against a posture (e.g., lying dogs look less "dog-like" to COCO YOLO), and it matters a
    lot because the runtime pipeline would never get a chance to classify those frames.
    """
    # TODO


def sanity_checks(crop_rows, drop_rows, all_rows, cfg) -> None:
    """
    assert len(crop_rows) + len(drop_rows) == len(all_rows)
    assert {crop ids} and {drop ids} are disjoint, and each is unique
    for every crop row:
        assert sx2 - sx1 == sy2 - sy1 == crop_side                  # square
        assert sx1 <= bx1 and sy1 <= by1 and sx2 >= bx2 and sy2 >= by2   # square contains the whole box
        assert crop file exists and its size == (crop_side, crop_side)
        assert det_conf >= cfg.detector.conf_threshold
        assert 0 <= fill_fraction < 1
        assert split and label copied unchanged from split manifest
    assert every drop row with reason == "multi_dog" has n_dogs >= 2
    warn (not crash) if drop_rate > 0.15: feasibility red flag
    assert every class still has >= 30 crops in the test split
    """
    # TODO


def main():
    """
    cfg = load_config(...); set_seed(cfg.seed)
    rows = read_csv(cfg.paths.split_manifest)  (apply debug.limit_images)
    detector = DogDetector.from_config(cfg.detector)   # loads YOLO weights once; logs device
    detector.warmup()
    for batch in chunks(rows, cfg.detector.batch_size):          # tqdm progress bar
        t0 = perf_counter(); c, d = process_batch(detector, batch, cfg); t1 = perf_counter()
        latencies.append((t1 - t0) / len(batch))   # includes decode + detect + crop + save
                                                   # (detector.py also exposes pure model latency)
    write crops_manifest.csv, crop_drops.csv, detection_report.json
    print report
    sanity_checks(...)
    debug grids (if enabled):
        results/debug/04_crops_<label>.jpg          random 32 crops per class
        results/debug/04_boxes_overlay.jpg          16 source images with raw box + square box drawn
        results/debug/04_drops_<reason>.jpg         random 32 dropped images per reason, boxes drawn
    """
    # TODO


# if __name__ == "__main__":
#     main()


# ---------------------------------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------------------------------
# Intermediates:
#   data/crops/<label>/*.jpg        Square dog crops at native resolution. What the classifier sees.
#   data/crops_manifest.csv         One row per crop, with raw box, square box, det confidence.
#   data/crop_drops.csv             Every image that did not produce a crop, with a reason.
#   results/detection_report.json   Rates and latency, the numbers you'd quote in the report.
#   results/debug/04_*.jpg          Visual checks.
#
# How to check:
#   1. Open 04_boxes_overlay.jpg: the raw box should hug the dog; the square should contain it with a
#      small margin. If boxes are shifted/scaled wrong, you're mixing normalized (xyxyn) and pixel
#      (xyxy) coords.
#   2. Open 04_crops_down.jpg: lying dogs are wide, so their crops will be square with the dog across
#      the middle and gray bars if near an edge. Head and tail must both be visible. If not, increase
#      crop.pad_fraction.
#   3. Open 04_drops_no_dog.jpg / below_conf: are failures mostly weird images (drawings, tiny dogs,
#      heavy occlusion)? Or clear dogs in a certain posture? The latter is a real problem.
#   4. Open 04_drops_multi_dog.jpg: should actually show multiple dogs. Lots of single-dog images
#      here means false-positive second boxes (see NOTE in process_batch).
#   5. Compare per-class no-detection rates. Similar = fine. `down` much higher = detector bias.
#
# Tuning:
#   - Lower detector.conf_threshold (e.g. 0.15) to recover more dogs; this also raises multi_dog drops.
#   - Switch weights to the "s" model if the nano model misses too many.
#   - Changing any detector.* or crop.* value requires re-running 04 and 05 (crops change).
#   - Smoke test: --limit 32 on CPU should run in well under a minute.
