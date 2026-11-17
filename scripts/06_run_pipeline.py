#!/usr/bin/env python3
"""
06_run_pipeline.py

Run the full runtime pipeline (frame -> YOLO dog detector -> square crop -> posture classifier ->
per-frame reward decision) on any image source. This is the code path the live system will use;
only the --source changes.

Examples
--------
  # held-out test images, straight from the ORIGINAL photos (not the pre-made crops)
  python scripts/06_run_pipeline.py --source data/split_manifest.csv --split test --max-frames 300

  # a folder of your own photos, deciding treats for "sit"
  python scripts/06_run_pipeline.py --source path/to/photos --command sit

  # a video file, every 5th frame
  python scripts/06_run_pipeline.py --source clip.mp4 --every-n 5

  # the ESP32-CAM MJPEG stream (later)
  python scripts/06_run_pipeline.py --source http://<esp32-ip>:81/stream --every-n 3 --max-frames 200

Outputs (results/pipeline/<run name>/):
  results.csv          one row per frame: status, prediction, probabilities, reward, timings
  annotated.jpg        contact sheet of annotated frames (box, square crop, prediction)
  summary.json         status counts, latency, and accuracy if the source carries true labels
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trickforTreat.common import (  # noqa: E402
    header, load_config, print_table, rel, resolve, safe_filename, save_contact_sheet, save_json,
    setup_logging, write_csv, DEFAULT_CONFIG,
)
from trickforTreat.metrics import classification_metrics, plot_confusion  # noqa: E402
from trickforTreat.pipeline import PosturePipeline, annotate  # noqa: E402
from trickforTreat.sources import open_source  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run detector + classifier on any image source.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--source", required=True, help="folder, image, .csv manifest, video file, webcam index, or URL")
    p.add_argument("--split", default=None, help="for .csv sources: only rows with this split (e.g. test)")
    p.add_argument("--checkpoint", default="checkpoints/final.pt")
    p.add_argument("--command", choices=["sit", "down"], default=None, help="commanded posture for reward decisions")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--every-n", type=int, default=1, help="video sources: keep every n-th frame")
    p.add_argument("--annotate", type=int, default=48, help="how many annotated frames to put on the contact sheet")
    p.add_argument("--name", default=None, help="run name for the output folder")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("06_pipeline", cfg.debug.log_level)
    torch.set_num_threads(int(cfg.train.torch_threads))
    name = args.name or safe_filename(Path(args.source).stem + (f"_{args.split}" if args.split else ""))
    out_dir = resolve("results/pipeline") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    header("1. Load pipeline")
    pipe = PosturePipeline.from_config(cfg, str(resolve(args.checkpoint)))
    pipe.detector.warmup()
    clf = pipe.classifier
    log.info("detector %s | classifier %s @%dpx, classes %s, reward thresholds %s",
             cfg.detector.weights, clf.backbone, clf.input_size, clf.classes, clf.reward_thresholds)

    header("2. Run")
    source = open_source(args.source, split=args.split, every_n=args.every_n, max_frames=args.max_frames)
    rows, sheet, t_start = [], [], time.perf_counter()
    with source:
        for n, (frame, res) in enumerate(pipe.run(source, commanded=args.command)):
            if args.max_frames and n >= args.max_frames:
                break
            pr = res.prediction
            row = {"frame_id": res.frame_id, "status": res.status, "true": frame.meta.get("label", ""),
                   "pred": pr.label if pr else "", "confidence": round(pr.confidence, 4) if pr else "",
                   **{f"p_{c}": round(pr.probs[c], 4) if pr else "" for c in clf.classes},
                   "reward": "" if res.reward is None else res.reward,
                   **{f"ms_{k}": round(v, 1) for k, v in res.timings_ms.items()}}
            rows.append(row)
            if n < 15:
                print(f"  {res.frame_id:22s} {res.status:10s} {row['pred']:6s} {row['confidence']!s:7s} "
                      f"true={row['true'] or '?':6s} reward={row['reward']!s:5s} {res.timings_ms['total']:.0f} ms")
            elif n == 15:
                print("  ...")
            if len(sheet) < args.annotate:
                sheet.append((annotate(frame.image, res), f"true {row['true'] or '?'}"))
    assert rows, "the source produced no frames"
    elapsed = time.perf_counter() - t_start

    fields = list(rows[0])
    write_csv(rows, out_dir / "results.csv", fields)
    save_contact_sheet(sheet, out_dir / "annotated.jpg", cols=8, thumb=200,
                       title=f"{args.source}: blue = dog box, orange = crop, caption = prediction")

    header("3. Summary")
    status = Counter(r["status"] for r in rows)
    print_table(sorted(status.items()), ["status", "frames"])
    preds = Counter(r["pred"] for r in rows if r["pred"])
    print_table(sorted(preds.items()), ["prediction", "frames"])
    lat = {k: float(np.mean([r[f"ms_{k}"] for r in rows])) for k in ("detect", "crop", "classify", "total")}
    print_table([(k, v) for k, v in lat.items()], ["stage", "mean ms"])
    print(f"\n  {len(rows)} frames in {elapsed:.1f}s ({len(rows) / elapsed:.1f} fps end to end, batch 1)")
    summary = {"source": args.source, "frames": len(rows), "status": dict(status), "predictions": dict(preds),
               "latency_ms": lat, "fps": len(rows) / elapsed, "command": args.command}

    labeled = [r for r in rows if r["true"] in clf.classes and r["status"] == "ok"]
    if labeled:
        y = np.array([clf.classes.index(r["true"]) for r in labeled])
        probs = np.array([[r[f"p_{c}"] for c in clf.classes] for r in labeled], dtype=float)
        m = classification_metrics(probs, y, clf.classes)
        print_table([(c, m["per_class"][c]["precision"], m["per_class"][c]["recall"], m["per_class"][c]["support"])
                     for c in clf.classes], ["class (frames with a dog)", "precision", "recall", "support"])
        print(f"  accuracy {m['accuracy']:.4f}  macro F1 {m['macro_f1']:.4f} on {len(labeled)} frames where a "
              f"single dog was found ({len(labeled) / len(rows):.1%} of frames)")
        plot_confusion(m["confusion_matrix"], clf.classes, out_dir / "confusion.png", f"Pipeline on {name}")
        summary["metrics"] = m
    if args.command:
        r = [x for x in rows if x["reward"] is True]
        print(f"\n  command '{args.command}': {len(r)} / {len(rows)} frames would earn a treat")
        if labeled:
            wrong = [x for x in r if x["true"] and x["true"] != args.command]
            print(f"  ...of which {len(wrong)} have a different true label (false rewards)")
    save_json(summary, out_dir / "summary.json")
    log.info("wrote %s/{results.csv, annotated.jpg, summary.json}", rel(out_dir))


if __name__ == "__main__":
    main()
