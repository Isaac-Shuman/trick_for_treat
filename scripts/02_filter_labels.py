#!/usr/bin/env python3
"""
02_filter_labels.py

Map DogPoseCV's original labels to project labels and drop the rest.
    sitting    -> sit
    lying      -> down     (the CSVs say "lying"; the dataset card says "lying down")
    standing   -> stand
    undefined  -> dropped  (labels.dropped_original_labels)

Inputs
------
- data/raw/manifest.csv  (from 01)
- configs/default.yaml   (labels.*, paths.*)

Outputs
-------
- data/filtered_manifest.csv: raw manifest columns + `label` (project label) + `label_id`
- data/classes.json: {"classes": [...], "target_classes": [...]}. The single source of truth
  for class index order. 05 and trickforTreat/classifier.py read it; nothing re-derives it.

Run
---
    python scripts/02_filter_labels.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trickforTreat.common import (  # noqa: E402
    add_common_args, header, load_config, print_table, read_csv, rel, resolve, save_json,
    setup_logging, write_csv,
)

RAW_FIELDS = ["image_id", "path", "original_label", "breed", "source_member", "orig_mode", "width", "height"]
FIELDS = RAW_FIELDS + ["label", "label_id"]
EXPECTED_KEPT = 14165  # 7047 standing + 3008 sitting + 4110 lying, measured 2026-11-12


def build_class_list(cfg) -> list[str]:
    classes = list(dict.fromkeys(cfg.labels.mapping.values()))  # order preserved: sit, down, stand
    assert len(classes) == len(cfg.labels.mapping), "two original labels map to the same class"
    assert set(cfg.labels.target_classes) <= set(classes), "target_classes must be project classes"
    return classes


def build_label_map(cfg, observed: set[str]) -> dict[str, str | None]:
    m: dict[str, str | None] = dict(cfg.labels.mapping)
    for lbl in cfg.labels.dropped_original_labels:
        m[lbl] = None
    missing = set(m) - observed
    if missing:
        raise ValueError(f"config labels {sorted(missing)} not in manifest labels {sorted(observed)}")
    unknown = observed - set(m)
    if unknown:
        raise ValueError(f"unmapped original labels {sorted(unknown)}: add them to labels.mapping "
                         "or labels.dropped_original_labels")
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="Map original labels to sit/down/stand.")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, limit=args.limit)
    log = setup_logging("02_filter", cfg.debug.log_level)

    rows = read_csv(cfg.paths.raw_manifest)
    assert list(rows[0]) == RAW_FIELDS, f"unexpected raw manifest columns {list(rows[0])}; re-run 01"
    classes = build_class_list(cfg)
    label_map = build_label_map(cfg, {r["original_label"] for r in rows})

    header("Label mapping")
    print_table([(k, v or "(dropped)") for k, v in label_map.items()], ["original", "project"])

    kept, dropped = [], Counter()
    for r in rows:
        new = label_map[r["original_label"]]
        if new is None:
            dropped[r["original_label"]] += 1
            continue
        kept.append({**r, "label": new, "label_id": classes.index(new)})

    header("Counts")
    kc = Counter(r["label"] for r in kept)
    print_table([(c, kc[c], f"{100 * kc[c] / len(kept):.1f}%") for c in classes], ["class", "kept", "share"])
    print_table(sorted(dropped.items()), ["dropped original label", "images"])
    print(f"\n  imbalance (largest / smallest class): {max(kc.values()) / min(kc.values()):.2f}x")

    write_csv(kept, cfg.paths.filtered_manifest, FIELDS)
    save_json({"classes": classes, "target_classes": list(cfg.labels.target_classes)}, cfg.paths.classes_json)
    log.info("wrote %s (%d rows) and %s", cfg.paths.filtered_manifest, len(kept), cfg.paths.classes_json)

    header("Sanity checks")
    assert len(kept) + sum(dropped.values()) == len(rows), "rows lost"
    assert {r["label"] for r in kept} == set(classes), "a class has no images"
    assert all(classes[r["label_id"]] == r["label"] for r in kept)
    assert not ({r["original_label"] for r in kept} & set(cfg.labels.dropped_original_labels))
    with open(resolve(cfg.paths.classes_json)) as f:
        assert json.load(f)["classes"] == classes
    print("  [ok] no rows lost, every class present, ids consistent, dropped labels gone, classes.json round-trips")
    if not cfg.debug.limit_images and len(kept) != EXPECTED_KEPT:
        log.warning("kept %d rows, expected %d. Did the dataset change upstream?", len(kept), EXPECTED_KEPT)
    elif not cfg.debug.limit_images:
        print(f"  [ok] kept {len(kept):,} rows, as expected")
    print("\n  All sanity checks passed. Next: python scripts/03_split.py")


if __name__ == "__main__":
    main()

# DEBUGGING GUIDE
#   data/filtered_manifest.csv  raw manifest rows for sit/down/stand, plus `label` and `label_id`.
#   data/classes.json           ordered class list. Changing labels.mapping changes it: re-run 03-05.
#   Failure "config labels ... not in manifest labels": config keys must match 01's printed names exactly.
