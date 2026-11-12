"""
02_filter_labels.py  (PSEUDOCODE, not runnable yet)

Purpose
-------
Map DogPoseCV's original labels to project labels and drop the rest.
    sitting    -> sit
    lying down -> down
    standing   -> stand
    undefined  -> dropped (always; listed in labels.dropped_original_labels)

Inputs
------
- data/raw/manifest.csv  (from 01)
- configs/default.yaml   (labels.*, paths.*)

Outputs
-------
- data/filtered_manifest.csv with columns:
      image_id, path, original_label, label, label_id, width, height
  label_id is the index into the class list ["sit", "down", "stand"].
- data/classes.json: {"classes": [...], "target_classes": [...]}. This file is the single source
  of truth for class index order. 05 and trickforTreat/classifier.py read it, never re-derive it.

Run
---
    python scripts/02_filter_labels.py --config configs/default.yaml
"""

# import argparse, csv, json, logging, collections
# from trickforTreat.common import load_config, setup_logging, read_csv, write_csv


def build_class_list(cfg) -> list[str]:
    """
    classes = list(dict.fromkeys(cfg.labels.mapping.values()))   # ["sit", "down", "stand"], order preserved
    assert len(classes) == len(cfg.labels.mapping)               # no two original labels map to one class
    assert set(cfg.labels.target_classes) <= set(classes)
    return classes
    """
    # TODO


def build_label_map(cfg, observed_labels: set[str]) -> dict[str, str | None]:
    """
    Returns {original_label: project_label or None (= drop)}.

    m = dict(cfg.labels.mapping)
    for lbl in cfg.labels.dropped_original_labels: m[lbl] = None
    missing = set(m) - observed_labels
    if missing: raise ValueError(f"config labels {missing} not in manifest labels {observed_labels}."
                                 " Re-check the names 01 printed.")      # HARD FAIL: a silent
                                                                          # mismatch would drop a whole class
    unknown = observed_labels - set(m)
    if unknown: raise ValueError(f"unmapped original labels {unknown}")  # force an explicit decision
    return m
    """
    # TODO


def filter_rows(rows, label_map, classes) -> tuple[list[dict], Counter]:
    """
    kept, dropped = [], Counter()
    for r in rows:
        new = label_map[r["original_label"]]
        if new is None: dropped[r["original_label"]] += 1; continue
        kept.append({**r, "label": new, "label_id": classes.index(new)})
    return kept, dropped
    """
    # TODO


def report(kept, dropped, classes) -> None:
    """
    Print:
      - kept counts per project label + percent
      - dropped counts per original label
      - imbalance ratio max_count / min_count  (expect about 2.3x: down 7090 vs sit 3038)
    """
    # TODO


def sanity_checks(kept, rows, dropped, classes, cfg) -> None:
    """
    assert len(kept) + sum(dropped.values()) == len(rows)                  # nothing silently lost
    assert {r["label"] for r in kept} == set(classes)                      # every class present
    assert all(classes[r["label_id"]] == r["label"] for r in kept)         # id <-> name consistent
    assert not ({r["original_label"] for r in kept} & set(cfg.labels.dropped_original_labels))
    if not limited: expected 14271 kept (4143 + 3038 + 7090); warn if different
    assert data/classes.json round-trips to `classes`
    """
    # TODO


def main():
    """
    cfg = load_config(...); rows = read_csv(cfg.paths.raw_manifest)
    classes = build_class_list(cfg)
    label_map = build_label_map(cfg, {r["original_label"] for r in rows})
    kept, dropped = filter_rows(rows, label_map, classes)
    write_csv(kept, cfg.paths.filtered_manifest)       # sorted by image_id
    json.dump({"classes": classes, "target_classes": cfg.labels.target_classes},
              open(cfg.paths.classes_json, "w"))
    report(kept, dropped, classes)
    sanity_checks(...)
    NOTE: no images are copied. Paths still point into data/raw/.
    """
    # TODO


# if __name__ == "__main__":
#     main()


# ---------------------------------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------------------------------
# Intermediates:
#   data/filtered_manifest.csv  Subset of manifest.csv plus `label` and `label_id` columns.
#   data/classes.json           Ordered class list ["sit", "down", "stand"].
#
# How to check:
#   - `cut -d, -f4 data/filtered_manifest.csv | sort | uniq -c` gives counts per project label.
#   - Expect ~14.3k rows and 3 classes; no "undefined" anywhere in original_label.
#
# Common failures:
#   - "config labels ... not in manifest labels": the config keys don't match the strings
#     01 printed. Copy them exactly.
#   - Changing labels.mapping changes classes.json, so re-run 03, 04, 05 afterwards.
