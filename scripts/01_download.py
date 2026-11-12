"""
01_download.py  (PSEUDOCODE, not runnable yet)

Purpose
-------
Download DogPoseCV (`stockeh/dog-pose-cv`) from Hugging Face, discover the real label
names from the dataset's ClassLabel feature, and dump every image to disk with a manifest.

Inputs
------
- configs/default.yaml  (dataset.*, paths.raw_dir, paths.raw_manifest, paths.hf_cache_dir, seed, debug.*)
- Network access to the Hugging Face Hub.

Outputs
-------
- data/raw/<original_label>/<image_id>.jpg
- data/raw/manifest.csv with columns:
      image_id, path, original_label, original_label_id, hf_split, width, height
  (hf_split is extra: it records which split the HF repo shipped the row in, for traceability only.
   We do NOT reuse the HF split. 03_split.py makes its own.)
- stdout: feature schema, exact ClassLabel names, per-class counts.

Run
---
    python scripts/01_download.py --config configs/default.yaml [--limit N]
"""

# ---------------------------------------------------------------------------
# Imports (planned)
# ---------------------------------------------------------------------------
# import argparse, csv, logging, pathlib, collections
# from datasets import load_dataset, ClassLabel, Image as HFImage, concatenate_datasets
# from trickforTreat.common import load_config, set_seed, setup_logging, safe_filename


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def parse_args() -> "argparse.Namespace":
    """--config (default configs/default.yaml), --limit (overrides debug.limit_images), --overwrite."""
    # TODO


def load_hf_dataset(cfg) -> "datasets.DatasetDict":
    """
    ds = load_dataset(cfg.dataset.hf_repo, cache_dir=cfg.paths.hf_cache_dir)   # latest version, not pinned
    log the split names and row counts, e.g. {"train": 20578} or {"train": ..., "test": ...}
    return ds
    """
    # TODO


def find_columns(features) -> tuple[str, str]:
    """
    Auto-detect the image column and label column unless set in config.

    image_cols = [name for name, f in features.items() if isinstance(f, HFImage)]
    label_cols = [name for name, f in features.items() if isinstance(f, ClassLabel)]
    if cfg overrides are set, use them
    assert len(image_cols) == 1, f"ambiguous image columns: {image_cols}"
    assert len(label_cols) == 1, f"ambiguous label columns: {label_cols}"
       -> if this fails, the labels may be stored as plain strings/ints. Print features and
          set dataset.label_column manually. (Open question until we actually look.)
    return image_col, label_col
    """
    # TODO


def print_label_names(features, label_col) -> list[str]:
    """
    names = features[label_col].names
    for idx, name in enumerate(names):
        print(f"{idx}: {name!r}")       # repr() so trailing spaces / underscores / case are visible
    print reminder: "Copy these EXACT strings into configs/default.yaml -> labels.mapping"
    return names
    """
    # TODO


def save_images(ds_dict, image_col, label_col, label_names, cfg) -> list[dict]:
    """
    rows = []
    for split_name, split in ds_dict.items():
        for i, example in enumerate(split):            # streaming-friendly; avoid ds[:] which loads all
            if limit reached: break
            img = example[image_col]                    # PIL.Image
            label_id = example[label_col]
            label = label_names[label_id]
            image_id = f"{split_name}_{i:06d}"          # TODO: use a native id/filename column if one exists
                                                        #       (e.g. Stanford Dogs filename like n02085620_10074)
            if img.mode != "RGB": img = img.convert("RGB")   # some Stanford Dogs images are CMYK/L/RGBA
            out = raw_dir / safe_filename(label) / f"{image_id}.jpg"
            if out exists and not --overwrite: skip writing, still record row
            img.save(out, quality=cfg.dataset.jpeg_quality)
            rows.append(dict(image_id, path=str(out relative to repo), original_label=label,
                             original_label_id=label_id, hf_split=split_name,
                             width=img.width, height=img.height))
    return rows
    """
    # TODO


def write_manifest(rows, path) -> None:
    """csv.DictWriter with fixed column order; sort rows by image_id for deterministic diffs."""
    # TODO


def print_class_counts(rows, label_names) -> None:
    """
    counts = Counter(r["original_label"] for r in rows)
    print table: label | count | percent
    compare against expected counts from the dataset card:
        standing 4143, sitting 3038, lying down 7090, undefined 6307  (total 20578)
    print a WARNING (not a crash) if they differ by >1%: the HF version may differ from the paper.
    """
    # TODO


def sanity_checks(rows, label_names, cfg) -> None:
    """
    assert len(rows) > 0
    if not limited: assert len(rows) == 20578 (or warn, see above)
    assert len({r["image_id"] for r in rows}) == len(rows)          # unique ids
    assert set(r["original_label"] for r in rows) <= set(label_names)
    assert every path exists on disk                                  # spot-check a random 100 if slow
    assert every width, height > 0
    assert every key in cfg.labels.mapping and cfg.labels.dropped_original_labels is in label_names
        -> if not, print a CLEAR message:
        "Config label keys {missing} not found in dataset names {label_names}. Fix labels.mapping."
        (Make this a warning here, but 02 will hard-fail on it.)
    """
    # TODO


def main():
    """
    args = parse_args(); cfg = load_config(args.config); set_seed(cfg.seed); setup_logging(cfg)
    ds = load_hf_dataset(cfg)
    image_col, label_col = find_columns(first_split.features)
    label_names = print_label_names(features, label_col)
    rows = save_images(ds, image_col, label_col, label_names, cfg)
    write_manifest(rows, cfg.paths.raw_manifest)
    print_class_counts(rows, label_names)
    sanity_checks(rows, label_names, cfg)
    if cfg.debug.save_debug_grid: save a 4x8 contact sheet (8 random images per class) to
        results/debug/01_samples_by_class.jpg, so you can eyeball what each label looks like
    """
    # TODO


# if __name__ == "__main__":
#     main()


# ---------------------------------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------------------------------
# Intermediates this script produces:
#   data/hf_cache/                  HF's Arrow cache. Safe to delete; re-download takes a few minutes.
#   data/raw/<label>/*.jpg          One JPEG per dataset row, foldered by ORIGINAL label string.
#   data/raw/manifest.csv           One row per image. This is the source of truth for 02.
#   results/debug/01_samples_by_class.jpg   Visual check of what each label means.
#
# How to check it worked:
#   1. The printed ClassLabel names: do they match labels.mapping keys EXACTLY (case, spaces,
#      underscores)? "lying down" vs "lying_down" vs "lying" is the most likely mismatch.
#   2. `ls data/raw` should show one folder per label (4 expected).
#   3. `wc -l data/raw/manifest.csv` should be 20579 (20578 + header) on a full run.
#   4. Open the contact sheet. "undefined" should look like close-up faces. If sitting/lying look
#      swapped, the label-id -> name mapping is wrong.
#
# Common failures:
#   - find_columns assert fires: labels aren't a ClassLabel. Print ds["train"].features, then set
#     dataset.label_column / image_column in the config and map ints to names manually.
#   - PIL "cannot write mode CMYK as JPEG": the convert("RGB") line is missing.
#   - Counts differ from the card: note it in the open questions. Don't "fix" it.
#   - Smoke test: `--limit 50` should finish in seconds and exercise every code path.
