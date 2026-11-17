#!/usr/bin/env python3
"""
01_download.py

Download DogPoseCV (`stockeh/dog-pose-cv`) from Hugging Face, join images to their labels,
and write every labeled image to disk with a manifest.

What the HF repo actually contains (found by inspecting it on 2026-11-12):
  data/images.tar.gz   images/<synset>-<breed>/<synset>_<n>.jpg   (121 folders, ~20.7k JPEGs)
                       plus macOS junk (._* files, .DS_Store) and an images/validation/ folder
                       with 11 unlabeled images + 139 copies of Pekinese images.
  data/labels.tar.gz   labels/<synset>-<breed>.csv, columns `id,label`, one CSV per breed (119 CSVs).
                       Label strings: standing, sitting, lying, undefined.
The labels are NOT wired into `datasets.load_dataset` (it sees only images), so this script
downloads both tarballs with huggingface_hub and joins them itself.

Known differences from the dataset card (all checked visually and by count):
  - The card says "lying down"; the CSVs say "lying".
  - The card's standing (4143) and lying (7090) counts look swapped. The CSVs have about 7.0k
    standing and 4.1k lying, and a contact sheet confirms the CSV labels match the pictures.
  - Standard Schnauzer (155 images) has no label CSV, and 2 Leonberg images are missing from
    their CSV. These are skipped. 20,578 (card) - 155 (schnauzer) = 20,423 labeled images.

Inputs
------
- configs/default.yaml  (dataset.*, labels.*, paths.raw_dir, paths.raw_manifest, paths.hf_cache_dir, seed, debug.*)
- Network access to the Hugging Face Hub (first run only; the tarballs are cached in data/hf_cache/).

Outputs
-------
- data/raw/<original_label>/<image_id>.jpg
- data/raw/manifest.csv, one row per labeled image:
      image_id        original Stanford Dogs filename stem, e.g. n02085936_8767 (globally unique)
      path            repo-relative path of the saved JPEG
      original_label  label string exactly as the CSV has it (standing / sitting / lying / undefined)
      breed           from the image folder, e.g. Maltese_dog (not used for balancing, kept for analysis)
      source_member   path inside images.tar.gz, for traceability
      orig_mode       PIL mode of the source image (RGB, L, CMYK, ...) before conversion
      width, height   size of the SAVED image, after any EXIF rotation
- data/raw/skipped.csv            every tar entry that was NOT saved, with a reason
- data/raw/download_summary.json  label names, counts, skip reasons, save actions
- results/debug/01_class_counts.png       bar chart of images per label
- results/debug/01_image_sizes.png        histogram of the shorter image side
- results/debug/01_samples_by_class.jpg   8 random images per label, captioned with id, breed, size

Run
---
    python scripts/01_download.py                 # full run (~20.4k images)
    python scripts/01_download.py --limit 50      # smoke test: random 50 labeled images (seeded)
    python scripts/01_download.py --overwrite     # rewrite JPEGs that already exist
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `trickforTreat` importable

from PIL import Image, ImageOps  # noqa: E402
from tqdm import tqdm  # noqa: E402

from trickforTreat.common import (  # noqa: E402
    add_common_args, bar_chart, header, histogram, load_config, print_table, read_csv, rel,
    resolve, safe_filename, save_contact_sheet, save_json, set_seed, setup_logging, write_csv,
)

# From the dataset card, shown for comparison only. The card's standing/lying counts appear swapped.
CARD_COUNTS = {"standing": 4143, "sitting": 3038, "lying": 7090, "undefined": 6307}
CARD_TOTAL = 20578
# Measured from the CSVs on 2026-11-12. A full run that doesn't reproduce these warns loudly.
EXPECTED_CSV_ROWS = 20423

MANIFEST_FIELDS = ["image_id", "path", "original_label", "breed", "source_member",
                   "orig_mode", "width", "height"]
SKIPPED_FIELDS = ["member", "reason"]
EXIF_ORIENTATION_TAG = 0x0112


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download DogPoseCV and write labeled images + manifest.")
    add_common_args(p)
    p.add_argument("--overwrite", action="store_true", help="rewrite JPEGs that already exist")
    return p.parse_args()


def is_macos_junk(name: str) -> bool:
    base = PurePosixPath(name).name
    return base.startswith("._") or base == ".DS_Store" or "__MACOSX" in name


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download(cfg, log) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download
    header("1. Download tarballs from Hugging Face")
    cache = str(resolve(cfg.paths.hf_cache_dir))
    paths = []
    for f in (cfg.dataset.labels_file, cfg.dataset.images_file):
        p = Path(hf_hub_download(cfg.dataset.hf_repo, f, repo_type="dataset", cache_dir=cache))
        log.info("%-22s -> %s (%.1f MB)", f, rel(p), p.stat().st_size / 1e6)
        paths.append(p)
    return paths[0], paths[1]


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------

def load_labels(labels_tar: Path, log) -> tuple[dict[str, str], dict[str, str]]:
    """Returns ({image_id: label}, {image_id: csv_breed}). Asserts the CSV format we rely on."""
    header("2. Parse label CSVs")
    labels, csv_breed = {}, {}
    n_csv = 0
    with tarfile.open(labels_tar, "r:gz") as t:
        for m in t.getmembers():
            if not m.isfile() or is_macos_junk(m.name) or not m.name.endswith(".csv"):
                continue
            n_csv += 1
            breed_key = PurePosixPath(m.name).stem  # e.g. n02085936-Maltese_dog
            rows = list(csv.reader(io.StringIO(t.extractfile(m).read().decode("utf-8"))))
            assert rows and rows[0] == ["id", "label"], f"{m.name}: unexpected header {rows[:1]}"
            for r in rows[1:]:
                if not r:
                    continue
                assert len(r) == 2, f"{m.name}: malformed row {r}"
                fname, label = r[0].strip(), r[1].strip()
                assert fname.lower().endswith(".jpg"), f"{m.name}: id {fname!r} is not a .jpg name"
                image_id = fname[:-4]
                assert image_id not in labels, f"{image_id} is labeled in two CSVs"
                # Stanford Dogs ids start with the breed's WordNet synset, so they must agree.
                assert image_id.split("_")[0] == breed_key.split("-")[0], \
                    f"{m.name}: {image_id} belongs to a different breed"
                labels[image_id] = label
                csv_breed[image_id] = breed_key
    assert labels, "no labels parsed"
    log.info("parsed %d label CSVs, %d labeled images", n_csv, len(labels))
    return labels, csv_breed


def print_label_names(labels: dict[str, str], cfg) -> list[str]:
    header("3. EXACT label names in the CSVs (must match configs/default.yaml -> labels)")
    counts = Counter(labels.values())
    names = sorted(counts)
    for name in names:
        print(f"  {name!r}: {counts[name]:,} rows")  # repr() makes stray spaces / case visible
    configured = set(cfg.labels.mapping) | set(cfg.labels.dropped_original_labels)
    missing, unknown = sorted(configured - set(names)), sorted(set(names) - configured)
    if missing or unknown:
        print("\n  !! CONFIG MISMATCH. Fix configs/default.yaml before running 02:")
        if missing:
            print(f"     config labels not in the data: {missing}")
        if unknown:
            print(f"     data labels not mentioned in the config: {unknown}")
    else:
        print("\n  OK: labels.mapping + labels.dropped_original_labels cover exactly these names.")
    return names


# ---------------------------------------------------------------------------
# 3. Images
# ---------------------------------------------------------------------------

def choose_ids(labels: dict[str, str], cfg, log) -> set[str]:
    """All labeled ids, or a seeded random sample with --limit.
    Random, not first-N: the tar is ordered by breed, so first-N would test only one breed."""
    ids = sorted(labels)
    limit = cfg.debug.limit_images
    if limit and limit < len(ids):
        log.info("--limit %d: random sample of %d / %d labeled images (seed %d)",
                 limit, limit, len(ids), cfg.seed)
        return set(random.Random(cfg.seed).sample(ids, limit))
    return set(ids)


def save_one(raw: bytes, out_path: Path, quality: int, overwrite: bool) -> tuple[int, int, str, str]:
    """Write one image as an RGB JPEG. Returns (width, height, orig_mode, action).
    action: "copied" (original JPEG bytes kept), "converted" (re-encoded to RGB JPEG),
    "rotated" (EXIF orientation baked into the pixels), or "skipped" (file already existed)."""
    im = Image.open(io.BytesIO(raw))
    orig_mode, fmt = im.mode, im.format
    if out_path.exists() and not overwrite:
        with Image.open(out_path) as saved:
            return saved.width, saved.height, orig_mode, "skipped"
    orientation = im.getexif().get(EXIF_ORIENTATION_TAG, 1)
    if orientation != 1:
        # Bake the rotation in so every later reader (PIL, YOLO) sees the same pixels.
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.save(out_path, "JPEG", quality=quality)
        action = "rotated"
    elif fmt == "JPEG" and orig_mode == "RGB":
        out_path.write_bytes(raw)  # lossless: no re-encoding
        action = "copied"
    else:
        im = im.convert("RGB")  # CMYK / L / RGBA / P -> RGB
        im.save(out_path, "JPEG", quality=quality)
        action = "converted"
    return im.width, im.height, orig_mode, action


def save_images(images_tar: Path, labels, csv_breed, wanted: set[str], label_names, cfg, overwrite, log):
    """Stream the image tar once, save wanted labeled images, and record why everything else was skipped."""
    header("4. Stream images.tar.gz and save labeled images")
    raw_dir = resolve(cfg.paths.raw_dir)
    for name in label_names:
        (raw_dir / safe_filename(name)).mkdir(parents=True, exist_ok=True)
    skip_folders = set(cfg.dataset.skip_folders)

    rows, skipped, actions = [], [], Counter()
    seen: dict[str, str] = {}
    with tarfile.open(images_tar, "r|gz") as t, tqdm(desc="tar entries", unit=" entries") as bar:
        for m in t:  # streaming mode: each member must be read before moving on
            bar.update()
            if not m.isfile():
                continue
            parts = PurePosixPath(m.name).parts  # ("images", "<synset>-<breed>", "<id>.jpg")
            folder = parts[1] if len(parts) >= 3 else "(root)"
            stem = PurePosixPath(m.name).stem
            if is_macos_junk(m.name):
                reason = "macos_metadata"
            elif not m.name.lower().endswith(".jpg"):
                reason = "not_jpg"
            elif folder in skip_folders:
                reason = f"skip_folder:{folder}"
            elif stem not in labels:
                reason = "no_label"
            elif stem not in wanted:
                continue  # --limit: labeled but not sampled. Not a "skip" worth logging.
            else:
                reason = None
            if reason:
                skipped.append({"member": m.name, "reason": reason})
                continue

            assert stem not in seen, f"{stem} appears twice: {seen[stem]} and {m.name}"
            seen[stem] = m.name
            assert folder == csv_breed[stem], f"{m.name} is in folder {folder} but labeled in {csv_breed[stem]}.csv"
            label = labels[stem]
            out = raw_dir / safe_filename(label) / f"{stem}.jpg"
            w, h, mode, action = save_one(t.extractfile(m).read(), out, cfg.dataset.jpeg_quality, overwrite)
            actions[action] += 1
            rows.append({"image_id": stem, "path": rel(out), "original_label": label,
                         "breed": folder.split("-", 1)[1], "source_member": m.name,
                         "orig_mode": mode, "width": w, "height": h})

    print_table(sorted(actions.items()), ["save action", "images"])
    print("  copied = original JPEG kept byte-for-byte; converted = non-RGB/non-JPEG source re-encoded;")
    print("  rotated = EXIF orientation applied; skipped = already on disk from an earlier run\n")
    by_reason = Counter(s["reason"] for s in skipped)
    print_table(sorted(by_reason.items()), ["not saved, reason", "tar entries"])
    no_label_folders = Counter(PurePosixPath(s["member"]).parts[1] for s in skipped
                               if s["reason"] == "no_label")
    if no_label_folders:
        print("\n  'no_label' images by folder:")
        print_table(no_label_folders.most_common(), ["folder", "images"])
    return rows, skipped, actions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_class_counts(rows, labels, label_names, limited: bool) -> Counter:
    header("5. Class counts")
    saved = Counter(r["original_label"] for r in rows)
    in_csv = Counter(labels.values())
    total = sum(saved.values())
    table = [(n, in_csv[n], saved[n], f"{100 * saved[n] / total:.1f}%", CARD_COUNTS.get(n, "?"))
             for n in label_names]
    table.append(("TOTAL", sum(in_csv.values()), total, "100%", CARD_TOTAL))
    print_table(table, ["label", "CSV rows", "saved", "share", "dataset card"])
    if limited:
        print("  (--limit run: 'saved' is a random sample)")
    print("  NOTE: the card's standing/lying counts look swapped relative to the CSVs. The CSV labels")
    print("        were checked visually and match the images, so the CSVs are treated as the truth.")
    print("        Card total 20,578 minus 155 unlabeled Standard Schnauzer images = 20,423.")
    breeds = Counter(r["breed"] for r in rows)
    print(f"\n  breeds: {len(breeds)}   images per breed: min {min(breeds.values())}, "
          f"max {max(breeds.values())}")
    short = sorted(min(int(r["width"]), int(r["height"])) for r in rows)
    n_small = sum(s < 224 for s in short)
    print(f"  shorter image side: min {short[0]}, median {short[len(short) // 2]}, max {short[-1]}; "
          f"{n_small:,} images ({100 * n_small / len(short):.1f}%) are under 224 px")
    print("  (the dog crop in 04 is smaller than the whole image, so more crops than this will be upscaled)")
    return saved


def save_debug_visuals(rows, label_names, counts, cfg, limited: bool, log) -> None:
    header("6. Debug visuals")
    dbg = resolve(cfg.debug.debug_dir)
    paths = [
        bar_chart(label_names, [counts[n] for n in label_names], dbg / "01_class_counts.png",
                  title="Images per original label" + (" (--limit sample)" if limited else "")),
        histogram([min(int(r["width"]), int(r["height"])) for r in rows], dbg / "01_image_sizes.png",
                  title="Shorter image side", xlabel="pixels", vlines={"224 (classifier input)": 224}),
    ]
    rng = random.Random(cfg.seed)
    items = []
    for name in label_names:
        pool = [r for r in rows if r["original_label"] == name]
        for r in rng.sample(pool, min(8, len(pool))):
            items.append((r["path"], f"{name}\n{r['image_id']}\n{r['breed'][:20]}\n{r['width']}x{r['height']}"))
    paths.append(save_contact_sheet(items, dbg / "01_samples_by_class.jpg", cols=8,
                                    title="8 random images per original label (one row per label)"))
    for p in paths:
        log.info("wrote %s", rel(p))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(rows, skipped, labels, wanted, label_names, cfg, limited: bool, log) -> None:
    header("7. Sanity checks")
    assert rows, "no rows were written"

    saved_ids = {r["image_id"] for r in rows}
    assert len(saved_ids) == len(rows), "duplicate image_ids"
    assert saved_ids == wanted, (f"{len(wanted - saved_ids)} labeled images were never found in the tar, "
                                 f"e.g. {sorted(wanted - saved_ids)[:5]}")
    print(f"  [ok] every selected labeled image was found and saved once ({len(rows):,})")
    if not limited:
        if len(labels) != EXPECTED_CSV_ROWS:
            log.warning("CSV row count %d != %d measured on 2026-11-12. The HF dataset changed upstream.",
                        len(labels), EXPECTED_CSV_ROWS)
        else:
            print(f"  [ok] labeled image count matches the {EXPECTED_CSV_ROWS:,} measured when this was written")

    for r in rows:
        assert r["original_label"] in label_names, f"unknown label {r['original_label']!r}"
        assert r["original_label"] == labels[r["image_id"]], f"label changed in transit: {r}"
        assert int(r["width"]) > 0 and int(r["height"]) > 0, f"bad size: {r}"
        assert resolve(r["path"]).is_file(), f"missing file {r['path']}"
        assert Path(r["path"]).parent.name == safe_filename(r["original_label"]), f"wrong folder: {r}"
    print("  [ok] every label is known and unchanged, every file exists in its label folder")

    if not limited:
        assert {r["original_label"] for r in rows} == set(label_names), "some label has zero images"
        print("  [ok] every label has at least one image")
    assert not any(s["reason"] == "no_label" and PurePosixPath(s["member"]).stem in labels for s in skipped)
    print("  [ok] no labeled image was skipped by mistake")

    # Re-open a random sample from disk: what 02-04 read must match what the manifest says.
    sample = random.Random(cfg.seed).sample(rows, min(100, len(rows)))
    for r in sample:
        with Image.open(resolve(r["path"])) as im:
            assert im.mode == "RGB", f"{r['path']} saved as {im.mode}, expected RGB"
            assert im.size == (int(r["width"]), int(r["height"])), f"{r['path']} size != manifest"
    print(f"  [ok] {len(sample)} random saved images reopen as RGB with the manifest's size")

    on_disk = set(resolve(cfg.paths.raw_dir).glob("*/*.jpg"))
    stale = on_disk - {resolve(r["path"]) for r in rows}
    if stale and not limited:
        log.warning("%d JPEGs under %s are not in the manifest (left over from an earlier run?). "
                    "Later scripts only read the manifest, but consider deleting them.",
                    len(stale), cfg.paths.raw_dir)
    else:
        print(f"  [ok] {len(on_disk) - len(stale):,} manifest files on disk"
              + (f" (+{len(stale)} from earlier runs, expected with --limit)" if stale else ""))

    back = read_csv(cfg.paths.raw_manifest)
    assert len(back) == len(rows) and list(back[0]) == MANIFEST_FIELDS, "manifest did not round-trip"
    print("  [ok] manifest.csv re-reads with the expected columns")
    print("\n  All sanity checks passed.")


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, limit=args.limit)
    log = setup_logging("01_download", cfg.debug.log_level)
    set_seed(cfg.seed)
    limited = bool(cfg.debug.limit_images)

    labels_tar, images_tar = download(cfg, log)
    labels, csv_breed = load_labels(labels_tar, log)
    label_names = print_label_names(labels, cfg)
    wanted = choose_ids(labels, cfg, log)
    rows, skipped, actions = save_images(images_tar, labels, csv_breed, wanted, label_names, cfg,
                                         args.overwrite, log)
    log.info("wrote %s (%d rows)", rel(write_csv(rows, cfg.paths.raw_manifest, MANIFEST_FIELDS)), len(rows))
    log.info("wrote %s (%d rows)", rel(write_csv(skipped, resolve(cfg.paths.raw_dir) / "skipped.csv",
                                                 SKIPPED_FIELDS)), len(skipped))
    counts = print_class_counts(rows, labels, label_names, limited)
    summary = save_json({
        "hf_repo": cfg.dataset.hf_repo, "label_names": label_names,
        "csv_label_counts": dict(Counter(labels.values())), "saved_counts": dict(counts),
        "skip_reasons": dict(Counter(s["reason"] for s in skipped)), "save_actions": dict(actions),
        "limited": cfg.debug.limit_images,
    }, resolve(cfg.paths.raw_dir) / "download_summary.json")
    log.info("wrote %s", rel(summary))

    if cfg.debug.save_debug_grid:
        save_debug_visuals(rows, label_names, counts, cfg, limited, log)
    sanity_checks(rows, skipped, labels, wanted, label_names, cfg, limited, log)

    header("NEXT STEP")
    print("  1. Section 3 must say OK (config label names match the CSVs exactly).")
    print("  2. Open results/debug/01_samples_by_class.jpg: each row should look like its label.")
    print("  3. Then run: python scripts/02_filter_labels.py")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------------------------------
# What it prints, in order:
#   1. Where the two tarballs were cached, and their sizes.
#   2. How many label CSVs and labeled images were parsed. The parser asserts the header is
#      `id,label`, ids end in .jpg, no id is labeled twice, and each id's synset matches its CSV.
#   3. The EXACT label strings (repr'd) with counts, and whether the config covers them.
#   4. Save actions, and a table of every tar entry NOT saved, by reason:
#        macos_metadata      ._* and .DS_Store files from the uploader's Mac (expected, ~20.8k)
#        skip_folder:validation   the extra images/validation/ folder (dataset.skip_folders)
#        no_label            images with no CSV row (Standard Schnauzer has no CSV at all)
#        not_jpg             anything else
#   5. Class counts: CSV rows vs. saved vs. the dataset card.
#   6. Paths of the debug images.
#   7. Sanity checks, each printed as [ok]. Any failure stops the script with an assert message.
#
# Intermediates:
#   data/hf_cache/                     Downloaded tarballs (~780 MB). Safe to delete; re-download takes minutes.
#   data/raw/<label>/*.jpg             One JPEG per labeled image, foldered by ORIGINAL label string.
#   data/raw/manifest.csv              Source of truth for 02. Later scripts read only this, not the folders.
#   data/raw/skipped.csv               Every tar entry not saved, with its reason. grep it to find a file.
#   data/raw/download_summary.json     Counts, skip reasons, save actions, for the record.
#   results/debug/01_class_counts.png  Images per label.
#   results/debug/01_image_sizes.png   Images left of the 224 line will give soft crops in 04.
#   results/debug/01_samples_by_class.jpg  One row per label. "undefined" should be mostly close-up faces.
#
# Common failures:
#   - "appears twice": a stem exists in two non-skipped folders. Add the extra folder to dataset.skip_folders.
#   - "labeled images were never found in the tar": the CSVs reference files the tar doesn't have.
#     The upstream dataset changed; look at the listed ids.
#   - Smoke test: `--limit 50` still streams the whole tar (it's gzip, so no random access),
#     which takes a minute or two, but it only writes 50 images.
