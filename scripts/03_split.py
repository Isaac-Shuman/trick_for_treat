#!/usr/bin/env python3
"""
03_split.py

1. Drop near-duplicate images. A perceptual hash (phash) is computed per image; any two images
   within `dedup.max_hamming_distance` bits are linked, linked images form groups (union-find),
   and ONE image per group is kept at random (seeded), even if the group's labels differ.
2. Carve out a stratified, seeded held-out TEST set (split.test_fraction). Everything else is
   "trainval", which 05 splits further with K-fold CV. 05 touches the test set exactly once.

Dedup runs BEFORE the split so near-identical photos can't land on both sides of train/test.
The split is at the image level, before cropping; 04 later drops some images (no dog, several
dogs, tiny box) from both sides, so the final test fraction ends up slightly under 15%.

Inputs:  data/filtered_manifest.csv (02), configs/default.yaml (dedup.*, split.*, seed)
Outputs:
  data/duplicates_dropped.csv   image_id, label, kept_image_id, kept_label, hamming_distance, group_id
  data/split_manifest.csv       deduped rows + `phash` + `split` (trainval | test)
  data/test_ids.txt             sorted test image_ids
  results/debug/03_nearest_neighbor_distance.png   how close each image's nearest other image is
  results/debug/03_duplicate_groups.jpg            what counted as a duplicate (kept image first)

Run:  python scripts/03_split.py
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tqdm import tqdm  # noqa: E402

from trickforTreat.common import (  # noqa: E402
    add_common_args, header, histogram, load_config, print_table, read_csv, rel, resolve,
    save_contact_sheet, set_seed, setup_logging, write_csv,
)

IN_FIELDS = ["image_id", "path", "original_label", "breed", "source_member", "orig_mode",
             "width", "height", "label", "label_id"]
OUT_FIELDS = IN_FIELDS + ["phash", "split"]
DUP_FIELDS = ["image_id", "label", "kept_image_id", "kept_label", "hamming_distance", "group_id"]


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _phash(args: tuple[str, int]) -> int:
    import imagehash
    from PIL import Image
    path, hash_size = args
    with Image.open(resolve(path)) as im:
        h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
    return int(str(h), 16)


def compute_hashes(rows, hash_size: int) -> np.ndarray:
    assert hash_size == 8, "pairwise search packs hashes into uint64, so hash_size must be 8"
    with Pool() as pool:
        hashes = list(tqdm(pool.imap(_phash, [(r["path"], hash_size) for r in rows], chunksize=64),
                           total=len(rows), desc="phash", unit="img"))
    return np.array(hashes, dtype=np.uint64)


def pairwise_close_pairs(hashes: np.ndarray, max_dist: int) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """All pairs (i, j, dist) with i < j and Hamming distance <= max_dist, plus each image's
    nearest-neighbor distance. Vectorized in row chunks: ~1e8 comparisons for 14k images."""
    n = len(hashes)
    pairs, nn = [], np.full(n, 64, dtype=np.int64)
    for start in tqdm(range(0, n, 512), desc="pairwise", unit="chunk"):
        block = hashes[start:start + 512]
        d = np.bitwise_count(block[:, None] ^ hashes[None, :]).astype(np.int64)  # [b, n]
        idx = np.arange(start, start + len(block))
        d[np.arange(len(block)), idx] = 64  # ignore self
        nn[idx] = d.min(axis=1)
        ii, jj = np.nonzero(d <= max_dist)
        for a, j in zip(ii, jj):
            i = start + a
            if i < j:
                pairs.append((int(i), int(j), int(d[a, j])))
    return pairs, nn


def union_find_groups(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def drop_duplicates(rows, hashes, groups, seed: int):
    rng = random.Random(seed)
    drop, log = set(), []
    keep_of = {}
    for gid, g in enumerate(groups):
        keep = rng.choice(g)
        keep_of[gid] = keep
        for i in g:
            if i != keep:
                drop.add(i)
                log.append({"image_id": rows[i]["image_id"], "label": rows[i]["label"],
                            "kept_image_id": rows[keep]["image_id"], "kept_label": rows[keep]["label"],
                            "hamming_distance": int(np.bitwise_count(hashes[i] ^ hashes[keep])),
                            "group_id": gid})
    return [r for i, r in enumerate(rows) if i not in drop], log, keep_of


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Dedup + stratified held-out test split.")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, limit=args.limit)
    log = setup_logging("03_split", cfg.debug.log_level)
    set_seed(cfg.seed)
    dbg = resolve(cfg.debug.debug_dir)

    rows = read_csv(cfg.paths.filtered_manifest)
    assert list(rows[0]) == IN_FIELDS, f"unexpected columns {list(rows[0])}; re-run 02"
    rows.sort(key=lambda r: r["image_id"])
    if cfg.debug.limit_images:
        rows = sorted(random.Random(cfg.seed).sample(rows, min(cfg.debug.limit_images, len(rows))),
                      key=lambda r: r["image_id"])
    n_in = len(rows)

    # ---- dedup -------------------------------------------------------------
    header("1. Near-duplicate detection (perceptual hash)")
    max_dist = int(cfg.dedup.max_hamming_distance)
    hashes = compute_hashes(rows, int(cfg.dedup.hash_size))
    for r, h in zip(rows, hashes):
        r["phash"] = f"{int(h):016x}"
    pairs, nn = pairwise_close_pairs(hashes, max_dist)
    groups = union_find_groups(len(rows), pairs)
    rows_dedup, dup_log, keep_of = drop_duplicates(rows, hashes, groups, cfg.seed)
    cross = sum(len({rows[i]["label"] for i in g}) > 1 for g in groups)
    sizes = Counter(len(g) for g in groups)
    print_table([
        ("images in", n_in), ("close pairs (<= %d bits)" % max_dist, len(pairs)),
        ("duplicate groups", len(groups)), ("  ...with mixed labels", cross),
        ("largest group", max(sizes) if sizes else 0),
        ("images dropped", len(dup_log)), ("images kept", len(rows_dedup)),
    ], ["dedup", "value"])
    print_table(sorted(Counter(d["label"] for d in dup_log).items()), ["dropped, by label", "images"])
    write_csv(dup_log, cfg.paths.dedup_log, DUP_FIELDS)

    # ---- split -------------------------------------------------------------
    header("2. Stratified held-out test split")
    ids = [r["image_id"] for r in rows_dedup]
    _, test_ids = train_test_split(ids, test_size=float(cfg.split.test_fraction),
                                   stratify=[r["label"] for r in rows_dedup],
                                   random_state=cfg.seed, shuffle=True)
    test_ids = set(test_ids)
    for r in rows_dedup:
        r["split"] = "test" if r["image_id"] in test_ids else "trainval"
    classes = sorted({r["label"] for r in rows_dedup})
    by = Counter((r["label"], r["split"]) for r in rows_dedup)
    print_table([(c, by[(c, "trainval")], by[(c, "test")],
                  f"{100 * by[(c, 'test')] / (by[(c, 'trainval')] + by[(c, 'test')]):.1f}%") for c in classes],
                ["class", "trainval", "test", "test %"])
    write_csv(rows_dedup, cfg.paths.split_manifest, OUT_FIELDS)
    resolve(cfg.paths.test_ids).write_text("\n".join(sorted(test_ids)) + "\n")
    log.info("wrote %s, %s, %s", cfg.paths.split_manifest, cfg.paths.test_ids, cfg.paths.dedup_log)

    # ---- visuals -----------------------------------------------------------
    if cfg.debug.save_debug_grid:
        header("3. Debug visuals")
        p1 = histogram(nn.tolist(), dbg / "03_nearest_neighbor_distance.png",
                       title="Hamming distance to each image's nearest other image", xlabel="bits (of 64)",
                       bins=int(nn.max()) + 1, vlines={f"dedup threshold ({max_dist})": max_dist + 0.5},
                       clip_pct=None)
        items = []
        for gid, g in enumerate(groups[:24]):
            keep = keep_of[gid]
            for i in [keep] + [i for i in g if i != keep][:3]:
                d = int(np.bitwise_count(hashes[i] ^ hashes[keep]))
                tag = "KEPT" if i == keep else f"dropped d={d}"
                items.append((rows[i]["path"], f"g{gid} {tag}\n{rows[i]['label']}\n{rows[i]['image_id']}"))
        if items:
            p2 = save_contact_sheet(items, dbg / "03_duplicate_groups.jpg", cols=8,
                                    title="Duplicate groups (first 24): kept image, then up to 3 dropped")
            log.info("wrote %s", rel(p2))
        log.info("wrote %s", rel(p1))

    # ---- sanity ------------------------------------------------------------
    header("4. Sanity checks")
    assert len(rows_dedup) + len(dup_log) == n_in
    assert len(dup_log) == sum(len(g) - 1 for g in groups), "not exactly one kept per group"
    kept_ids = {r["image_id"] for r in rows_dedup}
    assert not ({d["image_id"] for d in dup_log} & kept_ids) and {d["kept_image_id"] for d in dup_log} <= kept_ids
    print("  [ok] exactly one image kept per duplicate group; dropped ids are gone")
    kh = np.array([int(r["phash"], 16) for r in rows_dedup], dtype=np.uint64)
    smp = np.random.default_rng(cfg.seed).choice(len(kh), size=min(2000, len(kh)), replace=False)
    d = np.bitwise_count(kh[smp][:, None] ^ kh[None, :])
    d[np.arange(len(smp)), smp] = 64
    assert d.min() > max_dist, "kept images still contain near-duplicates"
    print(f"  [ok] no kept pair within {max_dist} bits (checked {len(smp)} images against all)")
    tv = kept_ids - test_ids
    assert tv.isdisjoint(test_ids) and len(tv) + len(test_ids) == len(rows_dedup)
    assert abs(len(test_ids) / len(rows_dedup) - float(cfg.split.test_fraction)) < 0.01
    total = Counter(r["label"] for r in rows_dedup)
    for c in classes:
        frac = by[(c, "test")] / total[c]
        assert abs(frac - float(cfg.split.test_fraction)) < 0.01, f"stratification off for {c}"
        if not cfg.debug.limit_images:
            assert by[(c, "test")] >= 30, f"too few test images for {c}"
    print("  [ok] train/test disjoint, test fraction and per-class stratification within 1%")
    if len(dup_log) / n_in > 0.05:
        log.warning("dropped %.1f%% as duplicates: threshold may be too loose", 100 * len(dup_log) / n_in)
    if sizes and max(sizes) > 20:
        log.warning("a duplicate group has %d images: look at 03_duplicate_groups.jpg", max(sizes))
    print("\n  All sanity checks passed. Next: python scripts/04_crop_dogs.py")


if __name__ == "__main__":
    main()

# DEBUGGING GUIDE
#   03_nearest_neighbor_distance.png  Real duplicates sit at 0-4 bits, unrelated photos around 20-35.
#       If there's no clear gap near the threshold line, re-tune dedup.max_hamming_distance.
#   03_duplicate_groups.jpg  Each group should look like the same photo. Different photos -> lower
#       the threshold. Obvious duplicates missed -> raise it.
#   data/duplicates_dropped.csv  every dropped image and the image kept in its place.
#   Rule: 05 uses split == "trainval" for CV and loads split == "test" only in its final evaluation.
