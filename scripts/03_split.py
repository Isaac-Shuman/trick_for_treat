"""
03_split.py  (PSEUDOCODE, not runnable yet)

Purpose
-------
1. Drop near-duplicate images (perceptual hash), keeping ONE image per duplicate group at random
   (seeded), even if the group's images have different labels.
2. Carve out a stratified, seeded held-out TEST set (default 15%) from what remains. Everything
   else is "trainval", which 05 splits further with K-fold CV. The test set is touched exactly ONCE:
   final evaluation in 05 after model and threshold selection are done.

Dedup runs BEFORE the split so near-identical photos can't land on both sides of train/test
(or on both sides of a CV fold), which would inflate scores.

The split is at the IMAGE level, BEFORE cropping, so a crop always inherits its source image's
split. Note: 04 later drops some images (no dog, multiple dogs, tiny box) from both splits, so the
final test fraction will be slightly under 15%. That's expected and harmless.

Inputs
------
- data/filtered_manifest.csv  (from 02)
- configs/default.yaml        (dedup.*, split.*, seed)

Outputs
-------
- data/duplicates_dropped.csv: image_id, label, kept_image_id, kept_label, hamming_distance, group_id
- data/split_manifest.csv = deduped filtered manifest + columns `phash` and `split` in {"trainval", "test"}
- data/test_ids.txt       = sorted test image_ids, one per line
- stdout: dedup summary, per-class counts per split

Run
---
    python scripts/03_split.py --config configs/default.yaml
"""

# import argparse, random
# import numpy as np, imagehash
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from trickforTreat.common import load_config, set_seed, read_csv, write_csv


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def compute_hashes(rows, cfg) -> "np.ndarray":
    """
    For each row: h = imagehash.phash(Image.open(path).convert("RGB"), hash_size=cfg.dedup.hash_size)
    Pack into a uint64 array [N] (hash_size 8 -> 64 bits). Store hex string in row["phash"].
    ~14k images: takes a minute or two on a laptop.
    """
    # TODO


def find_duplicate_groups(hashes, max_dist) -> list[list[int]]:
    """
    All-pairs Hamming distance, vectorized in chunks to bound memory:
        for each chunk of rows i: xor = hashes[i, None] ^ hashes[None, :]; dist = popcount(xor)
        ~14k^2 / 2 = 1e8 comparisons; fine with numpy (np.bitwise_count on numpy >= 2.0,
        else a byte lookup table).
    Every pair (i, j), i < j, with dist <= max_dist is an edge. Union-find over edges -> connected
    components. Return only components with size >= 2.
    NOTE: union-find is transitive (A~B, B~C groups A,B,C even if A and C are far apart). That's
          fine for "same photo shoot" clusters. If one giant group appears, max_hamming_distance is
          too loose (see debugging guide).
    """
    # TODO


def drop_duplicates(rows, groups, cfg) -> tuple[list[dict], list[dict]]:
    """
    rng = random.Random(cfg.seed)
    drop = set(); log = []
    for gid, g in enumerate(groups):
        keep = rng.choice(sorted(g))           # sorted so the choice is deterministic given the seed
        for idx in g:
            if idx != keep:
                drop.add(idx)
                log.append({image_id, label, kept_image_id, kept_label, hamming_distance(idx, keep), group_id=gid})
    return [r for i, r in enumerate(rows) if i not in drop], log
    """
    # TODO


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_rows(rows, cfg) -> tuple[list[dict], list[dict]]:
    """
    ids = [r["image_id"] for r in rows]
    trainval_ids, test_ids = train_test_split(
        ids, test_size=cfg.split.test_fraction,
        stratify=[r["label"] for r in rows],
        random_state=cfg.seed, shuffle=True)
    tag each row with split = "test" if id in test_ids else "trainval"
    return trainval_rows, test_rows
    """
    # TODO


def sanity_checks(rows_in, rows_dedup, dup_log, groups, trainval, test, cfg) -> None:
    """
    Dedup:
      assert len(rows_dedup) + len(dup_log) == len(rows_in)
      assert len(dup_log) == sum(len(g) - 1 for g in groups)     # exactly one kept per group
      assert no dropped id appears in rows_dedup; every kept_image_id does
      assert all pairwise distances among rows_dedup > max_dist   (spot-check a random 2k subset)
      warn if dropped fraction > 5%: threshold likely too loose
      warn if largest group > 20 images: suspicious (e.g. many blank/placeholder images)
    Split:
      assert set(trainval ids).isdisjoint(test ids)
      assert len(trainval) + len(test) == len(rows_dedup)
      assert abs(len(test)/len(rows_dedup) - cfg.split.test_fraction) < 0.01
      for each class c:
          assert abs(frac_c(test) - frac_c(rows_dedup)) < 0.01      # stratification held
          assert count_c(test) >= 30, "too few test samples to estimate precision"
    """
    # TODO


def main():
    """
    cfg = load_config(...); set_seed(cfg.seed)
    rows = read_csv(cfg.paths.filtered_manifest)
    hashes = compute_hashes(rows, cfg)
    groups = find_duplicate_groups(hashes, cfg.dedup.max_hamming_distance)
    rows_dedup, dup_log = drop_duplicates(rows, groups, cfg)
    write_csv(dup_log, cfg.paths.dedup_log)
    print dedup summary: n_groups, n_dropped per label, n cross-label groups, largest group size
    trainval, test = split_rows(rows_dedup, cfg)
    write_csv(trainval + test sorted by image_id, cfg.paths.split_manifest)
    write sorted test ids to cfg.paths.test_ids
    print a table: class | trainval | test | test %
    sanity_checks(...)
    debug grid (if enabled): results/debug/03_duplicate_groups.jpg (up to 16 groups, one row each,
        kept image first, captioned with label + hamming distance)
    """
    # TODO


# if __name__ == "__main__":
#     main()


# ---------------------------------------------------------------------------
# DEBUGGING GUIDE
# ---------------------------------------------------------------------------
# Intermediates:
#   data/duplicates_dropped.csv  Every image removed as a near-duplicate, and which image was kept.
#   data/split_manifest.csv      Deduped rows + `phash` + `split`. 04 and 05 read this.
#   data/test_ids.txt            The held-out test set.
#   results/debug/03_duplicate_groups.jpg   Visual check of what counted as a duplicate.
#
# How to check:
#   1. Open 03_duplicate_groups.jpg. Each row should look like the same photo (resized, recompressed,
#      slightly cropped). If rows contain clearly different photos, lower max_hamming_distance
#      (try 2). If you can spot obvious duplicates that weren't grouped, raise it (try 6-8).
#   2. Expect a small dropped fraction (low single-digit %). Much higher means the threshold is too loose.
#   3. Printed split table: each class ~15% in test.
#
# Rules:
#   - 05 must filter `split == "trainval"` for CV and only load `split == "test"` in final_eval().
#   - Re-running with the same seed and data gives the same split. If the HF dataset changes
#     upstream (we don't pin it), the split will change too.
