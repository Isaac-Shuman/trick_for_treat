"""
trickforTreat/common.py

Small helpers used by every script:
  Config / load_config   YAML config with attribute access (cfg.paths.raw_dir) and clear errors on typos
  resolve / rel          repo-relative path handling (manifests store repo-relative POSIX paths)
  set_seed, seed_worker  reproducible randomness
  pick_device            cuda > mps > cpu (torch imported lazily)
  read_csv / write_csv   manifests, with column checks and deterministic row order
  safe_filename          "lying down" -> "lying_down" for folder names
  header / print_table   readable stdout sections for debugging
  save_contact_sheet     image grids for visual debugging (results/debug/*.jpg)
  bar_chart / histogram  simple static debug plots (results/debug/*.png)

trickforTreat/transforms.py (later) holds the training augmentations, kept apart from
classifier.py because the live pipeline never needs them.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

REQUIRED_SECTIONS = (
    "seed", "paths", "dataset", "labels", "dedup", "split", "detector",
    "crop", "classifier", "train", "eval", "augment", "debug",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config(dict):
    """A dict with recursive attribute access. A missing key raises an error listing the valid keys."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"config has no key {name!r}; available keys: {sorted(self)}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}


def _wrap(v: Any) -> Any:
    if isinstance(v, dict) and not isinstance(v, Config):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def load_config(path: str | Path = DEFAULT_CONFIG, limit: int | None = None) -> Config:
    """Load the YAML config. `limit` (from --limit) overrides debug.limit_images."""
    path = resolve(path)
    assert path.is_file(), f"config file not found: {path}"
    with open(path) as f:
        raw = yaml.safe_load(f)
    assert isinstance(raw, dict), f"config {path} did not parse to a mapping"
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    assert not missing, f"config {path} is missing sections: {missing}"
    cfg = Config(raw)
    if limit is not None:
        assert limit > 0, f"--limit must be positive, got {limit}"
        cfg.debug.limit_images = limit
    return cfg


def add_common_args(parser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to the YAML config")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test: only process N images (overrides debug.limit_images)")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve(p: str | Path) -> Path:
    """Config and manifest paths are relative to the repo root, wherever you run the script from."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def rel(p: str | Path) -> str:
    """Repo-relative POSIX path string, for storing in manifests (absolute if outside the repo)."""
    p = Path(p).resolve()
    return p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else p.as_posix()


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(s: str) -> str:
    out = _SAFE_RE.sub("_", str(s).strip()).strip("_")
    assert out, f"safe_filename produced an empty name from {s!r}"
    return out


# ---------------------------------------------------------------------------
# Seeding / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects subprocesses (e.g. DataLoader workers)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's per-worker seed."""
    import numpy as np
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Logging / stdout formatting
# ---------------------------------------------------------------------------

def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "matplotlib", "PIL", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def header(title: str) -> None:
    bar = "=" * max(60, len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def print_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> None:
    cells = [[str(h) for h in headers]] + [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.rjust(w) if j and _isnum(c) else c.ljust(w) for c, w in zip(r, widths)))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


def _fmt(c: Any) -> str:
    return f"{c:.3f}" if isinstance(c, float) else str(c)


def _isnum(s: str) -> bool:
    return bool(re.fullmatch(r"-?[\d.,%]+", s))


# ---------------------------------------------------------------------------
# CSV / JSON
# ---------------------------------------------------------------------------

def read_csv(path: str | Path) -> list[dict]:
    path = resolve(path)
    assert path.is_file(), f"{path} not found. Did you run the previous script?"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{path} has no rows"
    return rows


def write_csv(rows: Iterable[dict], path: str | Path, fieldnames: Sequence[str]) -> Path:
    """Writes rows sorted by image_id (when present) so re-runs produce identical files."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    for r in rows:
        assert set(r) == set(fieldnames), f"row keys {sorted(r)} != expected columns {sorted(fieldnames)}"
    if rows and "image_id" in fieldnames:
        rows.sort(key=lambda r: r["image_id"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Visual debugging
# ---------------------------------------------------------------------------

def _font(size: int = 12):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def save_contact_sheet(items: Sequence[tuple[Any, str]], path: str | Path, cols: int = 8,
                       thumb: int = 160, title: str | None = None) -> Path:
    """
    items: (PIL.Image or image path, caption) pairs. Captions may contain "\n".
    Each image is shrunk to fit a thumb x thumb tile (aspect kept, gray letterbox) with its
    caption underneath, so you can see exactly which image_id each tile is.
    """
    from PIL import Image, ImageDraw

    assert items, "save_contact_sheet called with no items"
    font, title_font = _font(11), _font(16)
    cap_lines = max(c.count("\n") + 1 for _, c in items)
    cap_h = 14 * cap_lines + 4
    title_h = 28 if title else 0
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, title_h + rows * (thumb + cap_h)), (252, 252, 251))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 6), title, fill=(11, 11, 11), font=title_font)
    for k, (img, cap) in enumerate(items):
        if not hasattr(img, "size"):
            img = Image.open(resolve(img))
        img = img.convert("RGB")
        img.thumbnail((thumb - 4, thumb - 4))
        x0, y0 = (k % cols) * thumb, title_h + (k // cols) * (thumb + cap_h)
        draw.rectangle([x0 + 1, y0 + 1, x0 + thumb - 2, y0 + thumb - 2], fill=(200, 200, 200))
        sheet.paste(img, (x0 + (thumb - img.width) // 2, y0 + (thumb - img.height) // 2))
        draw.multiline_text((x0 + 3, y0 + thumb + 1), cap, fill=(82, 81, 78), font=font, spacing=2)
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return path


# Debug-plot style: one blue series on a near-white surface, recessive grid and axes.
_SURFACE, _INK, _INK2, _SERIES1, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"


def _styled_axes(figsize=(7, 4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax, plt


def bar_chart(labels: Sequence[str], values: Sequence[float], path: str | Path, title: str,
              ylabel: str = "count", reference: Sequence[float] | None = None,
              reference_label: str = "expected") -> Path:
    """Vertical bars with value labels. `reference` draws a dark tick per bar (e.g. expected counts)."""
    fig, ax, plt = _styled_axes()
    xs = range(len(labels))
    bars = ax.bar(xs, values, width=0.6, color=_SERIES1, label="observed")
    tops = [max(v, r) for v, r in zip(values, reference)] if reference is not None else list(values)
    pad = 0.02 * max(tops) if tops and max(tops) > 0 else 0.1
    for b, v, top in zip(bars, values, tops):
        ax.text(b.get_x() + b.get_width() / 2, top + pad, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=9, color=_INK)
    ax.set_ylim(0, max(tops) * 1.12 if tops and max(tops) > 0 else 1)
    if reference is not None:
        ax.scatter(xs, reference, marker="_", s=900, linewidths=2, color=_INK, label=reference_label, zorder=3)
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK2, ncol=2,
                  loc="lower right", bbox_to_anchor=(1.0, 1.0))
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel(ylabel, color=_INK2, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def histogram(values: Sequence[float], path: str | Path, title: str, xlabel: str, bins: int = 50,
              vlines: dict[str, float] | None = None, clip_pct: float | None = 99.5) -> Path:
    """Single-series histogram. `vlines` marks named values (e.g. {"min crop side": 32}).
    Values above the `clip_pct` percentile are left out so a few outliers don't squash the plot;
    the title says how many were left out."""
    vals = sorted(values)
    assert vals, "histogram called with no values"
    if clip_pct is not None:
        cut = vals[min(len(vals) - 1, int(len(vals) * clip_pct / 100))]
        n_out = sum(v > cut for v in vals)
        vals = [v for v in vals if v <= cut]
        if n_out:
            title = f"{title}  ({n_out} values > {cut:g} not shown)"
    fig, ax, plt = _styled_axes()
    ax.hist(vals, bins=bins, color=_SERIES1, edgecolor=_SURFACE, linewidth=1)
    for name, x in (vlines or {}).items():
        ax.axvline(x, color=_INK, linewidth=1.5, linestyle="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {name}", color=_INK, fontsize=8, va="top")
    ax.set_xlabel(xlabel, color=_INK2, fontsize=9)
    ax.set_ylabel("images", color=_INK2, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
