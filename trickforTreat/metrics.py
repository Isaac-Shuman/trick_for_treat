"""
trickforTreat/metrics.py

Classification metrics, the argmax-plus-threshold reward curve, threshold selection, and plots.
Pure numpy/sklearn/matplotlib, so the evaluation on the user's own dog videos can reuse it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .common import _INK, _INK2, _SERIES1, _styled_axes, resolve

# Fixed categorical order (palette slots 1..5), used for per-fold / per-series lines.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def classification_metrics(probs: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> dict:
    C = len(classes)
    assert probs.shape == (len(y), C), f"probs {probs.shape} vs {len(y)} labels, {C} classes"
    pred = probs.argmax(1)
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=range(C), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(C))
    out = {
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, c in enumerate(classes)},
        "macro_f1": float(f1.mean()),
        "accuracy": float((pred == y).mean()),
        "confusion_matrix": cm.tolist(),
    }
    if "sit" in classes and "down" in classes:
        s, d = classes.index("sit"), classes.index("down")
        out["sit_as_down"] = float(cm[s, d] / max(1, cm[s].sum()))
        out["down_as_sit"] = float(cm[d, s] / max(1, cm[d].sum()))
    return out


def threshold_grid(sweep_cfg) -> np.ndarray:
    return np.round(np.arange(float(sweep_cfg.start), float(sweep_cfg.stop) + 1e-9, float(sweep_cfg.step)), 4)


def reward_curve(probs: np.ndarray, y: np.ndarray, t: int, thresholds: Sequence[float]) -> list[dict]:
    """Reward for commanded class t iff argmax == t and probs[:, t] >= thr. One row per threshold."""
    pred = probs.argmax(1)
    is_t = y == t
    rows = []
    for thr in thresholds:
        fire = (pred == t) & (probs[:, t] >= thr)
        n_fired = int(fire.sum())
        rows.append({
            "threshold": float(thr),
            "precision": float((y[fire] == t).mean()) if n_fired else float("nan"),
            "recall": float((fire & is_t).sum() / max(1, is_t.sum())),
            "false_reward_rate": float((fire & ~is_t).sum() / max(1, (~is_t).sum())),
            "n_fired": n_fired,
        })
    return rows


def choose_threshold(curve: list[dict], precision_target: float, min_fired: int = 30) -> dict:
    """Lowest threshold reaching the precision target (keeps the most recall), counting only thresholds
    where at least `min_fired` images earn a reward. Falls back to the most precise usable threshold."""
    usable = [r for r in curve if r["n_fired"] >= min_fired and not np.isnan(r["precision"])]
    assert usable, "no threshold fires on enough images to estimate precision"
    hit = [r for r in usable if r["precision"] >= precision_target]
    best = min(hit, key=lambda r: r["threshold"]) if hit else max(usable, key=lambda r: (r["precision"], -r["threshold"]))
    return {**best, "reached_target": bool(hit), "precision_target": precision_target}


def row_at(curve: list[dict], thr: float) -> dict:
    return min(curve, key=lambda r: abs(r["threshold"] - thr))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, plt, path) -> Path:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion(cm, classes: Sequence[str], path, title: str) -> Path:
    """Rows = true class, columns = predicted. Cells show count and row percent (= recall on the diagonal)."""
    cm = np.asarray(cm)
    rowp = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax, plt = _styled_axes(figsize=(4.8, 4.2))
    ax.grid(False)
    ax.imshow(rowp, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * rowp[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if rowp[i, j] > 0.55 else _INK)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted", color=_INK2, fontsize=9)
    ax.set_ylabel("true", color=_INK2, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=10, loc="left")
    ax.spines["bottom"].set_visible(False)
    return _save(fig, plt, path)


def plot_reward_curves(curves: dict[str, list[dict]], chosen: dict[str, dict], path, title: str,
                       precision_target: float) -> Path:
    """One panel per commanded class: precision and recall vs. threshold, chosen threshold marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.8), dpi=120, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (cls, curve) in zip(axes[0], curves.items()):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e4e3df", linewidth=0.8)
        th = [r["threshold"] for r in curve]
        ax.plot(th, [r["precision"] for r in curve], color=SERIES[0], linewidth=2, label="precision")
        ax.plot(th, [r["recall"] for r in curve], color=SERIES[1], linewidth=2, label="recall")
        ax.axhline(precision_target, color=_INK2, linewidth=1, linestyle=":")
        c = chosen.get(cls)
        if c:
            ax.axvline(c["threshold"], color=_INK, linewidth=1.2, linestyle="--")
            ax.text(c["threshold"], 0.03, f" thr {c['threshold']:.2f}\n P {c['precision']:.2f} R {c['recall']:.2f}",
                    fontsize=8, color=_INK)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("reward threshold on p(class)", color=_INK2, fontsize=9)
        ax.set_title(f"commanded '{cls}'", color=_INK, fontsize=10, loc="left")
        ax.tick_params(colors=_INK2, labelsize=8)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, color=_INK, fontsize=11, x=0.01, ha="left")
    return _save(fig, plt, path)


def plot_training_curves(log_rows: list[dict], path, title: str) -> Path:
    """Two panels: train/val loss and val macro F1 per epoch, one line per fold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = sorted({r["fold"] for r in log_rows}, key=str)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.tick_params(colors=_INK2, labelsize=8)
        ax.set_xlabel("epoch", color=_INK2, fontsize=9)
    for k, f in enumerate(folds):
        rs = [r for r in log_rows if r["fold"] == f]
        color = SERIES[k % len(SERIES)]
        ep = [r["epoch"] for r in rs]
        a1.plot(ep, [r["train_loss"] for r in rs], color=color, linewidth=1.5, linestyle="--")
        if all(r.get("val_loss") is not None for r in rs):
            a1.plot(ep, [r["val_loss"] for r in rs], color=color, linewidth=2, label=f"fold {f}")
            a2.plot(ep, [r["val_macro_f1"] for r in rs], color=color, linewidth=2, marker="o", markersize=4,
                    label=f"fold {f}")
    a1.set_title("loss (solid = val, dashed = train)", color=_INK, fontsize=10, loc="left")
    a2.set_title("val macro F1", color=_INK, fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=_INK, fontsize=11, x=0.01, ha="left")
    return _save(fig, plt, path)


def plot_hbar(labels: Sequence[str], values: Sequence[float], path, title: str, xlabel: str,
              fmt: str = "{:.1%}") -> Path:
    fig, ax, plt = _styled_axes(figsize=(6.5, 0.45 * len(labels) + 1.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ys = range(len(labels))
    ax.barh(list(ys), values, height=0.6, color=_SERIES1)
    for y_, v in zip(ys, values):
        ax.text(v, y_, " " + fmt.format(v), va="center", fontsize=9, color=_INK)
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=_INK2, fontsize=9)
    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.set_title(title, color=_INK, fontsize=10, loc="left")
    return _save(fig, plt, path)
