"""
trickforTreat/classifier.py

Posture classifier: model construction (any timm backbone), checkpoint format, and inference.
Used by scripts/05_train_classifier.py (training) and trickforTreat/pipeline.py (runtime).

Swapping the classifier:
  - Different timm backbone: change classifier.backbone in the config (e.g. efficientnet_b0). Nothing else.
  - Something that isn't a timm model: implement the `PostureModel` protocol (classes, predict_batch,
    should_reward) and hand it to PosturePipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from PIL import Image

from .transforms import build_eval_transform

CHECKPOINT_VERSION = 1


@dataclass
class Prediction:
    label: str
    probs: dict[str, float]
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.probs[self.label]


class PostureModel(Protocol):
    classes: list[str]

    def predict_batch(self, crops: Sequence[Image.Image]) -> list[Prediction]: ...

    def should_reward(self, pred: Prediction, commanded: str) -> bool: ...


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> torch.nn.Module:
    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)


def head_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return list(model.get_classifier().parameters())


def backbone_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    head = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head]


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in backbone_parameters(model):
        p.requires_grad = trainable


# ---------------------------------------------------------------------------
# Inference wrapper + checkpoint format
# ---------------------------------------------------------------------------

class PostureClassifier:
    """A trained classifier + its preprocessing + its reward thresholds, loaded from one checkpoint."""

    def __init__(self, model: torch.nn.Module, classes: list[str], backbone: str, input_size: int,
                 mean, std, reward_thresholds: dict[str, float] | None = None, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.classes = list(classes)
        self.backbone, self.input_size = backbone, int(input_size)
        self.mean, self.std = list(mean), list(std)
        self.reward_thresholds = dict(reward_thresholds or {})
        self.device = device
        self.transform = build_eval_transform(self.input_size, self.mean, self.std)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": CHECKPOINT_VERSION,
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "classes": self.classes, "backbone": self.backbone, "input_size": self.input_size,
            "normalize": {"mean": self.mean, "std": self.std},
            "reward_thresholds": self.reward_thresholds,
            "extra": extra or {},
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PostureClassifier":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        assert ckpt.get("version") == CHECKPOINT_VERSION, f"unknown checkpoint version {ckpt.get('version')}"
        model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        obj = cls(model, ckpt["classes"], ckpt["backbone"], ckpt["input_size"], ckpt["normalize"]["mean"],
                  ckpt["normalize"]["std"], ckpt["reward_thresholds"], device)
        obj.extra = ckpt.get("extra", {})
        return obj

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, crops: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
        """[N, C] softmax probabilities, in self.classes order."""
        out = []
        for i in range(0, len(crops), batch_size):
            x = torch.stack([self.transform(c.convert("RGB")) for c in crops[i:i + batch_size]]).to(self.device)
            out.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), dtype=np.float32)

    def predict_batch(self, crops: Sequence[Image.Image]) -> list[Prediction]:
        probs = self.predict_probs(crops)
        return [self._to_prediction(p) for p in probs]

    @torch.no_grad()
    def predict(self, crop: Image.Image) -> Prediction:
        """Single crop with per-stage timing (preprocess / forward / postprocess, ms)."""
        t0 = time.perf_counter()
        x = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        t1 = time.perf_counter()
        logits = self.model(x)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = self._to_prediction(probs)
        t3 = time.perf_counter()
        pred.latency_ms = {"pre": 1e3 * (t1 - t0), "fwd": 1e3 * (t2 - t1), "post": 1e3 * (t3 - t2)}
        return pred

    def _to_prediction(self, probs: np.ndarray) -> Prediction:
        assert abs(float(probs.sum()) - 1.0) < 1e-3, "probabilities don't sum to 1"
        return Prediction(self.classes[int(probs.argmax())], {c: float(p) for c, p in zip(self.classes, probs)})

    def should_reward(self, pred: Prediction, commanded: str) -> bool:
        """Argmax plus threshold, per frame. The temporal gate (later) aggregates these over time."""
        assert commanded in self.reward_thresholds, f"no reward threshold for {commanded!r}"
        return pred.label == commanded and pred.probs[commanded] >= self.reward_thresholds[commanded]
