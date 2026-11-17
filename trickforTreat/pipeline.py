"""
trickforTreat/pipeline.py

Frame -> dog detector -> square crop -> posture classifier -> per-frame reward decision.
Depends only on the Detector and PostureModel protocols and on ImageSource, so the image source,
detector, and classifier can each be swapped independently. The temporal gate and LLM verifier
(later) consume the PipelineResult stream this produces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from PIL import Image

from .classifier import PostureClassifier, PostureModel, Prediction
from .detector import CropResult, Detector, FrameDetections, YoloDogDetector, draw_detections, select_and_crop
from .sources import Frame, ImageSource


@dataclass
class PipelineResult:
    frame_id: str
    status: str                          # ok | no_dog | below_conf | multi_dog | too_small
    detections: FrameDetections
    crop: CropResult
    prediction: Prediction | None = None
    reward: bool | None = None           # only set when a command is given
    timings_ms: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class PosturePipeline:
    def __init__(self, detector: Detector, classifier: PostureModel, crop_cfg, multi_dog: str = "drop",
                 second_dog_min_conf: float = 0.0):
        self.detector, self.classifier, self.crop_cfg, self.multi_dog = detector, classifier, crop_cfg, multi_dog
        self.second_dog_min_conf = second_dog_min_conf

    @classmethod
    def from_config(cls, cfg, checkpoint: str, device: str = "cpu") -> "PosturePipeline":
        return cls(YoloDogDetector.from_config(cfg.detector), PostureClassifier.load(checkpoint, device),
                   cfg.crop, multi_dog=cfg.get("pipeline", {}).get("multi_dog", "drop"),
                   second_dog_min_conf=float(cfg.detector.get("second_dog_min_conf", 0.0)))

    def process(self, image: Image.Image, frame_id: str = "", commanded: str | None = None,
                meta: dict | None = None) -> PipelineResult:
        t0 = time.perf_counter()
        det = self.detector.detect([image])[0]
        t1 = time.perf_counter()
        crop = select_and_crop(image, det, self.crop_cfg, self.multi_dog, self.second_dog_min_conf)
        t2 = time.perf_counter()
        res = PipelineResult(frame_id, crop.status, det, crop, meta=meta or {})
        if crop.status == "ok":
            res.prediction = self.classifier.predict_batch([crop.crop])[0]
            if commanded is not None:
                res.reward = self.classifier.should_reward(res.prediction, commanded)
        t3 = time.perf_counter()
        res.timings_ms = {"detect": 1e3 * (t1 - t0), "crop": 1e3 * (t2 - t1), "classify": 1e3 * (t3 - t2),
                          "total": 1e3 * (t3 - t0)}
        return res

    def run(self, source: ImageSource, commanded: str | None = None) -> Iterator[tuple[Frame, PipelineResult]]:
        for frame in source:
            yield frame, self.process(frame.image, frame.frame_id, commanded, frame.meta)


def annotate(image: Image.Image, res: PipelineResult) -> Image.Image:
    if res.prediction:
        p = res.prediction
        cap = f"{p.label} {p.confidence:.2f}"
        if res.reward is not None:
            cap += "  REWARD" if res.reward else "  no reward"
    else:
        cap = res.status
    return draw_detections(image, res.detections, res.crop.square, caption=f"{res.frame_id}: {cap}")
