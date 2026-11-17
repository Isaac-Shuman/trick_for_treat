"""
trickforTreat/detector.py

Dog detection + square cropping, shared by scripts/04_crop_dogs.py (offline, dataset images) and
trickforTreat/pipeline.py (live frames). No dataset/CSV logic lives here.

Swapping the detector: anything with a `detect(images) -> list[FrameDetections]` method satisfies
the `Detector` protocol and works with select_and_crop() and PosturePipeline unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image

from .common import pick_device, resolve

COCO_DOG = 16
PROBE_CONF = 0.01  # YOLO runs at this confidence so near-misses below the real threshold can be logged


@dataclass(frozen=True)
class DogBox:
    xyxy: tuple[float, float, float, float]  # pixel coords in the input image
    conf: float
    cls_name: str = "dog"                      # COCO class YOLO assigned (dog, or a dog-like class)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


@dataclass
class FrameDetections:
    boxes: list[DogBox]          # dog boxes with conf >= threshold, highest confidence first
    max_conf_below: float        # best dog confidence under the threshold (0 if none at all)
    image_size: tuple[int, int]  # (width, height)


class Detector(Protocol):
    def detect(self, images: Sequence[Image.Image]) -> list[FrameDetections]: ...


class YoloDogDetector:
    """Ultralytics YOLO restricted to animal classes that, in practice, are dogs (see config
    detector.animal_class_ids), with class-agnostic NMS so each animal yields one box."""

    def __init__(self, weights: str = "yolo11s.pt", conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 device: str = "auto", dog_class_id: int = COCO_DOG, weights_dir: str = "models",
                 class_ids: Sequence[int] | None = None):
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        path = Path(weights) if Path(weights).is_absolute() else resolve(weights_dir) / weights
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            attempt_download_asset(str(path))  # fetches official weights by file name into `path`
        assert path.exists(), f"YOLO weights not found at {path}"
        self.weights_path = path
        self.model = YOLO(str(path))
        self.conf, self.iou, self.imgsz, self.dog_id = conf, iou, imgsz, dog_class_id
        self.class_ids = sorted(set(class_ids or [dog_class_id]) | {dog_class_id})
        self.device = pick_device(device)
        names = self.model.names
        assert names.get(dog_class_id) == "dog", f"class {dog_class_id} is {names.get(dog_class_id)!r}, not 'dog'"
        self.last_speed_ms: dict[str, float] = {}

    @classmethod
    def from_config(cls, det_cfg) -> "YoloDogDetector":
        return cls(weights=det_cfg.weights, conf=float(det_cfg.conf_threshold), iou=float(det_cfg.iou_threshold),
                   imgsz=int(det_cfg.imgsz), device=det_cfg.device, dog_class_id=int(det_cfg.dog_class_id),
                   weights_dir=det_cfg.get("weights_dir", "models"),
                   class_ids=[int(c) for c in det_cfg.get("animal_class_ids", [COCO_DOG])])

    def warmup(self, n: int = 2) -> None:
        blank = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        for _ in range(n):
            self.detect([blank])

    def detect(self, images: Sequence[Image.Image]) -> list[FrameDetections]:
        images = [im if im.mode == "RGB" else im.convert("RGB") for im in images]
        results = self.model.predict(images, classes=self.class_ids, conf=PROBE_CONF, iou=self.iou,
                                     imgsz=self.imgsz, device=self.device, agnostic_nms=True, verbose=False)
        assert len(results) == len(images)
        out = []
        for im, res in zip(images, results):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            assert all(int(c) in self.class_ids for c in cls), "YOLO returned an unrequested class"
            above = [DogBox(tuple(float(v) for v in xyxy[k]), float(confs[k]), self.model.names[int(cls[k])])
                     for k in range(len(confs)) if confs[k] >= self.conf]
            above.sort(key=lambda b: -b.conf)
            below = max((float(c) for c in confs if c < self.conf), default=0.0)
            out.append(FrameDetections(above, below, im.size))
        if results:
            self.last_speed_ms = {k: float(v) for k, v in results[0].speed.items()}  # per image, ms
        return out


# ---------------------------------------------------------------------------
# Square crop
# ---------------------------------------------------------------------------

def square_crop_box(xyxy, pad_fraction: float) -> tuple[int, int, int, int]:
    """Integer square centered on the box, big enough for the box padded by pad_fraction on each side.
    NOT clamped to the image: the part outside becomes fill, so the dog is never cut off or off-center."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    assert w > 0 and h > 0, f"degenerate box {xyxy}"
    side = int(math.ceil(max(max(w, h) * (1 + 2 * pad_fraction), max(w, h) + 2)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1, sy1 = int(math.floor(cx - side / 2)), int(math.floor(cy - side / 2))
    sq = (sx1, sy1, sx1 + side, sy1 + side)
    assert sq[0] <= x1 and sq[1] <= y1 and sq[2] >= x2 and sq[3] >= y2, f"square {sq} does not contain {xyxy}"
    return sq


def crop_with_fill(image: Image.Image, square, fill_rgb=(114, 114, 114)) -> tuple[Image.Image, float]:
    """Crop `square` (may extend past the image) and fill the outside with gray.
    Returns (square RGB crop, fraction of the crop that is fill)."""
    sx1, sy1, sx2, sy2 = square
    side = sx2 - sx1
    assert side == sy2 - sy1 and side > 0
    W, H = image.size
    ix1, iy1, ix2, iy2 = max(0, sx1), max(0, sy1), min(W, sx2), min(H, sy2)
    canvas = Image.new("RGB", (side, side), tuple(int(c) for c in fill_rgb))
    if ix2 > ix1 and iy2 > iy1:
        canvas.paste(image.convert("RGB").crop((ix1, iy1, ix2, iy2)), (ix1 - sx1, iy1 - sy1))
    inside = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return canvas, 1.0 - inside / (side * side)


@dataclass
class CropResult:
    status: str                     # ok | no_dog | below_conf | multi_dog | too_small
    box: DogBox | None = None
    square: tuple[int, int, int, int] | None = None
    crop: Image.Image | None = None
    fill_fraction: float = 0.0
    n_dogs: int = 0


def select_and_crop(image: Image.Image, det: FrameDetections, crop_cfg, multi_dog: str = "drop",
                    second_dog_min_conf: float = 0.0) -> CropResult:
    """The one place that turns detections into a classifier input. Used offline (04) and live (pipeline).
    multi_dog: "drop" (skip frames with several dogs: the label is ambiguous) or "best" (use the top box).
    second_dog_min_conf: weaker secondary boxes (people, statues, partial boxes) don't count as another dog."""
    if not det.boxes:
        return CropResult("below_conf" if det.max_conf_below > 0 else "no_dog")
    n = 1 + sum(b.conf >= second_dog_min_conf for b in det.boxes[1:])
    if n > 1 and multi_dog == "drop":
        return CropResult("multi_dog", box=det.boxes[0], n_dogs=n)
    box = det.boxes[0]
    if min(box.width, box.height) < float(crop_cfg.min_box_side_px):
        return CropResult("too_small", box=box, n_dogs=n)
    sq = square_crop_box(box.xyxy, float(crop_cfg.pad_fraction))
    crop, fill = crop_with_fill(image, sq, crop_cfg.fill_rgb)
    return CropResult("ok", box=box, square=sq, crop=crop, fill_fraction=fill, n_dogs=n)


def draw_detections(image: Image.Image, det: FrameDetections | None, square=None, caption: str = "") -> Image.Image:
    """Debug overlay: dog boxes in blue (with confidence), the square crop in orange, a caption on top."""
    from PIL import ImageDraw
    from .common import _font
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(min(im.size) / 150))
    for b in (det.boxes if det else []):
        d.rectangle(b.xyxy, outline=(42, 120, 214), width=lw)
        d.text((b.xyxy[0] + 3, b.xyxy[1] + 2), f"{b.cls_name} {b.conf:.2f}", fill=(42, 120, 214), font=_font(14))
    if square:
        d.rectangle(square, outline=(235, 104, 52), width=lw)
    if caption:
        d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
        d.text((4, 3), caption, fill=(255, 255, 255), font=_font(13))
    return im
