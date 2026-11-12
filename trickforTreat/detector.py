"""
trickforTreat/detector.py  (PSEUDOCODE, shared module)

Dog detection + square cropping, shared by:
  - scripts/04_crop_dogs.py   (offline, batch over dataset images)
  - later: live loop on ESP32 MJPEG frames (VERIFY / WATCH states)
Keep this free of dataset/CSV logic so the live pipeline can import it as-is.
"""

# from dataclasses import dataclass
# from ultralytics import YOLO


# @dataclass
class DogBox:
    """xyxy: (x1, y1, x2, y2) in PIXEL coords of the input image; conf: float."""


# @dataclass
class FrameDetections:
    """boxes: list[DogBox] with conf >= threshold, sorted by conf descending;
    max_conf_below_threshold: float (0 if YOLO saw no dog at all). Used to tell "no_dog" from "below_conf"."""


class DogDetector:
    """
    from_config(det_cfg) -> DogDetector
        self.model = YOLO(det_cfg.weights); self.device = resolve(det_cfg.device)
        self.conf, self.iou, self.imgsz, self.dog_id = ...

    warmup(n=3): run on a blank 640x640 image so the first timed call isn't slow.

    detect_dogs(images: list[PIL.Image | np.ndarray]) -> list[FrameDetections]
        results = self.model.predict(images, classes=[self.dog_id], conf=0.01, iou=self.iou,
                                     imgsz=self.imgsz, device=self.device, verbose=False)
            (run at a very low conf, then apply self.conf here, so near-misses are visible for logging)
        for res in results:
            assert all(int(c) == self.dog_id for c in res.boxes.cls)
            above = [DogBox(xyxy, conf) for boxes with conf >= self.conf], sorted by conf desc
            below = max(conf of boxes < self.conf, default=0)
            yield FrameDetections(above, below)

    last_latency_ms -> dict  (from res.speed: preprocess / inference / postprocess, ms)
    """


def square_crop_box(xyxy, pad_fraction: float) -> tuple[int, int, int, int]:
    """
    Square box centered on the dog, big enough to contain the padded box. NOT clamped to the image:
    the part outside the image becomes gray fill in crop_with_fill, so the dog is never cut off and
    never off-center.
        w, h = x2 - x1, y2 - y1
        side = max(w * (1 + 2*pad_fraction), h * (1 + 2*pad_fraction))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx1, sy1 = floor(cx - side/2), floor(cy - side/2); sx2, sy2 = sx1 + ceil(side), sy1 + ceil(side)
        assert sx2 - sx1 == sy2 - sy1 and sx1 <= x1 and sy1 <= y1 and sx2 >= x2 and sy2 >= y2
    """


def crop_with_fill(image, box, fill_rgb) -> "PIL.Image":
    """
    canvas = Image.new("RGB", (side, side), tuple(fill_rgb))
    region = intersection of box with image bounds
    canvas.paste(image.crop(region), (region.x1 - box.x1, region.y1 - box.y1))
    return canvas
    Must be the same op the live loop uses.
    """


# DEBUGGING: run `python -m trickforTreat.detector <image>` (TODO: tiny __main__) to print
# detections and save the image with raw + square boxes drawn to results/debug/detector_check.jpg.
# Sanity: a stock dog photo should give exactly one box with conf > 0.5, and a square crop with
# the whole dog visible.
