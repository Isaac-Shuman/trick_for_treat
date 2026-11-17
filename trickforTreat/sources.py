"""
trickforTreat/sources.py

Image sources: anything that yields frames. The pipeline only depends on `ImageSource`, so moving
from dataset images to the ESP32-CAM is a change of source, not of pipeline code.

    open_source("data/raw/sitting")                 -> FolderSource (all images in a folder, recursive)
    open_source("dog.jpg")                          -> FileListSource
    open_source("data/split_manifest.csv")          -> ManifestSource (column `path`, optional split filter)
    open_source("clip.mp4") / open_source("0")      -> VideoSource (file or webcam index)
    open_source("http://<esp32-ip>:81/stream")      -> VideoSource (MJPEG over HTTP, via OpenCV)
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from .common import resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mjpeg", ".mjpg"}


@dataclass
class Frame:
    frame_id: str
    image: Image.Image          # RGB
    meta: dict = field(default_factory=dict)


class ImageSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FileListSource(ImageSource):
    def __init__(self, paths: Sequence[str | Path], meta: Sequence[dict] | None = None):
        self.paths = [resolve(p) for p in paths]
        self.meta = list(meta) if meta is not None else [{} for _ in self.paths]
        assert len(self.meta) == len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for p, m in zip(self.paths, self.meta):
            with Image.open(p) as im:
                yield Frame(p.stem, im.convert("RGB"), {"path": str(p), **m})


class FolderSource(FileListSource):
    def __init__(self, folder: str | Path):
        folder = resolve(folder)
        assert folder.is_dir(), f"not a folder: {folder}"
        super().__init__(sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS))


class ManifestSource(FileListSource):
    """Rows of a project manifest CSV; every column is passed along as frame meta (e.g. the true label)."""

    def __init__(self, csv_path: str | Path, split: str | None = None, path_column: str = "path"):
        with open(resolve(csv_path), newline="") as f:
            rows = [r for r in csv.DictReader(f) if split is None or r.get("split") == split]
        assert rows, f"no rows in {csv_path} (split={split})"
        super().__init__([r[path_column] for r in rows], rows)
        self.frame_ids = [r.get("image_id") for r in rows]

    def __iter__(self) -> Iterator[Frame]:
        for fid, frame in zip(self.frame_ids, super().__iter__()):
            frame.frame_id = fid or frame.frame_id
            yield frame


class VideoSource(ImageSource):
    """Video file, webcam index, or network stream (ESP32-CAM MJPEG) through OpenCV.
    every_n: keep every n-th frame. max_frames: stop after this many kept frames."""

    def __init__(self, uri: str | int, every_n: int = 1, max_frames: int | None = None):
        import cv2
        self.uri = int(uri) if isinstance(uri, str) and uri.isdigit() else uri
        self.cap = cv2.VideoCapture(self.uri)
        assert self.cap.isOpened(), f"could not open video source {uri!r}"
        self.every_n, self.max_frames = max(1, every_n), max_frames

    def __iter__(self) -> Iterator[Frame]:
        import cv2
        i = kept = 0
        while self.max_frames is None or kept < self.max_frames:
            ok, bgr = self.cap.read()
            if not ok:
                break
            if i % self.every_n == 0:
                yield Frame(f"frame_{i:06d}", Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                            {"index": i, "uri": str(self.uri)})
                kept += 1
            i += 1

    def close(self) -> None:
        self.cap.release()


def open_source(spec: str, split: str | None = None, every_n: int = 1, max_frames: int | None = None) -> ImageSource:
    if spec.isdigit() or "://" in spec:
        return VideoSource(spec, every_n, max_frames)
    p = resolve(spec)
    if p.is_dir():
        return FolderSource(p)
    if p.suffix.lower() == ".csv":
        return ManifestSource(p, split=split)
    if p.suffix.lower() in IMAGE_EXTS:
        return FileListSource([p])
    if p.suffix.lower() in VIDEO_EXTS:
        return VideoSource(str(p), every_n, max_frames)
    raise ValueError(f"don't know how to open {spec!r}")
