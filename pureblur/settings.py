"""Shared defaults for the command line and desktop app."""

from enum import StrEnum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "nanodet-int8.onnx"
FACE_MODEL = ROOT / "models" / "face-yunet.onnx"
APPEARANCE_MODEL = ROOT / "models" / "gender-v1.onnx"
DEFAULT_STRENGTH = 80
DEFAULT_CONFIDENCE = 0.30


class BlurTarget(StrEnum):
    ALL = "all"
    WOMEN = "women"
    MEN = "men"


# (maximum time between detections, codec and OpenCV thread count, H.264 CRF)
PRESETS = {
    "eco": (0.52, 1, 25),
    "balanced": (0.30, 2, 23),
    "quality": (0.15, 2, 21),
}
