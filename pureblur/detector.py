"""Decode COCO person boxes from the fixed-size OpenCV Zoo NanoDet export."""

from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]
Image = NDArray[np.uint8]
MODEL_SIZE = 416
LEVELS = ((8, "792", "795"), (16, "814", "817"), (32, "836", "839"))
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def orient(image: Image, rotation: int) -> Image:
    codes = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    return cv2.rotate(image, codes[rotation]) if rotation else image


class PersonDetector:
    def __init__(self, model: Path, confidence: float):
        # Passing bytes supports Windows paths with non-ASCII characters.
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(model.read_bytes(), dtype=np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.outputs = [name for _, *names in LEVELS for name in names]
        self.confidence = confidence
        self.canvas = np.empty((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        self.blob = np.empty((1, 3, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        self.bins = np.arange(8, dtype=np.float32)

    def detect(self, frame: av.VideoFrame, rotation: int) -> list[Box]:
        scale = MODEL_SIZE / max(frame.width, frame.height)
        small_width = max(1, round(frame.width * scale))
        small_height = max(1, round(frame.height * scale))
        small = frame.reformat(
            width=small_width, height=small_height, format="bgr24",
            interpolation=av.video.reformatter.Interpolation.AREA,
        ).to_ndarray()
        small = orient(small, rotation)
        height, width = small.shape[:2]
        x_pad = (MODEL_SIZE - width) // 2
        y_pad = (MODEL_SIZE - height) // 2
        self.canvas.fill(0)
        self.canvas[y_pad:y_pad + height, x_pad:x_pad + width] = small
        np.subtract(self.canvas, MEAN, out=self.blob[0].transpose(1, 2, 0))
        self.blob /= STD.reshape(1, 3, 1, 1)
        self.net.setInput(self.blob)
        values = self.net.forward(self.outputs)

        boxes: list[Box] = []
        scores: list[float] = []
        for (stride, _, _), classes, regression in zip(LEVELS, values[::2], values[1::2]):
            person = classes[0, :, 0]
            candidates = np.flatnonzero(person >= self.confidence)
            if not len(candidates):
                continue
            # Bound crowded frames before running box projection and NMS.
            candidates = candidates[np.argsort(person[candidates])[-240:]]
            distribution = regression[0, candidates].reshape(-1, 4, 8)
            distribution -= distribution.max(axis=2, keepdims=True)
            np.exp(distribution, out=distribution)
            offsets = (distribution @ self.bins) / distribution.sum(axis=2) * stride
            row, col = np.divmod(candidates, MODEL_SIZE // stride)
            centers_x = col * stride + (stride - 1) / 2
            centers_y = row * stride + (stride - 1) / 2
            for cx, cy, (left, top, right, bottom), index in zip(
                centers_x, centers_y, offsets, candidates
            ):
                x1 = float(np.clip((cx - left - x_pad) / width, 0, 1))
                y1 = float(np.clip((cy - top - y_pad) / height, 0, 1))
                x2 = float(np.clip((cx + right - x_pad) / width, 0, 1))
                y2 = float(np.clip((cy + bottom - y_pad) / height, 0, 1))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
                    scores.append(float(person[index]))

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, 0.50, top_k=128)
        return [
            (x, y, x + width, y + height)
            for x, y, width, height in (boxes[index] for index in keep[:64])
        ]
