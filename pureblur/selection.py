"""Estimate appearance locally and keep a temporary plan of body boxes."""

from struct import Struct
from typing import BinaryIO

import av
import cv2
import numpy as np

from .detector import Box, Image, orient
from .settings import APPEARANCE_MODEL, FACE_MODEL, BlurTarget
from .tracking import Track

FRAME = Struct("<qB")  # Input PTS and number of tracks.
REGION = Struct("<I4f")  # Track number and normalized person box.


class AppearanceClassifier:
    def __init__(self):
        self.faces = cv2.FaceDetectorYN.create(
            "onnx", np.frombuffer(FACE_MODEL.read_bytes(), dtype=np.uint8), [],
            (640, 640), 0.80, 0.30, 128, cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        self.net = cv2.dnn.readNetFromONNX(
            np.frombuffer(APPEARANCE_MODEL.read_bytes(), dtype=np.uint8),
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def observe(
        self, frame: av.VideoFrame, rotation: int, tracks: list[Track], timestamp: float,
    ) -> list[tuple[int, int]]:
        if not any(track.seen == timestamp for track in tracks):
            return []
        scale = min(1, 640 / max(frame.width, frame.height))
        small = frame.reformat(
            width=max(1, round(frame.width * scale)), height=max(1, round(frame.height * scale)),
            format="bgr24", interpolation=av.video.reformatter.Interpolation.AREA,
        )
        image = orient(small.to_ndarray(), rotation)
        height, width = image.shape[:2]
        self.faces.setInputSize((width, height))
        _, faces = self.faces.detect(image)
        if faces is None:
            return []

        assigned = {}
        for face in faces:
            x, y, face_width, face_height = face[:4]
            if min(face_width, face_height) < 20:
                continue
            cx = (x + face_width / 2) / width
            cy = (y + face_height / 2) / height
            candidates = [track for track in tracks if self._contains_head(track.box, cx, cy)]
            if len(candidates) == 1 and candidates[0].seen == timestamp:
                assigned.setdefault(candidates[0].number, []).append(face)
        return [(number, self._vote(image, faces[0])) for number, faces in assigned.items()
                if len(faces) == 1]

    @staticmethod
    def _contains_head(box: Box, x: float, y: float) -> bool:
        left, top, right, bottom = box
        return left <= x <= right and top <= y <= top + 0.65 * (bottom - top)

    def _vote(self, image: Image, face: np.ndarray) -> int:
        x, y, width, height = map(float, face[:4])
        rows, cols = image.shape[:2]
        left, top = max(0, round(x)), max(0, round(y))
        right, bottom = min(cols, round(x + width)), min(rows, round(y + height))
        if min(right - left, bottom - top) < 20:
            return 0
        blob = cv2.dnn.blobFromImage(
            image[top:bottom, left:right], 1 / 256, (32, 32),
            (122.781998, 117.000999, 104.297997), swapRB=True,
        )
        self.net.setInput(blob)
        female, male = self.net.forward().reshape(2)
        confidence = max(female, male)
        if not np.isfinite(confidence) or confidence < 0.85:
            return 0
        weight = 2 if confidence >= 0.97 else 1
        return weight if male > female else -weight


class SelectionPlan:
    """Replay tracked boxes after the full track has accumulated evidence."""

    def __init__(self, file: BinaryIO, target: BlurTarget):
        self.file = file
        self.target = target
        self.votes = bytearray()

    def capture(self, pts: int, tracks: list[Track], evidence: list[tuple[int, int]]) -> None:
        required = max((track.number + 1 for track in tracks), default=0)
        if required > len(self.votes):
            self.votes.extend(bytes(required - len(self.votes)))
        for number, vote in evidence:
            # Keep two bounded counts in one byte per track, including tracks
            # that disappear. Conflicting observations stay visible.
            female, male = divmod(self.votes[number], 16)
            female = min(15, female + max(0, -vote))
            male = min(15, male + max(0, vote))
            self.votes[number] = female * 16 + male
        record = bytearray(FRAME.pack(pts, len(tracks)))
        for track in tracks:
            record.extend(REGION.pack(track.number, *track.box))
        self.file.write(record)

    def boxes_for(self, pts: int) -> tuple[list[Box], bool]:
        header = self.file.read(FRAME.size)
        if len(header) != FRAME.size:
            raise ValueError("The input changed while rendering. Select it again.")
        expected, count = FRAME.unpack(header)
        if expected != pts:
            raise ValueError("The input timing changed while rendering. Select it again.")
        records = self.file.read(count * REGION.size)
        if len(records) != count * REGION.size:
            raise ValueError("The temporary analysis plan is incomplete. Retry the export.")
        boxes = []
        uncertain = False
        for number, left, top, right, bottom in REGION.iter_unpack(records):
            female, male = divmod(self.votes[number], 16)
            if abs(female - male) < 2 or min(female, male) >= 3:
                uncertain = True
            elif (female > male) == (self.target == BlurTarget.WOMEN):
                boxes.append((left, top, right, bottom))
        return boxes, uncertain
