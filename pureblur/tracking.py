"""Short-lived person tracking between timestamped detector refreshes."""

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .detector import Box, Image

Points = NDArray[np.float32]


@dataclass
class Track:
    box: Box
    points: Points
    seen: float
    number: int


def overlap(first: Box, second: Box) -> float:
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    if not intersection:
        return 0.0
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union


class RegionTracker:
    def __init__(self, interval: float, *, distinct_people: bool = False):
        self.interval = interval
        self.hold = max(0.38, 1.35 * interval)
        self.tracks: list[Track] = []
        self.previous: Image | None = None
        self.last_detection = float("-inf")
        self.cuts = 0
        self.distinct_people = distinct_people
        self.next_number = 0

    def advance(self, gray: Image, timestamp: float) -> bool:
        cut = self.previous is not None and cv2.mean(cv2.absdiff(gray, self.previous))[0] > 38
        uncertain = False
        if cut:
            self.tracks.clear()
            self.cuts += 1
        elif self.previous is not None and self.tracks:
            uncertain = self._move(gray)
        self.previous = gray
        self.tracks = [track for track in self.tracks if timestamp - track.seen <= self.hold]
        interval = min(self.interval, 0.16) if uncertain else self.interval
        return cut or timestamp - self.last_detection >= interval - 1e-6

    def refresh(self, boxes: list[Box], gray: Image, timestamp: float) -> None:
        self.last_detection = timestamp
        unmatched = self.tracks.copy()
        if self.distinct_people:
            # Restart tracks when competing boxes could inherit the same label.
            matches = [[track.number for track in unmatched if overlap(track.box, box) >= 0.28]
                       for box in boxes]
            counts = Counter(number for group in matches for number in group)
            ambiguous = {number for group in matches for number in group
                         if len(group) > 1 or counts[number] > 1}
            unmatched = [track for track in unmatched if track.number not in ambiguous]
        updated = []
        for box in boxes:
            prior = max(unmatched, key=lambda track: overlap(track.box, box), default=None)
            if prior is not None and overlap(prior.box, box) >= 0.28:
                unmatched.remove(prior)
                # Suppress jitter while retaining most of the newest measurement.
                box = tuple(0.80 * fresh + 0.20 * old for fresh, old in zip(box, prior.box))
                number = prior.number
            else:
                number = self.next_number
                self.next_number += 1
            updated.append(Track(box, self._features(gray, box), timestamp, number))
        self.tracks = (updated + unmatched)[:64]

    @staticmethod
    def _features(gray: Image, box: Box) -> Points:
        height, width = gray.shape
        x1, y1, x2, y2 = box
        left, right = int(x1 * width), round(x2 * width)
        top, bottom = int(y1 * height), round(y2 * height)
        if right - left < 5 or bottom - top < 5:
            return np.empty((0, 1, 2), dtype=np.float32)
        roi = gray[top:bottom, left:right]
        points = cv2.goodFeaturesToTrack(roi, maxCorners=12, qualityLevel=0.025, minDistance=3)
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        points[:, 0] += (left, top)
        return points

    def _move(self, gray: Image) -> bool:
        points = np.concatenate([track.points for track in self.tracks], axis=0)
        if not len(points):
            return True
        next_points, valid, error = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, points, None, winSize=(15, 15), maxLevel=2,
        )
        if next_points is None:
            return True
        image_height, image_width = gray.shape
        uncertain = False
        start = 0
        for track in self.tracks:
            end = start + len(track.points)
            moved = next_points[start:end]
            candidates = (valid[start:end, 0] != 0) & (error[start:end, 0] < 24)
            movement = (moved - track.points)[candidates, 0]
            start = end
            track.points = moved[candidates]
            if len(movement) < 3:
                uncertain = True
                continue
            dx, dy = np.median(movement, axis=0)
            if np.median(np.linalg.norm(movement - (dx, dy), axis=1)) > 3.5:
                uncertain = True
                continue
            x1, y1, x2, y2 = track.box
            dx = float(np.clip(dx / image_width, -x1, 1 - x2))
            dy = float(np.clip(dy / image_height, -y1, 1 - y2))
            track.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        return uncertain
