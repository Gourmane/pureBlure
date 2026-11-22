import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from threading import Event
from unittest.mock import patch

import av
import cv2
import numpy as np

from pureblur.detector import PersonDetector
from pureblur.settings import BlurTarget
from pureblur.tracking import RegionTracker
from pureblur.video import ExportCanceled, blur_regions, process


class TrackingTest(unittest.TestCase):
    def test_moves_region_and_resets_it_at_a_cut(self):
        texture = np.random.default_rng(13).integers(0, 160, (48, 48), dtype=np.uint8)
        gray = np.zeros((96, 96), dtype=np.uint8)
        gray[24:72, 24:72] = texture
        tracker = RegionTracker(0.5)
        self.assertTrue(tracker.advance(gray, 0))
        tracker.refresh([(0.25, 0.25, 0.75, 0.75)], gray, 0)
        moving = cv2.warpAffine(gray, np.float32([[1, 0, 4], [0, 1, 2]]), (96, 96))
        self.assertFalse(tracker.advance(moving, 0.03))
        self.assertAlmostEqual(tracker.tracks[0].box[0], 28 / 96, delta=0.01)
        self.assertAlmostEqual(tracker.tracks[0].box[1], 26 / 96, delta=0.01)
        self.assertTrue(tracker.advance(np.full_like(gray, 220), 0.06))
        self.assertEqual(tracker.tracks, [])

    def test_blur_reduces_detail_inside_the_region(self):
        pixels = np.random.default_rng(27).integers(0, 256, (96, 96, 3), dtype=np.uint8)
        original = pixels.copy()
        tracker = RegionTracker(0.5)
        tracker.refresh([(0.25, 0.25, 0.75, 0.75)], pixels[:, :, 0], 0)
        blur_regions(pixels, [track.box for track in tracker.tracks], 80)
        self.assertLess(pixels[26:70, 26:70].std(), original[26:70, 26:70].std() / 4)
        np.testing.assert_array_equal(pixels[:16], original[:16])


class MediaTest(unittest.TestCase):
    def test_selective_export_applies_later_evidence_to_first_frame(self):
        people = [(0.04, 0.10, 0.28, 0.80), (0.40, 0.10, 0.62, 0.80), (0.74, 0.10, 0.96, 0.80)]

        class Detector:
            def detect(self, frame, rotation):
                return people

        def observe(frame, rotation, tracks, timestamp):
            if not timestamp:
                return []
            return [(track.number, vote) for track, vote in zip(tracks, (-2, 2, 0))]

        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "people.mp4"
            checker = np.indices((96, 192)).sum(axis=0) // 3 % 2 * 200 + 25
            pixels = np.repeat(checker.astype(np.uint8)[:, :, None], 3, axis=2)
            with av.open(str(source), "w") as container:
                video = container.add_stream("libx264", rate=10)
                video.width, video.height = 192, 96
                video.pix_fmt = "yuv420p"
                video.codec_context.thread_count = 1
                for tick in range(6):
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts = tick
                    container.mux(video.encode(frame))
                container.mux(video.encode())

            options = dict(interval=0.25, threads=1, crf=21, strength=80)
            with patch("pureblur.video.AppearanceClassifier") as classifier:
                classifier.return_value.observe.side_effect = observe
                for target, selected in [(BlurTarget.WOMEN, 0), (BlurTarget.MEN, 1)]:
                    updates = []
                    output = root / f"{target}.mp4"
                    stats = process(source, output, Detector(), blur_target=target,
                                    progress=updates.append, **options)
                    self.assertEqual((stats.frames, stats.blurred_frames, stats.uncertain_frames), (6, 6, 6))
                    self.assertTrue(updates[0].analyzing)
                    self.assertFalse(updates[-1].analyzing)
                    self.assertEqual(self.timestamps(source), self.timestamps(output))
                    with av.open(str(output)) as container:
                        first = next(container.decode(video=0)).to_ndarray(format="gray")
                    detail = [first[20:65, start:start + 25].std() for start in (15, 85, 150)]
                    self.assertLess(detail[selected], 20)
                    for other in {0, 1, 2} - {selected}:
                        self.assertGreater(detail[other], 50)

                cancel = Event()
                canceled = root / "canceled.mp4"
                with self.assertRaises(ExportCanceled):
                    process(source, canceled, Detector(), blur_target=BlurTarget.WOMEN,
                            cancel=cancel, progress=lambda update: cancel.set(), **options)
                self.assertEqual({file.name for file in root.iterdir()}, {"people.mp4", "women.mp4", "men.mp4"})

    def test_preserves_variable_frame_timestamps_and_input(self):
        cv2.setNumThreads(1)
        model = Path(__file__).resolve().parents[1] / "models" / "nanodet-int8.onnx"
        detector = PersonDetector(model, 0.30)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fréquence variable.mp4"
            output = Path(directory) / "sortie floue.mp4"
            with av.open(str(source), "w") as container:
                video = container.add_stream("libx264", rate=30)
                video.width = video.height = 96
                video.pix_fmt = "yuv420p"
                video.codec_context.thread_count = 1
                video.codec_context.time_base = Fraction(1, 30)
                for tick in [0, 1, 3, 4, 7, 8, 9]:
                    image = np.full((96, 96, 3), tick, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    frame.pts, frame.time_base = tick, Fraction(1, 30)
                    container.mux(video.encode(frame))
                container.mux(video.encode())
            original = source.read_bytes()

            with self.assertRaisesRegex(ValueError, "already exists"):
                process(source, source, detector, interval=0.52, threads=1, crf=25, strength=80)
            stats = process(source, output, detector, interval=0.52, threads=1, crf=25, strength=80)
            self.assertEqual(stats.frames, 7)
            self.assertEqual(stats.blurred_frames, 0)
            self.assertEqual(source.read_bytes(), original)
            original_timing = self.timestamps(source)
            export_timing = self.timestamps(output)
            self.assertEqual([pts for pts, _ in original_timing], [pts for pts, _ in export_timing])
            self.assertEqual(original_timing[-1], export_timing[-1])
            self.assertEqual(list(Path(directory).glob("*.part.mp4")), [])

            canceled = Path(directory) / "canceled.mp4"
            with patch("pureblur.video._transcode", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    process(source, canceled, detector, interval=0.52, threads=1, crf=25, strength=80)
            self.assertFalse(canceled.exists())
            self.assertEqual(list(Path(directory).glob("*.part.mp4")), [])

            cancel = Event()
            updates = []

            def stop_after_update(progress):
                updates.append(progress)
                cancel.set()

            with self.assertRaises(ExportCanceled):
                process(source, canceled, detector, interval=0.52, threads=1, crf=25,
                        strength=80, progress=stop_after_update, cancel=cancel)
            self.assertEqual(updates[0].frames, 1)
            self.assertFalse(canceled.exists())
            self.assertEqual(list(Path(directory).glob("*.part.mp4")), [])

            corrupt = Path(directory) / "corrupt.mp4"
            corrupt.write_bytes(b"not a media file")
            failed = Path(directory) / "failed.mp4"
            with self.assertRaises(av.FFmpegError):
                process(corrupt, failed, detector, interval=0.52, threads=1, crf=25, strength=80)
            self.assertFalse(failed.exists())
            self.assertEqual(list(Path(directory).glob("*.part.mp4")), [])

    @staticmethod
    def timestamps(path: Path) -> list[tuple[Fraction, Fraction]]:
        with av.open(str(path)) as container:
            return [
                (frame.pts * frame.time_base, frame.duration * frame.time_base)
                for frame in container.decode(video=0)
            ]
