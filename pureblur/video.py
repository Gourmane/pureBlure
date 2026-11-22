"""Stream local frames into an H.264 MP4 while copying original audio packets."""

import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path
from threading import Event
from time import perf_counter

import av
import cv2
import numpy as np

from .detector import Box, Image, PersonDetector, orient
from .selection import AppearanceClassifier, SelectionPlan
from .settings import BlurTarget
from .tracking import RegionTracker

LOG = logging.getLogger(__name__)
MP4_AUDIO = {"aac", "ac3", "eac3", "alac", "mp3", "opus", "flac"}


@dataclass
class Statistics:
    frames: int = 0
    detections: int = 0
    positive_detections: int = 0
    blurred_frames: int = 0
    cuts: int = 0
    inference_seconds: float = 0
    uncertain_frames: int = 0


@dataclass(frozen=True)
class ExportProgress:
    frames: int
    blurred_frames: int
    fraction: float | None
    analyzing: bool = False


class ExportCanceled(Exception):
    """The user canceled before the output was published."""


def blur_regions(image: Image, boxes: Iterable[Box], strength: int) -> None:
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x_pad = (x2 - x1) * 0.12
        y_pad = (y2 - y1) * 0.15
        left = max(0, int((x1 - x_pad) * width))
        top = max(0, int((y1 - y_pad) * height))
        right = min(width, int(np.ceil((x2 + x_pad) * width)))
        # A box reaching the lower edge can be a cropped torso or legs.
        bottom = height if y2 + y_pad >= 0.87 else int(np.ceil((y2 + y_pad) * height))
        roi = image[top:bottom, left:right]
        rows, cols = roi.shape[:2]
        if not rows or not cols:
            continue
        cells = max(3, 33 - strength * 3 // 10)
        scale = min(1, cells / max(rows, cols))
        small = cv2.resize(
            roi, (max(1, round(cols * scale)), max(1, round(rows * scale))),
            interpolation=cv2.INTER_AREA,
        )
        small = cv2.GaussianBlur(small, (5, 5), 0)
        roi[:] = cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)


def _time(value: int | None, stream: av.stream.Stream) -> float | None:
    return None if value is None else float(value * stream.time_base)


def _validate(
    path: Path, source_video: av.video.stream.VideoStream,
    copied: list[av.stream.Stream], cadence: Fraction,
) -> None:
    with av.open(str(path)) as output:
        if len(output.streams.video) != 1 or output.streams.video[0].codec_context.name != "h264":
            raise ValueError("The temporary output has no valid H.264 video stream.")
        rendered = output.streams.video[0]
        rendered.codec_context.thread_count = 1
        checked = [source_video, *copied]
        if len(output.streams) != len(checked):
            raise ValueError("The temporary output is missing a stream.")
        if next(output.decode(rendered), None) is None:
            raise ValueError("The temporary video has no decodable frame.")
        tolerance = max(0.05, 2 / float(cadence))
        input_origin = _time(source_video.start_time, source_video)
        output_origin = _time(rendered.start_time, rendered)
        for original, actual in zip(checked, output.streams):
            if original.type != actual.type:
                raise ValueError("Unexpected stream ordering in the temporary output.")
            if original.type == "subtitle":
                continue
            if original.type == "audio" and original.codec_context.name != actual.codec_context.name:
                raise ValueError("An audio codec changed in the temporary output.")
            expected = _time(original.duration, original)
            duration = _time(actual.duration, actual)
            if expected is not None and (duration is None or abs(expected - duration) > tolerance):
                raise ValueError(f"Unexpected {actual.type} duration in the temporary output.")
            start = _time(original.start_time, original)
            actual_start = _time(actual.start_time, actual)
            if None not in (start, actual_start, input_origin, output_origin):
                if abs((start - input_origin) - (actual_start - output_origin)) > tolerance:
                    raise ValueError(f"Unexpected {actual.type} sync offset in the temporary output.")


def process(
    source_path: Path, output_path: Path, detector: PersonDetector, *,
    interval: float, threads: int, crf: int, strength: int,
    progress: Callable[[ExportProgress], None] | None = None,
    cancel: Event | None = None,
    blur_target: BlurTarget = BlurTarget.ALL,
) -> Statistics:
    blur_target = BlurTarget(blur_target)
    if cancel is not None and cancel.is_set():
        raise ExportCanceled
    if output_path.exists():
        raise ValueError("The output already exists. Choose a new filename.")
    if not output_path.parent.is_dir():
        raise ValueError("The output directory does not exist.")
    original = source_path.stat()
    if shutil.disk_usage(output_path.parent).free < max(64 * 1024**2, original.st_size):
        raise ValueError("Free some space on the output drive before starting.")
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.", suffix=".part.mp4", dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with av.open(str(source_path), options={"protocol_whitelist": "file"}) as source, ExitStack() as resources:
            video = next((
                stream for stream in source.streams.video
                if not stream.disposition & av.stream.Disposition.attached_pic
            ), None)
            if video is None:
                raise ValueError("The input has no usable video stream.")
            if video.codec_context.color_trc in (16, 18):
                raise ValueError("HDR input requires an HDR-aware rendering path. Use an SDR video.")
            if any(stream.codec_context.name not in MP4_AUDIO for stream in source.streams.audio):
                raise ValueError("The input contains audio that cannot be copied by this MP4 exporter.")
            copied = [
                stream for stream in source.streams if stream.type == "audio" or
                (stream.type == "subtitle" and stream.codec_context.name == "mov_text")
            ]
            skipped = len(source.streams.subtitles) - sum(s.type == "subtitle" for s in copied)
            if skipped:
                LOG.warning("Skipping %d subtitle stream(s) incompatible with MP4 stream copy.", skipped)
            cadence = video.average_rate or video.guessed_rate or Fraction(30)
            video.codec_context.thread_count = threads
            LOG.info(
                "Input %dx%d, nominal %.2f fps, %s, %d audio track(s).",
                video.width, video.height, float(cadence),
                f"{source.duration / av.time_base:.2f}s" if source.duration else "duration unknown",
                len(source.streams.audio),
            )
            selective = blur_target != BlurTarget.ALL
            tracker = RegionTracker(interval, distinct_people=selective)
            stats = Statistics()
            plan = None
            if selective:
                file = resources.enter_context(tempfile.TemporaryFile(dir=output_path.parent))
                plan = SelectionPlan(file, blur_target)
                _analyze_selection(source_path, video.index, detector, tracker, plan, stats,
                                   threads, progress, cancel)
            # MP4 edit lists express track start offsets in movie timescale ticks.
            # Use microseconds so the muxer can retain offsets within one audio sample.
            mux_options = {"max_interleave_delta": "1000000", "movie_timescale": "1000000"}
            with av.open(str(temporary), "w", options=mux_options) as target:
                _transcode(
                    source, target, video, copied, cadence, detector, tracker,
                    threads, crf, strength, progress, cancel, stats, plan,
                )
            _validate(temporary, video, copied, cadence)
        if selective:
            current = source_path.stat()
            if (original.st_size, original.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
                raise ValueError("The input file changed during processing. Select it again.")
        if cancel is not None and cancel.is_set():
            raise ExportCanceled
        if os.name == "nt":
            temporary.rename(output_path)
        else:
            # Publish atomically without overwriting a path created while the job ran.
            os.link(temporary, output_path)
        return stats
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp(frame: av.VideoFrame, width: int, height: int, previous: float) -> float:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("A video frame is missing its presentation timestamp.")
    timestamp = float(frame.pts * frame.time_base)
    if timestamp <= previous:
        raise ValueError("Video presentation timestamps are not increasing.")
    if (frame.width, frame.height) != (width, height):
        raise ValueError("Video dimensions changed while decoding.")
    return timestamp


def _track(
    frame: av.VideoFrame, timestamp: float, rotation: int, detector: PersonDetector,
    tracker: RegionTracker, stats: Statistics,
) -> bool:
    scale = 192 / max(frame.width, frame.height)
    gray = orient(frame.reformat(
        width=max(1, round(frame.width * scale)), height=max(1, round(frame.height * scale)),
        format="gray",
    ).to_ndarray(), rotation)
    if not tracker.advance(gray, timestamp):
        return False
    started = perf_counter()
    boxes = detector.detect(frame, rotation)
    stats.inference_seconds += perf_counter() - started
    stats.detections += 1
    stats.positive_detections += bool(boxes)
    tracker.refresh(boxes, gray, timestamp)
    return True


def _analyze_selection(
    path: Path, video_index: int, detector: PersonDetector, tracker: RegionTracker,
    plan: SelectionPlan, stats: Statistics, threads: int,
    progress: Callable[[ExportProgress], None] | None, cancel: Event | None,
) -> None:
    classifier = AppearanceClassifier()
    with av.open(str(path), options={"protocol_whitelist": "file"}) as source:
        video = source.streams[video_index]
        video.codec_context.thread_count = threads
        duration = _time(video.duration, video)
        frames = source.decode(video)
        first = next(frames, None)
        if first is None:
            raise ValueError("The input has no decodable video frames.")
        rotation = first.rotation % 360
        if rotation % 90:
            raise ValueError(f"Unsupported display rotation ({rotation} degrees).")
        width, height = first.width, first.height
        timestamp = float("-inf")
        start = None
        last_progress = 0.0
        for count, frame in enumerate(chain((first,), frames), 1):
            if cancel is not None and cancel.is_set():
                raise ExportCanceled
            timestamp = _timestamp(frame, width, height, timestamp)
            refreshed = _track(frame, timestamp, rotation, detector, tracker, stats)
            evidence = classifier.observe(frame, rotation, tracker.tracks, timestamp) if refreshed else []
            plan.capture(frame.pts, tracker.tracks, evidence)
            if progress is not None:
                if start is None:
                    start = timestamp
                now = perf_counter()
                if now - last_progress >= 0.25:
                    fraction = min(0.5, (timestamp - start) / duration / 2) if duration else None
                    progress(ExportProgress(count, 0, fraction, analyzing=True))
                    last_progress = now
    if cancel is not None and cancel.is_set():
        raise ExportCanceled
    plan.file.seek(0)


def _transcode(
    source: av.container.InputContainer, target: av.container.OutputContainer,
    video: av.video.stream.VideoStream, copied: list[av.stream.Stream],
    cadence: Fraction, detector: PersonDetector, tracker: RegionTracker,
    threads: int, crf: int, strength: int,
    progress: Callable[[ExportProgress], None] | None, cancel: Event | None,
    stats: Statistics, plan: SelectionPlan | None,
) -> None:
    packets = iter(source.demux(video, *copied))
    # Read the rotation from the first decoded frame on a second handle. This
    # avoids buffering audio packets encountered while waiting for that frame.
    with av.open(source.name, options={"protocol_whitelist": "file"}) as probe:
        probe_video = probe.streams[video.index]
        probe_video.codec_context.thread_count = threads
        first = next(probe.decode(probe_video), None)
    if first is None:
        raise ValueError("The input has no decodable video frames.")
    rotation = first.rotation % 360
    if rotation % 90:
        raise ValueError(f"Unsupported display rotation ({rotation} degrees).")
    width, height = first.width, first.height
    del first
    output = target.add_stream("libx264", rate=cadence)
    output.width, output.height = (height, width) if rotation % 180 else (width, height)
    output.width += output.width % 2
    output.height += output.height % 2
    output.pix_fmt = "yuv420p"
    output.codec_context.thread_count = threads
    output.codec_context.max_b_frames = 0
    output.codec_context.options = {
        "preset": "veryfast", "crf": str(crf), "rc-lookahead": "8",
    }
    output.codec_context.time_base = video.time_base
    output.time_base = video.time_base
    if video.sample_aspect_ratio:
        sar = video.sample_aspect_ratio
        output.codec_context.sample_aspect_ratio = 1 / sar if rotation % 180 else sar
    mappings = {stream.index: target.add_stream_from_template(stream) for stream in copied}
    audio_ends = {
        stream.index: stream.start_time + stream.duration for stream in copied
        if stream.type == "audio" and stream.start_time is not None and stream.duration is not None
    }
    last_pts = float("-inf")
    durations: dict[int, int] = {}
    duration_seconds = _time(video.duration, video)
    first_pts = None
    log_time = perf_counter()
    progress_time = 0.0
    for packet in packets:
        if cancel is not None and cancel.is_set():
            raise ExportCanceled
        if packet.stream.index in mappings:
            if packet.dts is not None:
                end = audio_ends.get(packet.stream.index)
                if end is not None and packet.pts is not None:
                    # The source edit list can end within the final AAC packet.
                    remaining = end - packet.pts
                    if 0 < remaining < packet.duration:
                        packet.duration = remaining
                packet.stream = mappings[packet.stream.index]
                target.mux(packet)
            continue
        for frame in packet.decode():
            timestamp = _timestamp(frame, width, height, last_pts)
            last_pts = timestamp
            if plan is None:
                _track(frame, timestamp, rotation, detector, tracker, stats)
                boxes = [track.box for track in tracker.tracks]
            else:
                boxes, uncertain = plan.boxes_for(frame.pts)
                stats.uncertain_frames += uncertain
            prepared = _prepare_frame(frame, output, rotation, boxes, strength)
            # libx264 can give the final VFR packet the nominal frame duration.
            # Keep decoder durations until their matching delayed packets are emitted.
            tick = round(frame.pts * frame.time_base / video.time_base)
            durations[tick] = round(frame.duration * frame.time_base / video.time_base)
            for encoded in output.encode(prepared):
                _mux_video(target, encoded, durations, video.time_base)
            stats.frames += 1
            stats.blurred_frames += bool(boxes)
            now = perf_counter()
            if now - log_time >= 5:
                LOG.info("Frames %d, inference runs %d, blurred frames %d.",
                         stats.frames, stats.detections, stats.blurred_frames)
                log_time = now
            if progress is not None:
                if first_pts is None:
                    first_pts = timestamp
                if now - progress_time >= 0.25:
                    fraction = min(0.99, (timestamp - first_pts) / duration_seconds) if duration_seconds else None
                    if plan is not None:
                        fraction = 0.5 + fraction / 2 if fraction is not None else 0.5
                    progress(ExportProgress(stats.frames, stats.blurred_frames, fraction))
                    progress_time = now
    if not stats.frames:
        raise ValueError("The input has no video frames.")
    if plan is not None and plan.file.read(1):
        raise ValueError("The input ended before the analysis plan. Select it again.")
    for packet in output.encode():
        _mux_video(target, packet, durations, video.time_base)
    stats.cuts = tracker.cuts


def _mux_video(
    target: av.container.OutputContainer, packet: av.Packet,
    durations: dict[int, int], time_base: Fraction,
) -> None:
    tick = round(packet.pts * packet.time_base / time_base)
    duration = durations.pop(tick, 0)
    if duration:
        packet.duration = max(1, round(duration * time_base / packet.time_base))
    target.mux(packet)


def _prepare_frame(
    frame: av.VideoFrame, output: av.video.stream.VideoStream,
    rotation: int, boxes: list[Box], strength: int,
) -> av.VideoFrame:
    if boxes or rotation or frame.width != output.width or frame.height != output.height:
        image = orient(frame.to_ndarray(format="bgr24"), rotation)
        blur_regions(image, boxes, strength)
        rows, cols = image.shape[:2]
        if rows != output.height or cols != output.width:
            image = cv2.copyMakeBorder(
                image, 0, output.height - rows, 0, output.width - cols, cv2.BORDER_REPLICATE,
            )
        prepared = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(format="yuv420p")
    else:
        prepared = frame.reformat(format="yuv420p")
    prepared.pts = frame.pts
    prepared.time_base = frame.time_base
    prepared.duration = frame.duration
    prepared.pict_type = av.video.frame.PictureType.NONE
    return prepared
