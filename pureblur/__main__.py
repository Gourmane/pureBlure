"""CPU-only person blur entry point."""

import argparse
import ctypes
import logging
import os
from pathlib import Path
from time import perf_counter, process_time

# Matrix operations are small. Avoid starting another native thread pool.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import av
import cv2

from .detector import PersonDetector
from .settings import BlurTarget, DEFAULT_CONFIDENCE, DEFAULT_MODEL, DEFAULT_STRENGTH, PRESETS
from .video import process


def _peak_memory_mib() -> float:
    if os.name != "nt":
        import resource
        import sys

        unit = 1024**2 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit

    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_faults", wintypes.DWORD),
            ("peak_working_set", ctypes.c_size_t),
            ("working_set", ctypes.c_size_t),
            ("peak_paged", ctypes.c_size_t),
            ("paged", ctypes.c_size_t),
            ("peak_nonpaged", ctypes.c_size_t),
            ("nonpaged", ctypes.c_size_t),
            ("page_file", ctypes.c_size_t),
            ("peak_page_file", ctypes.c_size_t),
        ]

    current_process = ctypes.WinDLL("kernel32").GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    get_memory = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    get_memory.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(current_process(), ctypes.byref(counters), counters.cb):
        return float("nan")
    return counters.peak_working_set / 1024**2


def main() -> int:
    parser = argparse.ArgumentParser(description="Blur detected people in a local video and export H.264 MP4.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="MP4 path, default INPUT_blurred.mp4")
    parser.add_argument("--preset", choices=PRESETS, default="eco")
    parser.add_argument("--blur", type=BlurTarget, choices=tuple(BlurTarget), default=BlurTarget.ALL,
                        help="People to blur (default: all). Uncertain people stay visible in selective modes.")
    parser.add_argument("--threads", type=int, help="Override the preset thread count, 1–8")
    parser.add_argument("--interval", type=float, help="Maximum seconds between inference runs, 0.05–2")
    parser.add_argument("--crf", type=int, help="Override H.264 quality and file size, 16–35")
    parser.add_argument("--strength", type=int, default=DEFAULT_STRENGTH, help="Blur strength, 1–100")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Minimum person confidence, 0.05–0.95")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Compatible OpenCV Zoo NanoDet ONNX file")
    args = parser.parse_args()
    interval, threads, crf = PRESETS[args.preset]
    interval = args.interval if args.interval is not None else interval
    threads = args.threads if args.threads is not None else threads
    crf = args.crf if args.crf is not None else crf
    if not 0.05 <= interval <= 2:
        parser.error("--interval must be between 0.05 and 2 seconds")
    if not 1 <= threads <= 8:
        parser.error("--threads must be between 1 and 8")
    if not 16 <= crf <= 35:
        parser.error("--crf must be between 16 and 35")
    if not 1 <= args.strength <= 100 or not 0.05 <= args.confidence <= 0.95:
        parser.error("use strength 1–100 and confidence 0.05–0.95")
    source = args.input.resolve()
    output = (args.output or source.with_name(f"{source.stem}_blurred.mp4")).resolve()
    if not source.is_file():
        parser.error("input file does not exist")
    if output == source or output.exists():
        parser.error("output must be a new path separate from the source")
    if output.suffix.lower() != ".mp4":
        parser.error("output must use the .mp4 extension")
    if not args.model.is_file():
        parser.error(f"model file not found: {args.model}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cv2.setNumThreads(threads)
    cv2.ocl.setUseOpenCL(False)
    logging.info("CPU %s, %d thread(s), %.2fs interval, CRF %d, %.2f MiB model.",
                 args.preset, threads, interval, crf, args.model.stat().st_size / 1024**2)
    started, cpu_started = perf_counter(), process_time()
    try:
        detector = PersonDetector(args.model, args.confidence)
        stats = process(
            source, output, detector, interval=interval, threads=threads,
            crf=crf, strength=args.strength, blur_target=args.blur,
        )
    except KeyboardInterrupt:
        logging.error("Canceled. The unfinished export was discarded.")
        return 130
    except (OSError, ValueError, av.FFmpegError, cv2.error) as exc:
        logging.error("Export failed. %s", exc)
        return 1
    elapsed = perf_counter() - started
    cpu_seconds = process_time() - cpu_started
    logging.info("Output %s", output)
    logging.info("%d frames, %d blurred, %d/%d inference runs found people, %d cut triggers.",
                 stats.frames, stats.blurred_frames, stats.positive_detections, stats.detections, stats.cuts)
    logging.info("%.2fs elapsed, %.1f export fps, %.1f ms mean inference, %.2f inference runs/s while exporting.",
                 elapsed, stats.frames / elapsed, 1000 * stats.inference_seconds / stats.detections,
                 stats.detections / elapsed)
    logging.info("%.2f CPU seconds (%.0f%% of one logical CPU), %.1f MiB peak working set, %.2f MiB output.",
                 cpu_seconds, 100 * cpu_seconds / elapsed, _peak_memory_mib(), output.stat().st_size / 1024**2)
    if not stats.positive_detections:
        logging.warning("No people met the confidence threshold. The export has no person blur.")
    elif stats.uncertain_frames:
        logging.warning("Uncertain people were left visible in %d frames.", stats.uncertain_frames)
    elif not stats.blurred_frames:
        logging.warning("No detected people matched the selected target. The export has no person blur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
