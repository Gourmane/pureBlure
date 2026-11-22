# PureBlur Studio

PureBlur Studio is a CPU-only Windows desktop app. It detects and tracks people in local videos, applies permanent blur to the selected people, and exports an H.264 MP4. It copies compatible audio packets and leaves the source file untouched. The models are included and processing runs offline.

## Run

Double-click **PureBlur Studio.exe** in this folder. Pick **Choose video…** for your own file, or click **Try demo** to load the included six-second clip. Under **People to blur**, choose **Everyone**, **Women only**, or **Men only**. Use **Save as…** to choose a folder and filename, or leave the destination next to the source. Click **Start export**, then **Open output**. The demo is an excerpt of the [NASA treadmill video](https://www.nasa.gov/video-detail/space-station-live-tracking-how-astronauts-run-in-space/).

The selective modes analyze the whole clip and then write the MP4. NanoDet finds people, YuNet finds faces, and a small ONNX model estimates appearance from face crops. The app combines observations within each continuous track so a face seen later can inform the beginning of that track. People without enough consistent evidence stay visible. The completion status shows how many frames contain uncertain detections. The estimate can be wrong, and overlapping person boxes can blur nearby pixels. The temporary plan stores box coordinates and is discarded after the export.

The launcher is 19 KiB and uses the existing workspace runtime on this Windows machine. Keep it in the project folder. Developers can rebuild it with `desktop/build.ps1`.

## Command line

Use 64-bit Python 3.12 on Windows. This one-time setup downloads the three pinned runtime packages from PyPI.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m pureblur "C:\Videos\input.mp4" -o "C:\Videos\input_blurred.mp4"
```

The output directory must already exist. Omit `-o` to save next to the input with the suffix `_blurred.mp4`. Use `--blur women` or `--blur men` for a selective export. The default is `--blur all`. Press Ctrl+C to cancel and discard the temporary export. Run `python -m pureblur --help` from an environment with the dependencies installed for other options.

| Preset | OpenCV and codec thread setting | Max detection interval | H.264 CRF |
| --- | ---: | ---: | ---: |
| Eco (default) | 1 | 0.52 s | 25 |
| Balanced | 2 | 0.30 s | 23 |
| Quality | 2 | 0.15 s | 21 |

Thread settings apply to the relevant codec and OpenCV operations. They are distinct from a hard OS utilization cap. All presets use the same 416 × 416 NanoDet model. A cut or uncertain motion estimate can trigger earlier inference. Lower CRF retains more detail and usually increases output size. `--threads`, `--interval`, `--crf`, `--confidence` and `--strength` can override defaults.

NanoDet is 1.07 MiB. The two selective models add about 0.73 MiB. The three Windows x64 dependency wheels total about 84 MB to download and about 234 MiB unpacked. They supply the native codecs and inference runtime, so no separate FFmpeg executable or GPU package is required.

The benchmark workspace has a 206.5 MiB local runtime. Its OpenCV video I/O plugin was excluded because PyAV handles media I/O. If `python` is a Windows Store alias, run the prototype with the Python already bundled with Codex.

```powershell
$pureBlurPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = (Resolve-Path .local-runtime).Path
& $pureBlurPython -m pureblur "C:\Videos\input.mp4"
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
ffprobe -v error -show_entries stream=codec_name,start_time,duration,nb_frames "C:\Videos\input_blurred.mp4"
```

FFprobe is an optional verification tool. The CLI validates a temporary file with PyAV before publishing it. Open the export in a normal video player and compare a shot with obvious motion to the source. Check that the blur follows the person, the original audio plays at the same moment, and a canceled job leaves no export.

The app blurs rectangles around detected people. Fast movement, distant subjects, overlapping people and near-threshold detections can leave visible areas or blur nearby background. This version handles SDR video with monotonic timestamps and compatible MP4 audio codecs. Unsupported audio fails with an error. Incompatible subtitles are skipped with a log message. Segmentation, sensitive-scene detection, and a timeline review mode are later milestones.

- [Performance benchmark](docs/benchmark.md)
- [30-second Eco export](samples/nasa-30s-blurred.mp4)
- [Windows architecture and roadmap](docs/architecture.md)
- [HaramBlur APK findings](docs/haramblur-reverse-engineering.md)
- [NanoDet model source](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) and [bundled Apache 2.0 license](models/LICENSE-NanoDet.txt)
- [YuNet model source](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) and [bundled MIT license](models/LICENSE-YuNet.txt)
- [Appearance classifier ONNX export](https://github.com/mowshon/age-and-gender/tree/6ddb6ce2d598705ccbe2067967b2f8be01fd3914/src/age_and_gender/models) and [bundled CC0 model notice](models/LICENSE-Gender-Model.md)
