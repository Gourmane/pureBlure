# Windows CPU benchmark

This run used a 30.03-second, 640 × 360, 29.97 fps clip on the Intel Core i5-6300U under Windows 10 build 19045. It starts 2:52 into NASA's treadmill segment and includes an interview and a runner in motion. Both presets used the bundled 1,123,958-byte model, a 0.30 person threshold and strength 80. The measured figures apply to this clip and machine. [NASA source video](https://www.nasa.gov/video-detail/space-station-live-tracking-how-astronauts-run-in-space/)

| Metric | Eco | Balanced |
| --- | ---: | ---: |
| End-to-end time | 19.78 s | 27.79 s |
| Export rate | 45.5 fps | 32.4 fps |
| Mean inference including preprocessing and postprocessing | 213.9 ms | 210.2 ms |
| Inference runs | 58 | 100 |
| Process CPU time | 19.02 s | 33.75 s |
| Mean logical CPU equivalent | 0.96 | 1.21 |
| Peak process working set | 99.3 MiB | 101.1 MiB |
| MP4 size | 887,781 bytes | 1,105,121 bytes |

One logical CPU equivalent means one logical processor's worth of work averaged across the run. Eco used 44% less CPU time and 19.7% less output space on this clip. The workload was CPU-only. No throughput estimate for long 1080p footage was derived from this 360p clip.

## Verification

- Three regression tests passed. They cover box movement and cut reset, blur inside the region, preservation of source bytes, variable frame timing, and temporary-file cleanup after cancellation or invalid media.
- FFprobe showed H.264 and AAC in the two benchmark exports. The source and Eco export matched on all 900 video presentation timestamps and frame durations, and on all 1,293 AAC packet hashes and timestamps. The source SHA-256 was identical before and after the run.
- A separate 26-frame variable-rate clip with Unicode paths matched all video presentation timestamps and durations and all 88 AAC packets. A two-track clip with a delayed second track matched all 164 AAC packet hashes, timestamps and durations. A rotated fixture exported upright at 640 × 360.
- Three source and export frame pairs were decoded and visually compared. The blur covered the interview subject and the visible head-to-shoe region of the treadmill runner in those pairs. Coverage on crowded or heavily occluded footage still needs measurement.

The [Eco MP4](../samples/nasa-30s-blurred.mp4) is included. To reproduce the test, download the NASA MP4 from the source page and run these commands with FFmpeg and FFprobe available.

```powershell
ffmpeg -threads 1 -ss 00:02:52 -i nasa-source.mp4 -t 30 -map 0:v:0 -map 0:a:0 -threads 1 -c:v libx264 -preset ultrafast -crf 20 -c:a aac -b:a 96k -n sample.mp4
python -m pureblur sample.mp4 -o sample-eco.mp4
python -m pureblur sample.mp4 --preset balanced -o sample-balanced.mp4
ffprobe -v error -show_entries stream=codec_name,start_time,duration,nb_frames sample-eco.mp4
```
