"""Small Windows desktop interface for the local video processor."""

import os
import tkinter as tk
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread
from time import perf_counter
from tkinter import filedialog, messagebox, ttk

from .settings import BlurTarget, DEFAULT_CONFIDENCE, DEFAULT_MODEL, DEFAULT_STRENGTH, PRESETS, ROOT

BACKGROUND = "#111420"
PANEL = "#1A1E2C"
WELL = "#10131D"
BORDER = "#34394C"
TEXT = "#F2F4FA"
MUTED = "#AAB3CA"
ACCENT = "#B9C4FF"
SUCCESS = "#A9E8CC"
ERROR = "#FFB6BA"
FONT = "Segoe UI"
DEMO = ROOT / "samples" / "demo-input.mp4"
TARGET_LABELS = {
    BlurTarget.ALL: "Everyone",
    BlurTarget.WOMEN: "Women only",
    BlurTarget.MEN: "Men only",
}


def available_output(destination: Path) -> Path:
    path = destination
    number = 2
    while path.exists():
        path = destination.with_stem(f"{destination.stem}_{number}")
        number += 1
    return path


class Studio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.source: Path | None = None
        self.destination: Path | None = None
        self.exported: Path | None = None
        self.photo: tk.PhotoImage | None = None
        self.messages = SimpleQueue()
        self.cancel: Event | None = None
        self.preview_count = 0
        self.polling = False
        self.closing = False

        root.title("PureBlur Studio")
        root.iconbitmap(default=str(ROOT / "desktop" / "PureBlur.ico"))
        root.configure(background=BACKGROUND)
        root.minsize(778, 580)
        root.geometry("880x600")
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Studio.TCombobox", fieldbackground=WELL, background=BORDER,
                        foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER,
                        padding=6, font=(FONT, 11))
        style.map("Studio.TCombobox", fieldbackground=[("readonly", WELL), ("disabled", PANEL)],
                  foreground=[("readonly", TEXT), ("disabled", MUTED)],
                  selectbackground=[("readonly", WELL)], selectforeground=[("readonly", TEXT)])
        style.configure("Studio.Horizontal.TProgressbar", troughcolor=WELL, background=ACCENT,
                        bordercolor=WELL, lightcolor=ACCENT, darkcolor=ACCENT, thickness=7)

        self.build()
        self.update_actions()

    def label(self, parent, text="", *, color=TEXT, background=PANEL, size=10, weight="normal", **kwargs):
        return tk.Label(parent, text=text, bg=background, fg=color, font=(FONT, size, weight), **kwargs)

    def button(self, parent, text, command, *, primary=False, **kwargs):
        fill, ink = (ACCENT, "#14182A") if primary else ("#282E42", TEXT)
        return tk.Button(
            parent, text=text, command=command, bg=fill, fg=ink, cursor="hand2",
            activebackground="#D7DFFF" if primary else "#3D4562", activeforeground=ink,
            disabledforeground="#929BB3", relief="flat", overrelief="flat", bd=0,
            highlightthickness=1, highlightbackground=fill, highlightcolor="#EDF0FF",
            padx=12, pady=7, font=(FONT, 10, "bold" if primary else "normal"), **kwargs,
        )

    @staticmethod
    def panel(parent):
        return tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)

    def build(self):
        page = tk.Frame(self.root, bg=BACKGROUND, padx=23, pady=14)
        page.pack(expand=True, fill="both")
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

        header = tk.Frame(page, bg=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        mark = tk.Canvas(header, width=39, height=39, bg=BACKGROUND, highlightthickness=0)
        mark.pack(side="left", padx=(0, 12), pady=(2, 0))
        mark.create_rectangle(2, 2, 37, 37, fill=ACCENT, outline=ACCENT)
        mark.create_rectangle(8, 11, 31, 28, outline="#171C32", width=2)
        mark.create_rectangle(13, 16, 17, 23, fill="#596AAE", outline="")
        mark.create_rectangle(19, 16, 25, 23, fill="#171C32", outline="")
        typography = tk.Frame(header, bg=BACKGROUND)
        typography.pack(side="left")
        self.label(typography, "PureBlur Studio", background=BACKGROUND, size=21, weight="bold").pack(anchor="w")
        self.label(typography, "Blur people in local videos", color=MUTED, background=BACKGROUND).pack(anchor="w")

        content = tk.Frame(page, bg=BACKGROUND)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3, minsize=395)
        content.columnconfigure(1, weight=2, minsize=274)
        content.rowconfigure(0, weight=1)

        video = self.panel(content)
        video.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.label(video, "Source video", size=13, weight="bold").pack(anchor="w", padx=15, pady=(12, 8))
        preview = tk.Frame(video, bg=WELL, height=204)
        preview.pack(fill="both", expand=True, padx=14)
        preview.pack_propagate(False)
        self.preview = self.label(preview, "Choose a video or load the demo", color=MUTED, background=WELL)
        self.preview.pack(expand=True, fill="both")
        self.source_name = self.label(video, "No video selected", anchor="w", wraplength=395)
        self.source_name.pack(fill="x", padx=15, pady=(7, 0))
        controls = tk.Frame(video, bg=PANEL)
        controls.pack(fill="x", padx=14, pady=(8, 13))
        self.choose = self.button(controls, "Choose video…", self.choose_video)
        self.choose.pack(side="left", padx=(0, 7))
        self.demo = self.button(controls, "Try demo", self.load_demo)
        self.demo.pack(side="left", padx=(0, 7))
        self.play_source = self.button(controls, "Play source", lambda: self.open_video(self.source))
        self.play_source.pack(side="left")

        settings = self.panel(content)
        settings.grid(row=0, column=1, sticky="nsew")
        self.label(settings, "Export settings", size=13, weight="bold").pack(anchor="w", padx=15, pady=(12, 8))
        self.label(settings, "Output MP4", weight="bold").pack(anchor="w", padx=15)
        output_row = tk.Frame(settings, bg=PANEL)
        output_row.pack(fill="x", padx=15, pady=(5, 12))
        self.output_name = tk.StringVar(value="Choose a video first")
        output = tk.Entry(output_row, textvariable=self.output_name, state="readonly", width=14,
                          readonlybackground=WELL, fg=TEXT, relief="flat", font=(FONT, 10),
                          highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)
        self.choose_output = self.button(output_row, "Save as…", self.save_as)
        self.choose_output.pack(side="right", padx=(7, 0))
        output.pack(side="left", fill="both", expand=True)

        self.label(settings, "People to blur", weight="bold").pack(anchor="w", padx=15)
        self.blur_target = ttk.Combobox(
            settings, values=tuple(TARGET_LABELS.values()), state="readonly",
            style="Studio.TCombobox", font=(FONT, 11), width=18,
        )
        self.blur_target.current(0)
        self.blur_target.pack(fill="x", padx=15, pady=(5, 3))
        self.target_help = self.label(settings, "", color=MUTED, wraplength=240, justify="left")
        self.target_help.pack(anchor="w", padx=15, pady=(0, 8))
        self.blur_target.bind("<<ComboboxSelected>>", self.describe_target)
        self.describe_target()

        self.label(settings, "Performance", weight="bold").pack(anchor="w", padx=15)
        self.preset = ttk.Combobox(settings, values=("Eco", "Balanced", "Quality"), state="readonly",
                                   style="Studio.TCombobox", font=(FONT, 11), width=18)
        self.preset.current(0)
        self.preset.pack(fill="x", padx=15, pady=(5, 3))
        self.preset_help = self.label(settings, "Lower CPU use · 1 thread", color=MUTED, wraplength=220, justify="left")
        self.preset_help.pack(anchor="w", padx=15, pady=(0, 8))
        self.preset.bind("<<ComboboxSelected>>", self.describe_preset)

        strength_row = tk.Frame(settings, bg=PANEL)
        strength_row.pack(fill="x", padx=15)
        self.label(strength_row, "Blur strength", weight="bold").pack(side="left")
        self.strength_label = self.label(strength_row, str(DEFAULT_STRENGTH), color=MUTED)
        self.strength_label.pack(side="right")
        self.strength = tk.IntVar(value=DEFAULT_STRENGTH)
        self.slider = tk.Scale(
            settings, from_=1, to=100, variable=self.strength, showvalue=False, orient="horizontal",
            bg=PANEL, fg=TEXT, activebackground=ACCENT, troughcolor=WELL, relief="flat",
            highlightthickness=1, highlightbackground=PANEL, highlightcolor=ACCENT,
            command=lambda value: self.strength_label.configure(text=str(round(float(value)))),
        )
        self.slider.pack(fill="x", padx=13, pady=(5, 6))

        footer = self.panel(page)
        footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        self.status = self.label(footer, "Select a video to start.", anchor="w", color=MUTED)
        self.status.grid(row=0, column=0, columnspan=2, sticky="ew", padx=13, pady=(9, 5))
        self.progress = ttk.Progressbar(footer, maximum=100, style="Studio.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, sticky="ew", padx=(13, 12), pady=(0, 11))
        actions = tk.Frame(footer, bg=PANEL)
        actions.grid(row=1, column=1, sticky="e", padx=(0, 11), pady=(0, 9))
        self.open_output = self.button(actions, "Open output", lambda: self.open_video(self.exported))
        self.open_output.pack(side="left", padx=(0, 7))
        self.cancel_button = self.button(actions, "Cancel", self.request_cancel)
        self.cancel_button.pack(side="left", padx=(0, 7))
        self.start_button = self.button(actions, "Start export", self.start_export, primary=True)
        self.start_button.pack(side="left")

    def describe_preset(self, _event):
        texts = ("Lower CPU use · 1 thread", "More frequent detection · 2 threads",
                 "Most frequent detection · 2 threads")
        self.preset_help.configure(text=texts[self.preset.current()])

    def describe_target(self, _event=None):
        self.target_help.configure(text=(
            "Blurs every detected person." if self.blur_target.current() == 0 else
            "Uncertain people stay visible."
        ))

    def update_actions(self):
        idle = self.cancel is None
        has_source = self.source is not None
        for item in (self.choose, self.slider):
            item.configure(state="normal" if idle else "disabled")
        self.demo.configure(state="normal" if idle and DEMO.is_file() else "disabled")
        for item in (self.play_source, self.choose_output):
            item.configure(state="normal" if idle and has_source else "disabled")
        can_start = idle and has_source
        fill = ACCENT if can_start else "#353C55"
        self.start_button.configure(state="normal" if can_start else "disabled", bg=fill, highlightbackground=fill)
        self.open_output.configure(state="normal" if idle and self.exported else "disabled")
        self.cancel_button.configure(state="normal" if not idle and not self.cancel.is_set() else "disabled")
        self.preset.configure(state="readonly" if idle else "disabled")
        self.blur_target.configure(state="readonly" if idle else "disabled")

    def set_status(self, text, color=MUTED):
        self.status.configure(text=text, fg=color)

    def choose_video(self):
        filename = filedialog.askopenfilename(
            parent=self.root, title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.mts"), ("All files", "*.*")],
        )
        if filename:
            self.select_source(Path(filename).resolve())

    def load_demo(self):
        self.select_source(DEMO)

    def select_source(self, source):
        self.source = source
        self.destination = available_output(source.with_name(f"{source.stem}_blurred.mp4"))
        self.exported = None
        self.photo = None
        self.preview.configure(text="Loading preview…", image="")
        self.output_name.set(self.destination.name)
        self.source_name.configure(text=source.name)
        self.progress.configure(value=0)
        self.set_status("Ready to export.")
        self.update_actions()
        self.preview_count += 1
        Thread(target=self.read_preview, args=(source,), daemon=True).start()
        self.schedule_poll()

    def save_as(self):
        filename = filedialog.asksaveasfilename(
            parent=self.root, title="Save blurred video", initialdir=str(self.destination.parent),
            initialfile=self.destination.name, defaultextension=".mp4", confirmoverwrite=False,
            filetypes=[("MP4 video", "*.mp4")],
        )
        if filename:
            destination = Path(filename).resolve()
            if destination.exists() or destination.suffix.lower() != ".mp4":
                self.set_status("Choose a new filename ending in .mp4.", ERROR)
                return
            self.destination = destination
            self.output_name.set(self.destination.name)
            self.set_status(f"Saving to {destination.parent}")

    def open_video(self, path):
        try:
            if path is not None:
                os.startfile(path)
        except OSError as error:
            messagebox.showerror("Could not open the video", str(error), parent=self.root)

    def read_preview(self, path):
        try:
            import av
            import numpy as np

            with av.open(str(path), options={"protocol_whitelist": "file"}) as source:
                stream = next(s for s in source.streams.video
                              if not s.disposition & av.stream.Disposition.attached_pic)
                stream.codec_context.thread_count = 1
                frame = next(source.decode(stream))
                rotation = frame.rotation % 360
                width, height = (frame.height, frame.width) if rotation % 180 else (frame.width, frame.height)
                scale = min(360 / width, 200 / height)
                small = frame.reformat(width=max(1, round(frame.width * scale)),
                                       height=max(1, round(frame.height * scale)), format="rgb24")
                pixels = np.rot90(small.to_ndarray(), k=rotation // 90)
                rows, cols = pixels.shape[:2]
                ppm = f"P6\n{cols} {rows}\n255\n".encode() + pixels.tobytes()
                length = f"  ·  {source.duration / av.time_base:.1f} s" if source.duration else ""
                detail = f"{path.name}  ·  {width} × {height}{length}"
                self.messages.put(("preview", path, ppm, detail))
        except Exception:
            self.messages.put(("preview", path, None, path.name))

    def schedule_poll(self):
        if not self.polling:
            self.polling = True
            self.root.after(100, self.poll)

    def poll(self):
        self.polling = False
        while True:
            try:
                event, *payload = self.messages.get_nowait()
            except Empty:
                break
            if event == "preview":
                self.preview_count -= 1
                path, ppm, detail = payload
                if path == self.source:
                    self.photo = tk.PhotoImage(data=ppm, format="PPM") if ppm else None
                    self.preview.configure(image=self.photo or "", text="" if self.photo else "Preview unavailable")
                    self.source_name.configure(text=detail)
            elif event == "progress":
                progress, = payload
                if self.cancel is not None and not self.cancel.is_set():
                    if progress.fraction is not None:
                        self.progress.configure(value=100 * progress.fraction)
                    if progress.analyzing:
                        self.set_status(f"Analyzing people…  {progress.frames:,} frames checked")
                    else:
                        self.set_status(f"Exporting…  {progress.frames:,} frames  ·  {progress.blurred_frames:,} blurred")
            else:
                self.finish_export(event, payload)
                if self.closing:
                    self.root.destroy()
                    return
        if self.cancel is not None or self.preview_count:
            self.schedule_poll()

    def start_export(self):
        if self.source is None or self.cancel is not None:
            return
        if not self.source.is_file() or not DEFAULT_MODEL.is_file():
            self.set_status("The selected video or the local detection model is missing.", ERROR)
            return
        if self.destination.exists():
            self.destination = available_output(self.destination)
            self.output_name.set(self.destination.name)
        self.exported = None
        self.cancel = Event()
        blur_target = tuple(TARGET_LABELS)[self.blur_target.current()]
        self.progress.configure(value=0)
        self.set_status("Starting local export…")
        self.update_actions()
        Thread(target=self.export, args=(self.source, self.destination, self.preset.get().lower(),
                                        self.strength.get(), self.cancel, blur_target), daemon=True).start()
        self.schedule_poll()

    def export(self, source, destination, preset, strength, cancel, blur_target):
        try:
            import cv2

            from .detector import PersonDetector
            from .video import ExportCanceled, process

            interval, threads, crf = PRESETS[preset]
            cv2.setNumThreads(threads)
            cv2.ocl.setUseOpenCL(False)
            started = perf_counter()
            detector = PersonDetector(DEFAULT_MODEL, DEFAULT_CONFIDENCE)
            try:
                stats = process(
                    source, destination, detector, interval=interval, threads=threads, crf=crf,
                    strength=strength, progress=lambda update: self.messages.put(("progress", update)),
                    cancel=cancel, blur_target=blur_target,
                )
            except ExportCanceled:
                self.messages.put(("canceled",))
                return
            self.messages.put(("done", destination, stats, perf_counter() - started))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def finish_export(self, event, payload):
        self.cancel = None
        if event == "done":
            self.exported, stats, elapsed = payload
            self.progress.configure(value=100)
            if not stats.positive_detections:
                self.set_status("Export complete. No people were detected.", MUTED)
            elif stats.uncertain_frames:
                self.set_status(f"Export complete  ·  {stats.blurred_frames:,} frames blurred  ·  "
                                f"Uncertain people left visible in {stats.uncertain_frames:,} frames", MUTED)
            elif not stats.blurred_frames:
                self.set_status("Export complete. No detected people matched the selected target.", MUTED)
            else:
                self.set_status(f"Export complete  ·  {stats.blurred_frames:,} frames blurred  ·  {elapsed:.1f} s", SUCCESS)
        elif event == "canceled":
            self.progress.configure(value=0)
            self.set_status("Export canceled. You can try again.")
        else:
            self.progress.configure(value=0)
            self.set_status("Export failed.", ERROR)
            if not self.closing:
                messagebox.showerror("Could not export the video", payload[0], parent=self.root)
        self.update_actions()

    def request_cancel(self):
        if self.cancel is not None:
            self.cancel.set()
            self.set_status("Canceling export…")
            self.update_actions()

    def close(self):
        if self.cancel is None:
            self.root.destroy()
        else:
            self.closing = True
            self.request_cancel()


def main():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
