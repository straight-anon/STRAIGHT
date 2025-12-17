#!/usr/bin/env python3
"""
Visualize estimated phase labels (rest/draw/release only).
Reads JSONs from estimated_labels/, draws a color-coded sidebar next to the
video, and writes to labeled_videos/ as <video>_estimated.mp4.
"""

import cv2
import json
import numpy as np
import subprocess
from pathlib import Path

from vlm_phase_estimation import prompt_for_video_id

DEFAULT_FPS = 60.0

PHASE_COLORS = {
    "rest": (200, 200, 200),
    "draw": (60, 180, 255),
    "full_draw": (0, 200, 0),
    "release": (0, 0, 255),
}


def get_phase_for_frame(frame_idx: int, phases):
    for phase in phases:
        if phase["start_frame"] <= frame_idx < phase["end_frame"]:
            return phase["name"]
    return phases[-1]["name"] if phases else "release"


def normalize_phase_segments(
    phases,
    label_fps: float,
    total_frames: int,
):
    if not phases:
        return []

    if "start_frame" in phases[0]:
        normalized = []
        for phase in phases:
            start = int(phase["start_frame"])
            end = int(phase["end_frame"])
            start = max(0, min(start, total_frames))
            end = max(start, min(end, total_frames))
            normalized.append({"name": phase["name"], "start_frame": start, "end_frame": end})
    else:
        normalized = []
        for phase in phases:
            start = int(round(phase["start"] * label_fps))
            end = int(round(phase["end"] * label_fps))
            start = max(0, min(start, total_frames))
            end = max(start, min(end, total_frames))
            normalized.append({"name": phase["name"], "start_frame": start, "end_frame": end})

    if normalized:
        normalized[-1]["end_frame"] = total_frames

    return normalized


def resolve_label_path(video_stem: str) -> Path:
    candidates = [
        Path("estimated_labels") / f"vlm_estimated_label_{video_stem}.json",
        Path("estimated_labels") / f"{video_stem}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No estimated label JSON found for {video_stem}. "
        "Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def visualize(video_identifier: str):
    video_stem = Path(video_identifier).stem
    video_name = f"{video_stem}.mp4"
    video_path = Path("training_videos") / video_name
    label_path = resolve_label_path(video_stem)
    out_path = Path("labeled_videos") / f"{video_stem}_estimated.mp4"
    out_path.parent.mkdir(exist_ok=True)

    with open(label_path) as f:
        data = json.load(f)
        phases_raw = data.get("phases", [])
        label_fps = float(data.get("fps") or DEFAULT_FPS)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or label_fps or DEFAULT_FPS)
    duration = frame_count / video_fps if video_fps else 0.0
    banner_w = 180
    phases = normalize_phase_segments(phases_raw, label_fps, frame_count)

    print(
        f"🎞️ {video_name}: {video_fps:.2f} fps | {frame_count} frames | {duration:.3f} s"
    )
    print(f"📄 Using labels from {label_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width + banner_w}x{height}",
        "-r",
        str(video_fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(video_fps),
        str(out_path),
    ]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        phase = get_phase_for_frame(i, phases)
        color = PHASE_COLORS.get(phase, (255, 255, 255))
        banner = np.full((height, banner_w, 3), color, np.uint8)
        label_text = phase.replace("_", " ").upper()
        cv2.putText(
            banner,
            label_text,
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        frame_out = np.hstack((frame, banner))
        pipe.stdin.write(frame_out.tobytes())

    cap.release()
    pipe.stdin.close()
    pipe.wait()
    print(f"✅ Saved estimated visualization → {out_path}")


if __name__ == "__main__":
    print("🎯 Estimated Phase Visualizer (rest/draw/release)")
    video_id = prompt_for_video_id()
    if video_id:
        visualize(video_id)
