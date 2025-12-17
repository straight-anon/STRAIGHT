#!/usr/bin/env python3
"""End-to-end analysis pipeline for STRAIGHT videos.

This script orchestrates the full workflow requested in the CLI:

1. Transcode the provided input video to 60 fps into ``training_videos/<video_id>.mp4``.
2. Run remote inference to generate ``inference_data/results_<video_id>.json``.
3. Run the VLM phase estimator to populate ``estimated_labels``.
4. Run Kalman smoothing to create ``smoothed_inference_data``.
5. Generate the ghost-overlay video against a reference shot (default nfst008).
6. Run the spine straightness prompt, draw-force-line visualization, and
   post-release follow-through analysis.
7. Copy every visualization (and the ghost video) into ``stream_output/`` and
   write a Markdown report that embeds the visuals and highlights the metrics.

The script expects that the OpenAI environment variables are configured so the
original tools can talk to GPT when needed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

import draw_force_line as dfl
import ghost_overlay
import visualize_skeleton as skeleton_viz
from rag_gpt_client import call_rag_assistant
from post_release_analysis import compute_crop_box as pra_compute_crop_box, crop_frame

ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training_videos"
INFERENCE_DIR = ROOT / "inference_data"
SMOOTHED_DIR = ROOT / "smoothed_inference_data"
LABEL_DIR = ROOT / "estimated_labels"
SAVED_VIDEOS_DIR = ROOT / "saved_videos"
STREAM_OUTPUT_DIR = ROOT / "stream_output"
SPINE_VIZ_DIR = ROOT / "spine_visualizations"
SPINE_ANALYSIS_DIR = ROOT / "spine_straight_analysis"
GHOST_ANALYSIS_DIR = ROOT / "ghost_overlay_analysis"
DFL_ANGLE_ANALYSIS_DIR = ROOT / "draw_force_angle_analysis"
DFL_LENGTH_ANALYSIS_DIR = ROOT / "draw_length_analysis"
POST_VIZ_DIR = ROOT / "post_release_visualizations"
POST_ANALYSIS_DIR = ROOT / "post_release_analysis_reports"
CROPPED_ASSETS_DIR = ROOT / "web" / "public" / "assets" / "originals"
WEB_ASSETS_RUNS_DIR = ROOT / "web" / "public" / "assets" / "runs"
RUNS_DIR = ROOT / "runs"
PRE_DRAW_FRAME_BUFFER = 30
POST_RELEASE_FRAME_BUFFER = 30
POST_RELEASE_SYSTEM_PROMPT = "You are an archery technique coach."
POST_RELEASE_DRAW_GOOD_PCT = 45.0
POST_RELEASE_BOW_TORSO_MAX_DELTA_DEG = 5.0
POST_RELEASE_DRAW_FILENAME = "post_release_draw.md"
POST_RELEASE_BOW_FILENAME = "post_release_bow.md"
POST_RELEASE_COMBINED_FILENAME = "post_release.md"
MIN_DATASET_SAMPLES = 5


@dataclass
class ReleaseSummary:
    draw_frame: Optional[int]
    release_frame: Optional[int]
    draw_to_release_frames: Optional[int]
    release_seconds: Optional[float]
    draw_to_release_seconds: Optional[float]
    dataset_mean_frame: Optional[float]
    dataset_std_frame: Optional[float]
    dataset_mean_seconds: Optional[float]
    dataset_std_seconds: Optional[float]
    dataset_count: int


@dataclass
class DrawForceSummary:
    image_path: Path
    sequence_paths: List[Path]
    draw_length_image: Optional[Path]
    draw_length_sequence: List[Path]
    geometry: dfl.DFLGeometry
    angle_stats: Optional[Tuple[float, float]]
    length_stats: Optional[Tuple[float, float]]
    angle_count: int
    length_count: int


@dataclass
class PostReleaseSummary:
    image_path: Path
    report_path: Path
    payload: Dict[str, object]
    release_image: Optional[Path] = None
    follow_image: Optional[Path] = None
    release_sequence: List[Path] = field(default_factory=list)
    follow_sequence: List[Path] = field(default_factory=list)
    markdown_path: Optional[Path] = None
    draw_markdown_path: Optional[Path] = None
    bow_markdown_path: Optional[Path] = None


def normalize_video_id(raw: str) -> str:
    base = Path(raw).stem
    return base.lower()


def run_command(label: str, args: List[str], input_text: Optional[str] = None) -> None:
    print(f"▶️  {label}: {' '.join(args)}")
    subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        input=input_text,
    )


def transcode_to_training_video(video_id: str, source: Path, force: bool) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    dest = TRAINING_DIR / f"{video_id}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Avoid writing input over itself; transcode to a temp file if paths collide.
    temp_dest = dest
    if source == dest:
        temp_dest = dest.with_suffix(".tmp.mp4")
        if temp_dest.exists():
            temp_dest.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        "fps=60",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(temp_dest),
    ]
    run_command("transcode to 60fps", cmd)
    if temp_dest != dest:
        dest.unlink(missing_ok=True)
        temp_dest.replace(dest)
    print(f"📼 Transcoded input video to 60 fps at {dest}")
    return dest


def run_remote_inference(video_id: str, force: bool, skip: bool) -> Tuple[Path, Path]:
    inference_json = INFERENCE_DIR / f"results_{video_id}.json"
    rendered_video = SAVED_VIDEOS_DIR / f"{video_id}.mp4"
    if skip:
        if not inference_json.exists():
            raise FileNotFoundError(
                f"--skip-remote requested but {inference_json} does not exist."
            )
        return inference_json, rendered_video
    TRAINING_DIR.mkdir(exist_ok=True)
    run_command(
        "remote inference",
        [
            sys.executable,
            "run_remote_inference.py",
            "--video",
            str(TRAINING_DIR / f"{video_id}.mp4"),
        ],
    )
    if not inference_json.exists():
        raise FileNotFoundError(f"Remote inference did not create {inference_json}.")
    return inference_json, rendered_video


def run_vlm_phase_estimation(
    video_id: str, use_st_prompt: bool, enable_debug: bool, force: bool, skip: bool
) -> Path:
    label_path = LABEL_DIR / f"vlm_estimated_label_{video_id}.json"
    if skip:
        if not label_path.exists():
            raise FileNotFoundError(
                f"--skip-vlm requested but {label_path} does not exist."
            )
        return label_path
    cmd = [
        sys.executable,
        "vlm_phase_estimation.py",
        "--video-id",
        video_id,
    ]
    if use_st_prompt:
        cmd.append("--use-st-prompt")
    if enable_debug:
        cmd.append("--enable-debug")
    run_command("VLM phase estimation", cmd)
    if not label_path.exists():
        raise FileNotFoundError(f"Phase estimation did not create {label_path}.")
    return label_path


def run_kalman_smoothing(video_id: str, force: bool, skip: bool) -> Path:
    smoothed_path = SMOOTHED_DIR / f"results_{video_id}.json"
    if skip:
        if not smoothed_path.exists():
            raise FileNotFoundError(
                f"--skip-smoothing requested but {smoothed_path} does not exist."
            )
        return smoothed_path
    inference_path = INFERENCE_DIR / f"results_{video_id}.json"
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference JSON missing: {inference_path}")
    run_command(
        "Kalman smoothing",
        [
            sys.executable,
            "kalman_smoothing.py",
            "--pose",
            str(inference_path),
            "--output",
            str(smoothed_path),
        ],
    )
    if not smoothed_path.exists():
        raise FileNotFoundError(f"Kalman smoothing did not create {smoothed_path}.")
    return smoothed_path


def load_keypoints_for_frame(pose_path: Path, frame_id: int) -> Dict[str, Point]:
    """Load named keypoints for a specific frame from a smoothed pose JSON."""
    data = json.loads(pose_path.read_text(encoding="utf-8"))
    meta = data.get("meta_info", {})
    frames: Dict[int, dict] = {}
    for frame in data.get("instance_info", []):
        try:
            fid = int(frame.get("frame_id", 1))
            frames[fid] = frame
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"No frames found in {pose_path}")
    if frame_id not in frames:
        # fall back to the closest frame id if the exact one is missing
        nearest = min(frames.keys(), key=lambda fid: abs(fid - frame_id))
        frame_id = nearest
    frame_entry = frames[frame_id]
    instances = frame_entry.get("instances") or []
    if not instances:
        raise RuntimeError(f"No instances found for frame {frame_id} in {pose_path}")
    source = instances[0]
    raw_points = source.get("smoothed_keypoints") or source.get("keypoints") or []
    name2id = meta.get("keypoint_name2id") or {}
    keypoints: Dict[str, Point] = {}
    for name, idx in name2id.items():
        try:
            i = int(idx)
        except Exception:
            continue
        if 0 <= i < len(raw_points):
            coords = raw_points[i]
            if isinstance(coords, Sequence) and len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if math.isfinite(x) and math.isfinite(y):
                    keypoints[name] = (x, y)
    if not keypoints:
        raise RuntimeError(f"No valid keypoints for frame {frame_id} in {pose_path}")
    return keypoints


def load_frame_points(pose_path: Path, frame_id: int) -> Tuple[Dict[str, Point], List[Point]]:
    """Load named keypoints and all points for a frame."""
    data = json.loads(pose_path.read_text(encoding="utf-8"))
    meta = data.get("meta_info", {})
    frames: Dict[int, dict] = {}
    for frame in data.get("instance_info", []):
        try:
            fid = int(frame.get("frame_id", 1))
            frames[fid] = frame
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"No frames found in {pose_path}")
    if frame_id not in frames:
        nearest = min(frames.keys(), key=lambda fid: abs(fid - frame_id))
        frame_id = nearest
    frame_entry = frames[frame_id]
    instances = frame_entry.get("instances") or []
    if not instances:
        raise RuntimeError(f"No instances found for frame {frame_id} in {pose_path}")
    source = instances[0]
    raw_points = source.get("smoothed_keypoints") or source.get("keypoints") or []
    name2id = meta.get("keypoint_name2id") or {}
    keypoints: Dict[str, Point] = {}
    all_points: List[Point] = []
    for coords in raw_points:
        if isinstance(coords, Sequence) and len(coords) >= 2:
            x, y = float(coords[0]), float(coords[1])
            if math.isfinite(x) and math.isfinite(y):
                all_points.append((x, y))
    for name, idx in name2id.items():
        try:
            i = int(idx)
        except Exception:
            continue
        if 0 <= i < len(raw_points):
            coords = raw_points[i]
            if isinstance(coords, Sequence) and len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if math.isfinite(x) and math.isfinite(y):
                    keypoints[name] = (x, y)
    if not keypoints:
        raise RuntimeError(f"No valid keypoints for frame {frame_id} in {pose_path}")
    return keypoints, all_points


def resolve_draw_bow_roles(keypoints: Dict[str, Point]) -> Dict[str, str]:
    """Identify draw-side and bow-side joints based on proximity to nose."""
    nose = keypoints.get("nose")
    left_wrist = keypoints.get("left_wrist")
    right_wrist = keypoints.get("right_wrist")
    if not (nose and left_wrist and right_wrist):
        raise RuntimeError("Missing nose or wrist keypoints required to resolve roles.")
    dist_left = math.hypot(left_wrist[0] - nose[0], left_wrist[1] - nose[1])
    dist_right = math.hypot(right_wrist[0] - nose[0], right_wrist[1] - nose[1])
    if dist_left <= dist_right:
        draw_side = "left"
        bow_side = "right"
    else:
        draw_side = "right"
        bow_side = "left"
    return {
        "draw_wrist": f"{draw_side}_wrist",
        "bow_wrist": f"{bow_side}_wrist",
        "bow_shoulder": f"{bow_side}_shoulder",
    }


def generate_line_sequence(
    frame,
    start: Point,
    end: Point,
    color: Tuple[int, int, int],
    base_path: Path,
) -> Tuple[List[Path], Path]:
    """Create a 5-frame sequence that grows a bright line from the right-most endpoint."""
    base_path.parent.mkdir(parents=True, exist_ok=True)

    def to_int(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0])), int(round(pt[1])))

    def render_frame(strength: float, fraction: float):
        if fraction <= 0:
            return frame.copy()
        anchor = start if start[0] >= end[0] else end
        target = end if anchor is start else start
        segment_end = (
            anchor[0] + (target[0] - anchor[0]) * fraction,
            anchor[1] + (target[1] - anchor[1]) * fraction,
        )
        base_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.line(base_mask, to_int(anchor), to_int(segment_end), 255, 5, cv2.LINE_AA)
        inner = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        outer = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        ring1 = cv2.subtract(inner, base_mask)
        ring2 = cv2.subtract(outer, inner)
        color_vec = np.array(color, dtype=np.float32)
        layer = np.zeros_like(frame, dtype=np.float32)
        layer += base_mask[:, :, None] * (color_vec * (2.3 * strength) / 255.0)
        layer += ring1[:, :, None] * (color_vec * (1.0 * strength) / 255.0)
        layer += ring2[:, :, None] * (color_vec * (0.55 * strength) / 255.0)
        glow = np.clip(layer, 0, 255).astype(np.uint8)
        return cv2.addWeighted(frame, 1.0, glow, 1.0, 0.0)

    fractions = [0.0, 0.35, 0.6, 0.85, 1.0]
    intensities = [1.0, 1.15, 1.35, 1.55, 1.75]
    sequence_paths: List[Path] = []
    for idx, (strength, fraction) in enumerate(zip(intensities, fractions), start=1):
        img = render_frame(strength, fraction)
        seq_path = base_path.with_name(f"{base_path.stem}_step{idx}{base_path.suffix}")
        cv2.imwrite(str(seq_path), img)
        sequence_paths.append(seq_path)
    final_path = base_path.with_name(f"{base_path.stem}{base_path.suffix}")
    cv2.imwrite(str(final_path), render_frame(intensities[-1], 1.0))
    return sequence_paths, final_path


def generate_side_by_side_sequence(
    left_img,
    right_img,
    left_line: Optional[Tuple[Point, Point]],
    right_line: Optional[Tuple[Point, Point]],
    left_color: Tuple[int, int, int],
    right_color: Tuple[int, int, int],
    base_path: Path,
    left_dot: Optional[Point] = None,
    right_dot: Optional[Point] = None,
) -> Tuple[List[Path], Path]:
    """Create a 5-frame sequence animating both panels side by side."""

    def to_int(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0])), int(round(pt[1])))

    def combine_with_divider(left_src, right_src):
        lh, lw = left_src.shape[:2]
        rh, rw = right_src.shape[:2]
        target_h = max(lh, rh)

        def resize_to_height(img, height):
            h, w = img.shape[:2]
            if h == height:
                return img
            scale = height / float(h)
            new_w = max(1, int(round(w * scale)))
            return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

        left_resized = resize_to_height(left_src, target_h)
        right_resized = resize_to_height(right_src, target_h)
        combined = cv2.hconcat([left_resized, right_resized])
        mid = left_resized.shape[1]
        cv2.line(combined, (mid, 0), (mid, combined.shape[0] - 1), (255, 255, 255), 2, cv2.LINE_AA)

        def add_label(text: str, x: int):
            pad_x = max(0, min(x, combined.shape[1] - 150))
            cv2.rectangle(combined, (pad_x, 14), (pad_x + 140, 50), (0, 0, 0), thickness=-1)
            cv2.putText(combined, text, (pad_x + 10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        add_label("Release", 12)
        add_label("Follow", mid + 12)
        return combined

    def render_panel(
        img,
        line: Optional[Tuple[Point, Point]],
        dot: Optional[Point],
        color: Tuple[int, int, int],
        strength: float,
        fraction: float,
    ):
        if fraction <= 0:
            return img.copy()
        base_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if line:
            start, end = line
            anchor = start if start[0] >= end[0] else end
            target_pt = end if anchor is start else start
            segment_end = (
                anchor[0] + (target_pt[0] - anchor[0]) * fraction,
                anchor[1] + (target_pt[1] - anchor[1]) * fraction,
            )
            cv2.line(base_mask, to_int(anchor), to_int(segment_end), 255, 5, cv2.LINE_AA)
        if dot:
            cv2.circle(base_mask, to_int(dot), 8, 255, thickness=-1, lineType=cv2.LINE_AA)
        inner = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        outer = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        ring1 = cv2.subtract(inner, base_mask)
        ring2 = cv2.subtract(outer, inner)
        color_vec = np.array(color, dtype=np.float32)
        layer = np.zeros_like(img, dtype=np.float32)
        layer += base_mask[:, :, None] * (color_vec * 2.0 * strength / 255.0)
        layer += ring1[:, :, None] * (color_vec * 0.9 * strength / 255.0)
        layer += ring2[:, :, None] * (color_vec * 0.5 * strength / 255.0)
        glow = np.clip(layer, 0, 255).astype(np.uint8)
        return cv2.addWeighted(img, 1.0, glow, 1.0, 0.0)

    base_path.parent.mkdir(parents=True, exist_ok=True)
    fractions = [0.0, 0.3, 0.55, 0.8, 1.0]
    intensities = [1.0, 1.2, 1.4, 1.6, 1.8]
    seq_paths: List[Path] = []
    left_hold = left_img
    right_hold = right_img
    for idx, (strength, fraction) in enumerate(zip(intensities, fractions), start=1):
        left_panel = render_panel(left_img, left_line, left_dot, left_color, strength, fraction)
        right_panel = render_panel(right_img, right_line, right_dot, right_color, strength, fraction)
        left_hold = left_panel
        right_hold = right_panel
        combined = combine_with_divider(left_panel, right_panel)
        seq_path = base_path.with_name(f"{base_path.stem}_step{idx}{base_path.suffix}")
        cv2.imwrite(str(seq_path), combined)
        seq_paths.append(seq_path)
    final_combined = combine_with_divider(left_hold, right_hold)
    final_path = base_path
    cv2.imwrite(str(final_path), final_combined)
    return seq_paths, final_path


def ensure_ghost_overlay(
    actual_pose: Path,
    reference_pose: Path,
    force: bool,
    actual_video: Optional[Path] = None,
    reference_video: Optional[Path] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
    asset_video_id: Optional[str] = None,
    report_video_id: Optional[str] = None,
) -> Path:
    out_path = ghost_overlay.OUTPUT_DIR / f"{actual_pose.stem}_ghost.mp4"
    if not reference_pose.exists():
        raise FileNotFoundError(f"Ghost reference pose missing: {reference_pose}")
    ghost_overlay.visualize_ghost(
        actual_pose,
        reference_pose,
        actual_video_path=actual_video,
        reference_video_path=reference_video,
        crop_offset=crop_offset,
        asset_video_id=asset_video_id,
        report_video_id=report_video_id,
    )
    if not out_path.exists():
        raise FileNotFoundError(f"Ghost overlay did not produce {out_path}.")
    return out_path


def detect_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for fps lookup: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    return float(fps)


def compute_release_summary(pose_path: Path, fps: float) -> ReleaseSummary:
    draw_frame, release_frame = ghost_overlay.get_draw_release_frames(pose_path)
    draw_to_release = None
    draw_to_release_seconds = None
    if draw_frame is not None and release_frame is not None:
        draw_to_release = max(0, release_frame - draw_frame)
        draw_to_release_seconds = draw_to_release / fps
    release_seconds = (
        release_frame / fps if release_frame is not None else None
    )
    avg_frame, std_frame, count = ghost_overlay.release_stats(pose_path)
    avg_seconds = avg_frame / fps if avg_frame is not None else None
    std_seconds = std_frame / fps if std_frame is not None else None
    return ReleaseSummary(
        draw_frame=draw_frame,
        release_frame=release_frame,
        draw_to_release_frames=draw_to_release,
        release_seconds=release_seconds,
        draw_to_release_seconds=draw_to_release_seconds,
        dataset_mean_frame=avg_frame,
        dataset_std_frame=std_frame,
        dataset_mean_seconds=avg_seconds,
        dataset_std_seconds=std_seconds,
        dataset_count=count,
    )


def load_phase_labels(label_path: Path) -> List[dict]:
    if not label_path.exists():
        return []
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
        return data.get("phases", []) or []
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to read phase labels from {label_path}: {exc}")
        return []


def compute_pose_crop_box(smoothed_path: Path, frame_shape: Tuple[int, int], release_frame: Optional[int]) -> Tuple[int, int, int, int]:
    """Return x1,y1,x2,y2 for the largest bbox up to release_frame (inclusive). Falls back to keypoint bounds."""
    height, width = frame_shape
    data = json.loads(smoothed_path.read_text(encoding="utf-8"))
    frames = data.get("instance_info", [])
    best_box: Optional[Tuple[float, float, float, float]] = None
    best_area = -1.0
    for entry in frames:
        frame_id = entry.get("frame_id")
        if release_frame is not None and frame_id is not None and frame_id > release_frame:
            continue
        instances = entry.get("instances") or []
        if not instances:
            continue
        bbox = instances[0].get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(float, bbox)
        else:
            kps = instances[0].get("smoothed_keypoints") or instances[0].get("keypoints") or []
            xs = [pt[0] for pt in kps if pt and len(pt) >= 2]
            ys = [pt[1] for pt in kps if pt and len(pt) >= 2]
            if not xs or not ys:
                continue
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        print(f"⚠️  No bounding box/keypoints found in {smoothed_path}; using full frame.")
        return (0, 0, width, height)

    x1, y1, x2, y2 = best_box
    pad = int(max(x2 - x1, y2 - y1) * 0.12)
    x1i = max(0, int(x1) - pad)
    y1i = max(0, int(y1) - pad)
    x2i = min(width, int(x2) + pad)
    y2i = min(height, int(y2) + pad)
    return (x1i, y1i, x2i, y2i)


def draw_phase_banner(
    canvas: np.ndarray,
    phases: List[dict],
    frame_idx: int,
    total_frames: int,
    bar_height: int = 14,
    banner_height: int = 80,
) -> None:
    colors = {
        "rest": (190, 190, 190),  # gray
        "draw": (0, 140, 255),  # bright orange
        "release": (0, 0, 255),  # red
    }
    h, w, _ = canvas.shape
    bar_x = 24
    bar_y = int(banner_height / 2)
    bar_w = int((w - 2 * bar_x) * 1.05)
    bar_x = max(12, bar_x - int(bar_w * 0.025))
    total = total_frames or 1

    labels: List[Tuple[str, Tuple[int, int], Tuple[int, int]]] = []
    if not phases:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height), colors["rest"], -1)
        labels.append(("rest", (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height)))
    else:
        for phase in phases:
            name = (phase.get("name") or "").lower()
            start = int(phase.get("start_frame") or 0)
            end = int(phase.get("end_frame") or start)
            start = max(0, min(start, total_frames))
            end = max(0, min(end, total_frames))
            start_px = bar_x + int((start / total) * bar_w)
            end_px = bar_x + int((end / total) * bar_w)
            x1 = min(start_px, end_px)
            x2 = max(start_px, end_px)
            cv2.rectangle(
                canvas,
                (x1, bar_y),
                (x2, bar_y + bar_height),
                colors.get(name, colors["rest"]),
                -1,
            )
            labels.append((name, (x1, bar_y), (x2, bar_y + bar_height)))

    progress_px = bar_x + int((frame_idx / total) * bar_w)
    tri_width = 16
    tri_height = 14
    base_y = bar_y - tri_height - 6  # position triangle above the bar
    tip_y = bar_y - 2  # tip closer to the bar to point down
    left = (progress_px - tri_width // 2, base_y)
    right = (progress_px + tri_width // 2, base_y)
    tip = (progress_px, tip_y)
    arrow_pts = np.array([tip, left, right], dtype=np.int32)
    cv2.fillPoly(canvas, [arrow_pts], (0, 90, 200))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, (x1, y1), (x2, y2) in labels:
        text = name.upper()
        text_size, _ = cv2.getTextSize(text, font, 0.45, 1)
        tx = x1 + max(4, (x2 - x1 - text_size[0]) // 2)
        ty = y2 + 18
        cv2.putText(canvas, text, (tx, ty), font, 0.45, (30, 30, 30), 1, cv2.LINE_AA)


def crop_and_banner_video(
    video_id: str,
    source_video: Path,
    smoothed_pose: Path,
    label_path: Path,
    force: bool,
) -> Tuple[Path, Tuple[int, int, int]]:
    CROPPED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CROPPED_ASSETS_DIR / f"{video_id}_original.mp4"
    meta_path = output_path.with_suffix(".meta.json")
    if output_path.exists() and not force:
        offset = (0, 0)
        banner_h = 0
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                crop = meta.get("crop") or [0, 0, 0, 0]
                offset = (int(crop[0]), int(crop[1]))
                banner_h = int(meta.get("banner_height") or 0)
            except Exception as exc:
                print(f"⚠️  Failed to read crop metadata ({exc}); defaulting offset to 0.")
        print(f"⏩ Using existing cropped video: {output_path}")
        return output_path, (offset[0], offset[1], banner_h)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    phases_raw = load_phase_labels(label_path)
    phases = []
    phase_total = None
    draw_phase_start = None
    release_phase_end = None
    try:
        lbl = json.loads(label_path.read_text(encoding="utf-8"))
        label_frames = int(lbl.get("frame_count") or 0) or None
        scale = 1.0
        if label_frames and label_frames > 0:
            scale = total_frames / float(label_frames)
        for ph in phases_raw:
            name = (ph.get("name") or "").lower()
            start = int(ph.get("start_frame") or 0)
            end = int(ph.get("end_frame") or start)
            start_scaled = int(start * scale)
            end_scaled = int(end * scale)
            phases.append({"name": name, "start_frame": start_scaled, "end_frame": end_scaled})
            if "draw" in name and draw_phase_start is None:
                draw_phase_start = start_scaled
            if "release" in name:
                release_phase_end = end_scaled
        phase_total = int(total_frames)
    except Exception:
        phases = phases_raw
        phase_total = total_frames
    release_phase = next((p for p in phases if (p.get("name") or "").lower() == "release"), None)
    release_frame = release_phase.get("start_frame") if release_phase else None
    phase_total = max(phase_total or 0, total_frames)

    # No trimming: use full video
    window_start = 0
    window_end = total_frames - 1
    clip_frame_count = max(1, window_end - window_start + 1)

    x1, y1, x2, y2 = compute_pose_crop_box(smoothed_pose, (height, width), release_frame)
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    banner_h = 100

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h + banner_h))
    if not writer.isOpened():
        print("⚠️  AVC1 codec unavailable; falling back to MP4V.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h + banner_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, window_start)
    frame_idx = 0
    while frame_idx < clip_frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        canvas = np.full((banner_h + crop.shape[0], crop.shape[1], 3), 255, dtype=np.uint8)
        canvas[banner_h:, :, :] = crop
        draw_phase_banner(canvas, phases, frame_idx, phase_total or clip_frame_count, banner_height=banner_h)
        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()
    meta = {
        "crop": [x1, y1, x2, y2],
        "banner_height": banner_h,
        "frame_window": [window_start, window_end],
        "clip_frame_count": clip_frame_count,
        "source_frame_count": total_frames,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"🎬 Cropped + bannered video saved to {output_path}")
    return output_path, (x1, y1, banner_h)


def adjust_pose_for_crop(smoothed_pose: Path, offset: Tuple[int, int, int], window_start: int, clip_frame_count: int) -> Path:
    """Shift all keypoints by the crop offset/banner height and rebase frame ids to the clipped video."""
    dx, dy, banner_h = offset
    if dx == 0 and dy == 0:
        return smoothed_pose
    out_path = smoothed_pose.with_name(f"{smoothed_pose.stem}_cropped.json")
    label_name = f"vlm_estimated_label_{smoothed_pose.stem.replace('results_', '')}.json"
    label_path = Path("estimated_labels") / label_name
    label_path_cropped = Path("estimated_labels") / f"vlm_estimated_label_{smoothed_pose.stem.replace('results_', '')}_cropped.json"
    if label_path.exists() and not label_path_cropped.exists():
        try:
            label_path_cropped.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"📄 Copied labels to {label_path_cropped}")
        except Exception as exc:
            print(f"⚠️  Failed to copy labels to cropped variant: {exc}")
    data = json.loads(smoothed_pose.read_text(encoding="utf-8"))
    for entry in data.get("instance_info", []):
        for inst in entry.get("instances", []):
            for field in ("keypoints", "smoothed_keypoints"):
                pts = inst.get(field) or []
                for pt in pts:
                    if len(pt) >= 2:
                        pt[0] -= dx
                        pt[1] -= dy
                        pt[1] += banner_h
        # rebase frame ids to start at 1 for the clipped video
        if window_start:
            fid = int(entry.get("frame_id", 1))
            entry["frame_id"] = max(1, fid - window_start)
    info = data.get("smoothing_info") or {}
    if window_start:
        for key in ("draw_frame", "release_frame", "window_start_frame", "window_end_frame"):
            if key in info and info[key] is not None:
                try:
                    val = int(info[key])
                    if key == "window_start_frame":
                        info[key] = 0
                    elif key == "window_end_frame":
                        info[key] = max(0, clip_frame_count - 1)
                    else:
                        info[key] = max(0, val - window_start)
                except (TypeError, ValueError):
                    pass
        data["smoothing_info"] = info
    out_path.write_text(json.dumps(data), encoding="utf-8")
    return out_path


def export_run_assets(
    video_id: str,
    processed_video: Path,
    ghost_video: Path,
    skeleton_video: Path,
    spine_image: Path,
    dfl_image: Path,
    dfl_sequence: Optional[List[Path]],
    draw_length_image: Optional[Path],
    draw_length_sequence: Optional[List[Path]],
    release_image: Optional[Path],
    release_sequence: Optional[List[Path]],
    follow_image: Optional[Path],
    follow_sequence: Optional[List[Path]],
    post_image: Path,
    report_path: Path,
    ghost_report: Optional[Path] = None,
    spine_report: Optional[Path] = None,
    dfl_angle_report: Optional[Path] = None,
    dfl_length_report: Optional[Path] = None,
    post_release_report: Optional[Path] = None,
    post_release_draw_report: Optional[Path] = None,
    post_release_bow_report: Optional[Path] = None,
) -> None:
    """Copy key artifacts into web/public/assets/runs/<video_id>/ for frontend use."""
    WEB_ASSETS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = WEB_ASSETS_RUNS_DIR / video_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Exporting run assets to {dest_dir}")

    def sort_paths(paths: Optional[List[Path]]) -> List[Path]:
        return sorted(paths or [], key=lambda p: p.name)

    # Fallback: if release/follow visuals were generated but not passed through,
    # pick them up directly from the visualization directory.
    def pick_release_follow(kind: str, primary: Optional[Path], seq: Optional[List[Path]]) -> Tuple[Optional[Path], Optional[List[Path]]]:
        if primary and primary.exists():
            return primary, sort_paths(seq)
        base = None
        steps: List[Path] = []
        try:
            pattern = f"*{video_id}*{kind}*.png"
            candidates = sorted(POST_VIZ_DIR.glob(pattern))
            for c in candidates:
                if "_step" in c.stem:
                    steps.append(c)
                elif base is None:
                    base = c
            if base is None and steps:
                # pick first non-step if missing
                base = steps[0]
        except Exception:
            pass
        return base, steps or sort_paths(seq)

    release_image, release_sequence = pick_release_follow("release", release_image, release_sequence)
    follow_image, follow_sequence = pick_release_follow("follow", follow_image, follow_sequence)

    copies = {
        "original.mp4": processed_video,
        "ghost.mp4": ghost_video,
        "skeleton.mp4": skeleton_video,
        "spine.png": spine_image,
        "draw_force.png": dfl_image,
        "draw_length.png": draw_length_image,
        "release.png": release_image,
        "follow_through.png": follow_image,
        "post_release.png": post_image,
        "report.md": report_path,
    }
    if dfl_sequence:
        for idx, path in enumerate(sort_paths(dfl_sequence), start=1):
            copies[f"draw_force_{idx}.png"] = path
    if draw_length_sequence:
        for idx, path in enumerate(sort_paths(draw_length_sequence), start=1):
            copies[f"draw_length_{idx}.png"] = path
    if release_sequence:
        for idx, path in enumerate(sort_paths(release_sequence), start=1):
            copies[f"release_{idx}.png"] = path
    if follow_sequence:
        for idx, path in enumerate(sort_paths(follow_sequence), start=1):
            copies[f"follow_through_{idx}.png"] = path
    if ghost_report:
        copies["ghost_overlay.md"] = ghost_report
    if spine_report:
        copies["spine.md"] = spine_report
    if dfl_angle_report:
        copies["draw_force_angle.md"] = dfl_angle_report
    if dfl_length_report:
        copies["draw_length.md"] = dfl_length_report
    if post_release_report:
        copies["post_release.md"] = post_release_report
    if post_release_draw_report:
        copies["post_release_draw.md"] = post_release_draw_report
    if post_release_bow_report:
        copies["post_release_bow.md"] = post_release_bow_report
    for name, path in copies.items():
        if path is None:
            continue
        try:
            copy_artifact(path, dest_dir / name)
            print(f"  ✅ copied {name}")
        except Exception as exc:
            print(f"⚠️  Failed to copy {path} to assets: {exc}")


def read_label_phases(label_path: Path) -> Tuple[List[Dict[str, int]], Optional[int]]:
    """Extract phase info and frame count from the estimated label JSON."""
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠️  Unable to read label file {label_path}: {exc}")
        return [], None
    phases_raw = data.get("phases") or []
    frame_count = data.get("frame_count")
    phases: List[Dict[str, int]] = []
    for phase in phases_raw:
        try:
            name = str(phase.get("name") or "").strip() or "phase"
            start = int(phase.get("start_frame") if "start_frame" in phase else phase.get("startFrame") or 0)
            end = int(phase.get("end_frame") if "end_frame" in phase else phase.get("endFrame") or start)
            phases.append({"name": name, "startFrame": start, "endFrame": end})
        except Exception:
            continue
    try:
        if frame_count is not None:
            frame_count = int(frame_count)
    except Exception:
        frame_count = None
    return phases, frame_count


def update_runs_manifest(
    video_id: str,
    label_path: Path,
    meta_path: Optional[Path],
    ghost_report: Optional[Path],
    spine_report: Optional[Path] = None,
    dfl_angle_report: Optional[Path] = None,
    dfl_length_report: Optional[Path] = None,
    post_release_report: Optional[Path] = None,
    post_release_draw_report: Optional[Path] = None,
    post_release_bow_report: Optional[Path] = None,
    dfl_sequence: Optional[List[Path]] = None,
    draw_length_sequence: Optional[List[Path]] = None,
    release_sequence: Optional[List[Path]] = None,
    follow_sequence: Optional[List[Path]] = None,
    release_image: Optional[Path] = None,
    follow_image: Optional[Path] = None,
) -> None:
    """Ensure the runs manifest has an entry for this video."""
    manifest_path = WEB_ASSETS_RUNS_DIR / "index.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"runs": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️  Failed to read existing manifest; recreating: {exc}")
            manifest = {"runs": []}

    phases, frame_count = read_label_phases(label_path)
    if meta_path and meta_path.exists() and frame_count is None:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame_count = int(meta.get("clip_frame_count") or meta.get("source_frame_count") or frame_count or 0) or None
        except Exception:
            pass

    entry = {
        "id": video_id,
        "label": video_id,
        "assets": {
          "original": f"runs/{video_id}/original.mp4",
          "skeleton": f"runs/{video_id}/skeleton.mp4",
          "ghost": f"runs/{video_id}/ghost.mp4",
          "spineImage": f"runs/{video_id}/spine.png",
          "drawForceImage": f"runs/{video_id}/draw_force.png",
          "drawForceFrames": [f"runs/{video_id}/draw_force_{i+1}.png" for i in range(len(dfl_sequence or []))] if dfl_sequence else None,
          "drawLengthImage": f"runs/{video_id}/draw_length.png",
          "drawLengthFrames": [f"runs/{video_id}/draw_length_{i+1}.png" for i in range(len(draw_length_sequence or []))] if draw_length_sequence else None,
          "releaseImage": f"runs/{video_id}/release.png" if release_image else f"runs/{video_id}/post_release.png",
          "releaseFrames": [f"runs/{video_id}/release_{i+1}.png" for i in range(len(release_sequence or []))] if release_sequence else None,
          "followThroughImage": f"runs/{video_id}/follow_through.png" if follow_image else f"runs/{video_id}/post_release.png",
          "followThroughFrames": [f"runs/{video_id}/follow_through_{i+1}.png" for i in range(len(follow_sequence or []))] if follow_sequence else None,
          "postReleaseImage": f"runs/{video_id}/post_release.png",
          "report": f"runs/{video_id}/report.md",
        },
        "frameCount": frame_count,
        "phases": phases,
    }
    if ghost_report:
        entry["ghostMarkdown"] = f"runs/{video_id}/ghost_overlay.md"
    if spine_report:
        entry["spineMarkdown"] = f"runs/{video_id}/spine.md"
    if dfl_angle_report:
        entry["drawForceMarkdown"] = f"runs/{video_id}/draw_force_angle.md"
    if dfl_length_report:
        entry["drawLengthMarkdown"] = f"runs/{video_id}/draw_length.md"
    if post_release_report:
        entry["postReleaseMarkdown"] = f"runs/{video_id}/post_release.md"
    if post_release_draw_report:
        entry["postReleaseDrawMarkdown"] = f"runs/{video_id}/post_release_draw.md"
        # Align with frontend expectations: treat draw-hand follow-through as the "release" analysis
        entry["releaseMarkdown"] = f"runs/{video_id}/post_release_draw.md"
    if post_release_bow_report:
        entry["postReleaseBowMarkdown"] = f"runs/{video_id}/post_release_bow.md"
        # Align with frontend expectations: treat bow-arm stability as the "followThrough" analysis
        entry["followThroughMarkdown"] = f"runs/{video_id}/post_release_bow.md"

    runs = [r for r in manifest.get("runs", []) if r.get("id") != video_id]
    runs.append(entry)
    manifest["runs"] = runs
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"🗂️  Updated runs manifest at {manifest_path} (runs={len(runs)})")
    except Exception as exc:
        print(f"⚠️  Failed to write runs manifest: {exc}")


def bundle_run_outputs(
    video_id: str,
    training_video: Path,
    processed_video: Path,
    inference_json: Path,
    smoothed_pose: Path,
    cropped_pose: Path,
    label_path: Path,
    label_cropped: Path,
    ghost_video: Path,
    skeleton_video: Path,
    spine_image: Path,
    dfl_image: Path,
    dfl_sequence: Optional[List[Path]],
    draw_length_image: Optional[Path],
    draw_length_sequence: Optional[List[Path]],
    release_image: Optional[Path],
    release_sequence: Optional[List[Path]],
    follow_image: Optional[Path],
    follow_sequence: Optional[List[Path]],
    post_image: Path,
    report_path: Path,
    ghost_report: Optional[Path] = None,
    post_report: Optional[Path] = None,
    post_draw_report: Optional[Path] = None,
    post_bow_report: Optional[Path] = None,
    dfl_reports: Optional[List[Path]] = None,
    meta_path: Optional[Path] = None,
) -> None:
    """Aggregate all outputs under runs/<video_id>/ for easier browsing."""
    run_dir = RUNS_DIR / video_id
    videos_dir = run_dir / "videos"
    poses_dir = run_dir / "poses"
    labels_dir = run_dir / "labels"
    images_dir = run_dir / "images"
    reports_dir = run_dir / "reports"
    meta_dir = run_dir / "meta"
    for d in (videos_dir, poses_dir, labels_dir, images_dir, reports_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    def safe_copy(src: Optional[Path], dest: Path):
        if src is None:
            return
        try:
            copy_artifact(src, dest)
            print(f"  📄 bundled {dest}")
        except Exception as exc:
            print(f"⚠️  Failed to copy {src} → {dest}: {exc}")

    # Videos
    safe_copy(training_video, videos_dir / "source.mp4")
    safe_copy(processed_video, videos_dir / "cropped_bannered.mp4")
    safe_copy(ghost_video, videos_dir / "ghost.mp4")
    safe_copy(skeleton_video, videos_dir / "skeleton.mp4")

    # Poses
    safe_copy(inference_json, poses_dir / "inference.json")
    safe_copy(smoothed_pose, poses_dir / "smoothed.json")
    safe_copy(cropped_pose, poses_dir / "smoothed_cropped.json")

    # Labels
    safe_copy(label_path, labels_dir / label_path.name)
    safe_copy(label_cropped, labels_dir / label_cropped.name)

    # Images
    safe_copy(spine_image, images_dir / "spine.png")
    safe_copy(dfl_image, images_dir / "draw_force.png")
    if dfl_sequence:
        for idx, img in enumerate(dfl_sequence, start=1):
            safe_copy(img, images_dir / f"draw_force_{idx}.png")
    safe_copy(draw_length_image, images_dir / "draw_length.png")
    if draw_length_sequence:
        for idx, img in enumerate(draw_length_sequence, start=1):
            safe_copy(img, images_dir / f"draw_length_{idx}.png")
    safe_copy(release_image, images_dir / "release.png")
    if release_sequence:
        for idx, img in enumerate(release_sequence, start=1):
            safe_copy(img, images_dir / f"release_{idx}.png")
    safe_copy(follow_image, images_dir / "follow_through.png")
    if follow_sequence:
        for idx, img in enumerate(follow_sequence, start=1):
            safe_copy(img, images_dir / f"follow_through_{idx}.png")
    safe_copy(post_image, images_dir / "post_release.png")

    # Reports
    safe_copy(report_path, reports_dir / "analysis_report.md")
    if ghost_report:
        safe_copy(ghost_report, reports_dir / ghost_report.name)
    if post_report:
        safe_copy(post_report, reports_dir / post_report.name)
    if post_draw_report:
        safe_copy(post_draw_report, reports_dir / post_draw_report.name)
    if post_bow_report:
        safe_copy(post_bow_report, reports_dir / post_bow_report.name)
    if dfl_reports:
        for r in dfl_reports:
            safe_copy(r, reports_dir / r.name)

    # Meta
    safe_copy(meta_path, meta_dir / meta_path.name if meta_path else meta_dir / "meta.json")


def run_spine_analysis(pose_path: Path, video_path: Path, force: bool) -> Tuple[Path, Optional[str]]:
    output_path = SPINE_VIZ_DIR / f"{pose_path.stem}_spine.png"
    run_command(
        "spine_straight",
        [
            sys.executable,
            "spine_straight.py",
            "--pose",
            str(pose_path),
            "--video",
            str(video_path),
            "--output",
            str(output_path),
        ],
    )
    report_path = SPINE_ANALYSIS_DIR / f"{pose_path.stem}.md"
    response = read_spine_report(report_path)
    return output_path, response


def run_skeleton_overlay(pose_path: Path, video_path: Path, force: bool) -> Path:
    output_path = skeleton_viz.OUTPUT_DIR / f"{pose_path.stem}.mp4"
    args = [
        sys.executable,
        "visualize_skeleton.py",
        "--pose",
        str(pose_path),
        "--video",
        str(video_path),
        "--output",
        str(output_path),
    ]
    run_command("visualize_skeleton", args)
    if not output_path.exists():
        raise FileNotFoundError(f"Skeleton overlay did not create {output_path}.")
    return output_path


def read_spine_report(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def read_ghost_report(pose_path: Path) -> Optional[str]:
    report_path = GHOST_ANALYSIS_DIR / f"{pose_path.stem}.md"
    if not report_path.exists():
        return None
    text = report_path.read_text(encoding="utf-8").strip()
    return text or None


def read_dfl_report(pose_path: Path) -> Optional[str]:
    parts: List[str] = []
    angle_path = DFL_ANGLE_ANALYSIS_DIR / f"{pose_path.stem}.md"
    length_path = DFL_LENGTH_ANALYSIS_DIR / f"{pose_path.stem}.md"
    if angle_path.exists():
        text = angle_path.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    if length_path.exists():
        text = length_path.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    if parts:
        return "\n\n".join(parts)
    return None


def read_post_release_report(pose_path: Path) -> Optional[str]:
    parts: List[str] = []
    combined = POST_ANALYSIS_DIR / f"{pose_path.stem}.md"
    draw_path = POST_ANALYSIS_DIR / pose_path.with_suffix(".draw.md").name
    bow_path = POST_ANALYSIS_DIR / pose_path.with_suffix(".bow.md").name
    for path in (draw_path, bow_path, combined):
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
    if parts:
        return "\n\n".join(parts)
    return None


def classify_metric(
    value: Optional[float],
    stats: Optional[Tuple[float, float]],
    label: str,
    unit: str,
    sample_count: Optional[int] = None,
    min_samples: int = MIN_DATASET_SAMPLES,
    grace: float = 0.0,
) -> Tuple[str, str]:
    if value is None:
        return "unknown", f"{label.capitalize()} unavailable; skipping comparison."
    if sample_count is not None and sample_count < min_samples:
        return "unknown", f"Not enough dataset stats available for {label} (n={sample_count}, need ≥{min_samples})."
    if not stats or stats[1] is None or stats[1] <= 0:
        return "unknown", f"No dataset stats available for {label}."
    avg, std = stats
    window = std + grace
    diff = value - avg
    if abs(diff) <= window:
        return "within", f"{label.capitalize()} stays within the reference window."
    direction = "higher" if diff > 0 else "lower"
    return (
        "outside",
        f"{label.capitalize()} is {abs(diff):.2f}{unit} {direction} than the reference mean."
    )


def write_dfl_reports(
    pose_stem: str,
    geometry: dfl.DFLGeometry,
    angle_stats: Optional[Tuple[float, float]],
    length_stats: Optional[Tuple[float, float]],
    dfl_image: Path,
    angle_count: Optional[int] = None,
    length_count: Optional[int] = None,
    min_samples: int = MIN_DATASET_SAMPLES,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Create separate draw-force angle and draw-length markdown reports, calling GPT when outside the window."""
    rel_image = Path("..") / dfl.OUTPUT_DIR.name / dfl_image.name

    def format_status(value: Optional[float], stats: Optional[Tuple[float, float]], label: str, unit: str) -> str:
        if value is None:
            return f"{label} unavailable."
        if not stats or stats[1] is None or stats[1] <= 0:
            return f"{label} comparison unavailable (no reference window)."
        avg, std = stats
        diff = value - avg
        if abs(diff) <= std:
            return f"{label} within reference window."
        direction = "higher" if diff > 0 else "lower"
        return f"{label} is {abs(diff):.2f}{unit} {direction} than the reference mean."

    angle_status, angle_desc = classify_metric(
        geometry.angle_deg, angle_stats, "dfl angle", "°", angle_count, min_samples
    )
    length_status, length_desc = classify_metric(
        geometry.normalized_draw_length, length_stats, "draw length", " hw", length_count, min_samples, 0.03
    )
    try:
        notes_text = dfl.load_dfl_notes()
    except Exception:
        notes_text = ""

    angle_report_path = None
    angle_lines = [
        "DATA_START",
        f"draw_force_angle_deg={geometry.angle_deg:.2f}",
        f"draw_force_angle_avg_deg={angle_stats[0]:.2f}" if angle_stats else "draw_force_angle_avg_deg=NA",
        f"draw_force_angle_std_deg={angle_stats[1]:.2f}" if angle_stats and angle_stats[1] is not None else "draw_force_angle_std_deg=NA",
        f"draw_force_angle_sample_count={angle_count if angle_count is not None else 0}",
        f"draw_force_min_samples_required={min_samples}",
        "DATA_END",
        "",
        "## DFL Angle",
    ]
    angle_body = ""
    if angle_status == "outside":
        try:
            prompt = dfl.load_prompt(dfl.PROMPT_DFL_ANGLE)
            question = "\n".join(
                [
                    prompt,
                    "",
                    f"Video ID: {pose_stem}",
                    f"DFL angle: {geometry.angle_deg:.2f}°",
                    f"Assessment: {angle_desc}",
                    "Issue flagged: provide concise corrective guidance (<120 words).",
                ]
            ).strip()
            response = call_rag_assistant(
                question,
                image_path=dfl_image,
                system_prompt=dfl.DFL_SYSTEM_PROMPT,
                notes_text=notes_text,
            )
            if response:
                angle_body = response.strip()
        except Exception as exc:
            print(f"⚠️  GPT angle guidance failed: {exc}")
    if not angle_body:
        angle_body = f"Result is good: {angle_desc}" if angle_status == "within" else f"Analysis unavailable: {angle_desc}"
    angle_lines.extend(
        [
            angle_body,
            "",
            f"![Draw-force visualization]({rel_image.as_posix()})",
            "",
        ]
    )
    DFL_ANGLE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    angle_report_path = DFL_ANGLE_ANALYSIS_DIR / f"{pose_stem}.md"
    angle_report_path.write_text("\n".join(angle_lines), encoding="utf-8")

    length_report_path = None
    if geometry.normalized_draw_length is not None:
        length_lines = [
            "DATA_START",
            f"draw_length_hipwidths={geometry.normalized_draw_length:.2f}",
            f"draw_length_avg_hipwidths={length_stats[0]:.2f}" if length_stats else "draw_length_avg_hipwidths=NA",
            f"draw_length_std_hipwidths={length_stats[1]:.2f}" if length_stats and length_stats[1] is not None else "draw_length_std_hipwidths=NA",
            f"draw_length_sample_count={length_count if length_count is not None else 0}",
            f"draw_length_min_samples_required={min_samples}",
            "DATA_END",
            "",
            "## Draw Length",
        ]
        length_body = ""
        if length_status == "outside":
            try:
                prompt = dfl.load_prompt(dfl.PROMPT_DFL_LENGTH)
                direction = "too long" if "higher" in length_desc else "too short"
                question = "\n".join(
                    [
                        prompt,
                        "",
                        f"Video ID: {pose_stem}",
                        f"Draw length is {direction} compared to the reference window.",
                        "Issue flagged: provide concise corrective guidance (<120 words).",
                    ]
                ).strip()
                response = call_rag_assistant(
                    question,
                    image_path=dfl_image,
                    system_prompt=dfl.DFL_SYSTEM_PROMPT,
                    notes_text=notes_text,
                )
                if response:
                    length_body = response.strip()
            except Exception as exc:
                print(f"⚠️  GPT draw-length guidance failed: {exc}")
        if not length_body:
            length_body = (
                f"Result is good: {length_desc}"
                if length_status == "within"
                else f"Analysis unavailable: {length_desc}"
            )
        length_lines.extend(
            [
                length_body,
                "",
                f"![Draw-force visualization]({rel_image.as_posix()})",
                "",
            ]
        )
        DFL_LENGTH_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        length_report_path = DFL_LENGTH_ANALYSIS_DIR / f"{pose_stem}.md"
        length_report_path.write_text("\n".join(length_lines), encoding="utf-8")

    return angle_report_path, length_report_path


def write_post_release_reports(
    pose_stem: str,
    post_summary: PostReleaseSummary,
    post_image: Path,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Create separate markdown reports for draw-hand and bow-arm follow-through with GPT fallback."""
    payload = post_summary.payload or {}

    def to_float(val: object) -> Optional[float]:
        try:
            num = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return num

    def fmt_pct(val: Optional[float]) -> str:
        return f"{val:.1f}%" if val is not None else "NA"

    def fmt_px(val: Optional[float]) -> str:
        return f"{val:.1f}px" if val is not None else "NA"

    def fmt_deg(val: Optional[float]) -> str:
        return f"{val:.1f}°" if val is not None else "NA"

    draw_change_pct = to_float(payload.get("nose_draw_length_change_pct"))
    draw_pre = to_float(payload.get("nose_draw_length_pre"))
    draw_follow = to_float(payload.get("nose_draw_length_follow"))
    bow_pre = to_float(payload.get("bow_torso_angle_pre"))
    bow_follow = to_float(payload.get("bow_torso_angle_follow"))
    bow_delta = None
    if bow_pre is not None and bow_follow is not None:
        bow_delta = abs(bow_follow - bow_pre)

    release_frame = payload.get("release_frame")
    follow_frame = payload.get("follow_frame") or payload.get("analyzed_frames")
    rel_image = Path("..") / POST_VIZ_DIR.name / post_image.name

    POST_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    draw_path = POST_ANALYSIS_DIR / f"{pose_stem}.draw.md"
    bow_path = POST_ANALYSIS_DIR / f"{pose_stem}.bow.md"
    combined_path = POST_ANALYSIS_DIR / f"{pose_stem}.md"

    # Draw-hand report
    draw_lines = [
        "DATA_START",
        f"nose_draw_length_pre_px={fmt_px(draw_pre)}",
        f"nose_draw_length_follow_px={fmt_px(draw_follow)}",
        f"nose_draw_length_change_pct={fmt_pct(draw_change_pct)}",
        f"nose_draw_length_good_threshold_pct={POST_RELEASE_DRAW_GOOD_PCT:.1f}",
        "DATA_END",
        "",
        "## Draw Hand Follow-Through",
    ]
    draw_body = ""
    if draw_change_pct is None:
        draw_body = "Analysis unavailable: nose-to-wrist change could not be computed."
    elif draw_change_pct > POST_RELEASE_DRAW_GOOD_PCT:
        draw_body = "Good release: follow-through hand stays back behind the head."
    else:
        try:
            question = "\n".join(
                [
                    "Follow-through coaching request (draw hand).",
                    f"Video ID: {pose_stem}",
                    f"Nose-to-wrist change: {fmt_pct(draw_change_pct)} (pre {fmt_px(draw_pre)} → post {fmt_px(draw_follow)}).",
                    f"Release frame: {release_frame}",
                    f"Follow-through frame: {follow_frame}",
                    "Change was below the 20% extension target; how to better the follow through? Keep it concise (<120 words).",
                ]
            ).strip()
            response = call_rag_assistant(
                question,
                image_path=post_image,
                system_prompt=POST_RELEASE_SYSTEM_PROMPT,
            )
            if response:
                draw_body = response.strip()
        except Exception as exc:  # pragma: no cover - GPT is best-effort
            print(f"⚠️  Post-release draw guidance failed: {exc}")
        if not draw_body:
            draw_body = (
                f"Follow-through needs work: nose-to-wrist change is {fmt_pct(draw_change_pct)} "
                f"(target > {POST_RELEASE_DRAW_GOOD_PCT:.1f}%)."
            )
    draw_lines.extend([draw_body, "", f"![Post-release visualization]({rel_image.as_posix()})", ""])
    draw_path.write_text("\n".join(draw_lines), encoding="utf-8")

    # Bow-arm report
    bow_lines = [
        "DATA_START",
        f"bow_torso_angle_pre_deg={fmt_deg(bow_pre)}",
        f"bow_torso_angle_follow_deg={fmt_deg(bow_follow)}",
        f"bow_torso_angle_delta_deg={fmt_deg(bow_delta)}",
        f"bow_torso_angle_threshold_deg={POST_RELEASE_BOW_TORSO_MAX_DELTA_DEG:.1f}",
        "DATA_END",
        "",
        "## Bow Arm vs Torso",
    ]
    bow_body = ""
    if bow_delta is None:
        bow_body = "Analysis unavailable: bow-arm vs torso angle could not be computed."
    elif bow_delta < POST_RELEASE_BOW_TORSO_MAX_DELTA_DEG:
        bow_body = "Good follow-through: bow arm did not collapse after release."
    else:
        try:
            question = "\n".join(
                [
                    "Bow-arm follow-through coaching request.",
                    f"Video ID: {pose_stem}",
                    f"Bow arm vs torso: pre {fmt_deg(bow_pre)}, post {fmt_deg(bow_follow)} (Δ {fmt_deg(bow_delta)}).",
                    "The bow arm collapsed after release; what suggestion do you make? Keep it concise (<120 words).",
                ]
            ).strip()
            response = call_rag_assistant(
                question,
                image_path=post_image,
                system_prompt=POST_RELEASE_SYSTEM_PROMPT,
            )
            if response:
                bow_body = response.strip()
        except Exception as exc:  # pragma: no cover - GPT is best-effort
            print(f"⚠️  Post-release bow-arm guidance failed: {exc}")
        if not bow_body:
            bow_body = (
                f"Bow arm stability needs work: angle changed {fmt_deg(bow_delta)} "
                f"(target < {POST_RELEASE_BOW_TORSO_MAX_DELTA_DEG:.1f}°)."
            )
    bow_lines.extend([bow_body, "", f"![Post-release visualization]({rel_image.as_posix()})", ""])
    bow_path.write_text("\n".join(bow_lines), encoding="utf-8")

    # Combined (for legacy consumers)
    combined_lines = [
        "DATA_START",
        f"nose_draw_length_pre_px={fmt_px(draw_pre)}",
        f"nose_draw_length_follow_px={fmt_px(draw_follow)}",
        f"nose_draw_length_change_pct={fmt_pct(draw_change_pct)}",
        f"nose_draw_length_good_threshold_pct={POST_RELEASE_DRAW_GOOD_PCT:.1f}",
        f"bow_torso_angle_pre_deg={fmt_deg(bow_pre)}",
        f"bow_torso_angle_follow_deg={fmt_deg(bow_follow)}",
        f"bow_torso_angle_delta_deg={fmt_deg(bow_delta)}",
        f"bow_torso_angle_threshold_deg={POST_RELEASE_BOW_TORSO_MAX_DELTA_DEG:.1f}",
        "DATA_END",
        "",
        "## Draw Hand Follow-Through",
        draw_body,
        "",
        "## Bow Arm vs Torso",
        bow_body,
        "",
        f"![Post-release visualization]({rel_image.as_posix()})",
        "",
    ]
    combined_path.write_text("\n".join(combined_lines), encoding="utf-8")

    return draw_path, bow_path, combined_path


def run_draw_force_analysis(pose_path: Path, video_path: Path, force: bool) -> DrawForceSummary:
    output_path = dfl.OUTPUT_DIR / f"{pose_path.stem}_force.png"
    draw_length_output = dfl.OUTPUT_DIR / f"{pose_path.stem}_draw_length.png"
    pose_frame = dfl.load_pose_frame(pose_path)
    geometry = dfl.compute_dfl_geometry(pose_frame.keypoints)
    frame = dfl.read_frame(video_path, pose_frame.frame_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def to_int(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (int(round(pt[0])), int(round(pt[1])))

    def line_effect(
        color: Tuple[int, int, int],
        start: Tuple[float, float],
        end: Tuple[float, float],
        strength: float,
        fraction: float,
    ) -> np.ndarray:
        fraction = max(0.0, min(1.0, fraction))
        if fraction <= 0:
            return np.zeros_like(frame)

        # Anchor on the right-most endpoint and grow toward the other end.
        if start[0] >= end[0]:
            anchor, target = start, end
        else:
            anchor, target = end, start
        segment_end = (
            anchor[0] + (target[0] - anchor[0]) * fraction,
            anchor[1] + (target[1] - anchor[1]) * fraction,
        )

        base_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.line(base_mask, to_int(anchor), to_int(segment_end), 255, dfl.LINE_THICKNESS, cv2.LINE_AA)
        inner = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        outer = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        ring1 = cv2.subtract(inner, base_mask)
        ring2 = cv2.subtract(outer, inner)

        color_vec = np.array(color, dtype=np.float32)
        layer = np.zeros_like(frame, dtype=np.float32)
        layer += base_mask[:, :, None] * (color_vec * (2.5 * strength) / 255.0)
        layer += ring1[:, :, None] * (color_vec * (1.2 * strength) / 255.0)
        layer += ring2[:, :, None] * (color_vec * (0.7 * strength) / 255.0)
        return np.clip(layer, 0, 255).astype(np.uint8)

    def combine_layers(layers: List[np.ndarray]) -> np.ndarray:
        combined = np.zeros_like(frame)
        for layer in layers:
            mask = layer.max(axis=2) > 0
            combined[mask] = layer[mask]
        return combined

    def build_glow_frame(strength: float, fraction: float) -> np.ndarray:
        if fraction <= 0:
            return frame.copy()
        # Draw green first, then red on top so the draw arm sits above the DFL.
        layers = [
            line_effect(dfl.DFL_COLOR, geometry.adjusted_bow, geometry.dfl_end, strength, fraction),
            line_effect(dfl.DRAW_ARM_COLOR, geometry.draw_line_start, geometry.draw_line_end, strength, fraction),
        ]
        glow = combine_layers(layers)
        return cv2.addWeighted(frame, 1.0, glow, 1.0, 0.0)

    def build_draw_length_frame(strength: float, fraction: float) -> np.ndarray:
        if fraction <= 0:
            return frame.copy()
        start = geometry.left_wrist_raw
        end = geometry.right_wrist_raw
        anchor = start if start[0] >= end[0] else end
        target = end if anchor is start else start
        segment_end = (
            anchor[0] + (target[0] - anchor[0]) * fraction,
            anchor[1] + (target[1] - anchor[1]) * fraction,
        )
        base_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.line(base_mask, to_int(anchor), to_int(segment_end), 255, dfl.LINE_THICKNESS + 1, cv2.LINE_AA)
        inner = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        outer = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        ring1 = cv2.subtract(inner, base_mask)
        ring2 = cv2.subtract(outer, inner)
        color_vec = np.array((0, 255, 255), dtype=np.float32)  # bright yellow
        layer = np.zeros_like(frame, dtype=np.float32)
        layer += base_mask[:, :, None] * (color_vec * (2.3 * strength) / 255.0)
        layer += ring1[:, :, None] * (color_vec * (1.0 * strength) / 255.0)
        layer += ring2[:, :, None] * (color_vec * (0.55 * strength) / 255.0)
        glow = np.clip(layer, 0, 255).astype(np.uint8)
        return cv2.addWeighted(frame, 1.0, glow, 1.0, 0.0)

    # Create a short glow-up sequence with increasingly intense halos.
    sequence_paths: List[Path] = []
    intensities = [1.1, 1.25, 1.4, 1.55, 1.7]
    fractions = [0.0, 0.35, 0.6, 0.85, 1.0]
    for idx, (strength, fraction) in enumerate(zip(intensities, fractions), start=1):
        blended = build_glow_frame(strength, fraction)
        seq_path = output_path.with_name(f"{output_path.stem}_step{idx}{output_path.suffix}")
        cv2.imwrite(str(seq_path), blended)
        sequence_paths.append(seq_path)

    # Draw-length-only sequence (highlight yellow connector only).
    draw_length_sequence: List[Path] = []
    length_intensities = [1.0, 1.15, 1.35, 1.55, 1.75]
    for idx, (strength, fraction) in enumerate(zip(length_intensities, fractions), start=1):
        length_frame = build_draw_length_frame(strength, fraction)
        seq_path = draw_length_output.with_name(f"{pose_path.stem}_draw_length_step{idx}{draw_length_output.suffix}")
        cv2.imwrite(str(seq_path), length_frame)
        draw_length_sequence.append(seq_path)

    # Save the static DFL image using the brightest glow and the dashed connector.
    final_overlay = build_glow_frame(intensities[-1], 1.0)
    dfl.draw_lines_and_text(final_overlay, (0, 0), geometry)
    cv2.imwrite(str(output_path), final_overlay)
    final_draw_length = build_draw_length_frame(length_intensities[-1], 1.0)
    cv2.imwrite(str(draw_length_output), final_draw_length)

    (
        angle_stats,
        length_stats,
        angle_count,
        length_count,
        _,
        _,
    ) = dfl.compute_dataset_stats(pose_path, dfl.DEFAULT_DATA_DIR)
    return DrawForceSummary(
        image_path=output_path,
        sequence_paths=sequence_paths,
        draw_length_image=draw_length_output,
        draw_length_sequence=draw_length_sequence,
        geometry=geometry,
        angle_stats=angle_stats,
        length_stats=length_stats,
        angle_count=angle_count,
        length_count=length_count,
    )


def run_post_release_analysis(pose_path: Path, video_path: Path, force: bool) -> PostReleaseSummary:
    viz_path = POST_VIZ_DIR / f"{pose_path.stem}_post.png"
    report_path = POST_VIZ_DIR / f"{pose_path.stem}_post.json"
    run_command(
        "post_release_analysis",
        [
            sys.executable,
            "post_release_analysis.py",
            "--pose",
            str(pose_path),
            "--video",
            str(video_path),
            "--output",
            str(viz_path),
            "--json-report",
            str(report_path),
        ],
    )
    if not report_path.exists():
        raise FileNotFoundError(f"Post-release report missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    release_frame = int(payload.get("release_frame") or payload.get("pre_frame") or 1)
    follow_frame = int(payload.get("follow_frame") or release_frame)

    release_seq: List[Path] = []
    release_still: Optional[Path] = None
    follow_seq: List[Path] = []
    follow_still: Optional[Path] = None
    try:
        kp_release, pts_release = load_frame_points(pose_path, release_frame)
        kp_follow, pts_follow = load_frame_points(pose_path, follow_frame)
        release_frame_img = dfl.read_frame(video_path, release_frame)
        follow_frame_img = dfl.read_frame(video_path, follow_frame)

        release_crop_box = pra_compute_crop_box(release_frame_img.shape, pts_release)
        release_cropped, release_offset = crop_frame(release_frame_img, release_crop_box)
        follow_crop_box = pra_compute_crop_box(follow_frame_img.shape, pts_follow)
        follow_cropped, follow_offset = crop_frame(follow_frame_img, follow_crop_box)

        roles_release = resolve_draw_bow_roles(kp_release)
        roles_follow = resolve_draw_bow_roles(kp_follow)
        bow_wrist_release = kp_release.get(roles_release["bow_wrist"])
        bow_shoulder_release = kp_release.get(roles_release["bow_shoulder"])
        bow_wrist_follow = kp_follow.get(roles_follow["bow_wrist"])
        bow_shoulder_follow = kp_follow.get(roles_follow["bow_shoulder"])

        left_line = (
            (bow_wrist_release[0] - release_offset[0], bow_wrist_release[1] - release_offset[1]),
            (bow_shoulder_release[0] - release_offset[0], bow_shoulder_release[1] - release_offset[1]),
        ) if bow_wrist_release and bow_shoulder_release else None
        right_line = (
            (bow_wrist_follow[0] - follow_offset[0], bow_wrist_follow[1] - follow_offset[1]),
            (bow_shoulder_follow[0] - follow_offset[0], bow_shoulder_follow[1] - follow_offset[1]),
        ) if bow_wrist_follow and bow_shoulder_follow else None

        if left_line or right_line:
            release_base = POST_VIZ_DIR / f"{pose_path.stem}_release.png"
            release_seq, release_still = generate_side_by_side_sequence(
                release_cropped,
                follow_cropped,
                left_line,
                right_line,
                (0, 220, 0),
                (0, 220, 0),
                release_base,
            )
        # Follow-through: highlight draw wrist only (both panels)
        draw_wrist = kp_release.get(roles_release["draw_wrist"])
        draw_wrist_follow = kp_follow.get(roles_release["draw_wrist"])
        if draw_wrist or draw_wrist_follow:
            follow_base = POST_VIZ_DIR / f"{pose_path.stem}_follow.png"
            follow_seq, follow_still = generate_side_by_side_sequence(
                release_cropped,
                follow_cropped,
                None,
                None,
                (0, 165, 255),
                (0, 165, 255),
                follow_base,
                left_dot=(draw_wrist[0] - release_offset[0], draw_wrist[1] - release_offset[1]) if draw_wrist else None,
                right_dot=(draw_wrist_follow[0] - follow_offset[0], draw_wrist_follow[1] - follow_offset[1]) if draw_wrist_follow else None,
            )
    except Exception as exc:
        print(f"⚠️  Failed to build release/follow frames: {exc}")

    # Fallback: if sequences weren't built (e.g., earlier errors), try to pick up any existing files.
    if not release_still:
        candidate = POST_VIZ_DIR / f"{pose_path.stem}_release.png"
        if candidate.exists():
            release_still = candidate
            release_seq = sorted(candidate.parent.glob(f"{pose_path.stem}_release_step*.png"))
    if not follow_still:
        candidate = POST_VIZ_DIR / f"{pose_path.stem}_follow.png"
        if candidate.exists():
            follow_still = candidate
            follow_seq = sorted(candidate.parent.glob(f"{pose_path.stem}_follow_step*.png"))

    return PostReleaseSummary(
        image_path=viz_path,
        report_path=report_path,
        payload=payload,
        release_image=release_still,
        follow_image=follow_still,
        release_sequence=release_seq,
        follow_sequence=follow_seq,
    )


def copy_artifact(src: Path, dest: Path) -> Path:
    src = src.resolve()
    dest = dest.resolve()
    if src == dest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"🔄 Copied {src} → {dest}")
    return dest


def to_blockquote(text: str) -> str:
    lines = []
    for raw in text.strip().splitlines():
        if raw.strip():
            lines.append(f"> {raw}")
        else:
            lines.append(">")
    return "\n".join(lines)


def format_frames_and_seconds(frame_val: Optional[float], seconds: Optional[float]) -> str:
    if frame_val is None:
        return "N/A"
    if seconds is None:
        return f"{frame_val:.0f} frames"
    return f"{frame_val:.0f} frames (~{seconds:.2f}s)"


def write_report(
    report_path: Path,
    video_id: str,
    training_video: Path,
    inference_json: Path,
    inference_video: Path,
    label_path: Path,
    smoothed_path: Path,
    reference_id: str,
    final_video_copy: Path,
    release_summary: ReleaseSummary,
    ghost_report: Optional[str],
    spine_copy: Path,
    spine_response: Optional[str],
    dfl_summary: DrawForceSummary,
    dfl_copy: Path,
    dfl_sequence: Optional[List[Path]],
    dfl_report: Optional[str],
    post_summary: PostReleaseSummary,
    post_copy: Path,
    post_report: Optional[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    angle_stats_text = (
        f"{dfl_summary.angle_stats[0]:.2f}° ± {dfl_summary.angle_stats[1]:.2f}°"
        if dfl_summary.angle_stats
        else "N/A"
    )
    length_stats_text = (
        f"{dfl_summary.length_stats[0]:.2f} ± {dfl_summary.length_stats[1]:.2f} hip-widths"
        if dfl_summary.length_stats
        else "N/A"
    )
    draw_length_text = (
        f"{dfl_summary.geometry.normalized_draw_length:.2f} hip-widths"
        if dfl_summary.geometry.normalized_draw_length is not None
        else "unavailable (missing hip width)"
    )
    spine_section = spine_response or "Spine straightness prompt did not return a response."
    draw_ratio = post_summary.payload.get("draw_ratio")
    draw_ratio_text = (
        f"{float(draw_ratio) * 100.0:.1f}% of frames behind the head"
        if isinstance(draw_ratio, (float, int))
        else "insufficient data"
    )
    bow_drop = post_summary.payload.get("bow_angle_drop")
    bow_drop_text = (
        f"{float(bow_drop):.1f}° change"
        if isinstance(bow_drop, (float, int))
        else "insufficient data"
    )
    lines: List[str] = [
        f"# {video_id} Analysis Report",
        "",
        "## Workflow Outputs",
        f"- Source video: `{training_video}`",
        f"- Remote inference JSON: `{inference_json}`",
        f"- Remote inference video: `{inference_video}`",
        f"- VLM labels: `{label_path}`",
        f"- Smoothed pose: `{smoothed_path}`",
        f"- Ghost reference: `results_{reference_id}.json`",
        f"- Stream-ready ghost video: `{final_video_copy.name}`",
        "",
        "## Release Timing (Ghost Overlay)",
        f"- Draw frame: {release_summary.draw_frame if release_summary.draw_frame is not None else 'N/A'}",
        f"- Release frame: {format_frames_and_seconds(release_summary.release_frame, release_summary.release_seconds)}",
        f"- Draw→release window: {format_frames_and_seconds(release_summary.draw_to_release_frames, release_summary.draw_to_release_seconds)}",
        f"- Dataset release avg ({release_summary.dataset_count} files): "
        f"{format_frames_and_seconds(release_summary.dataset_mean_frame, release_summary.dataset_mean_seconds)}",
        f"- Dataset release σ: "
        f"{format_frames_and_seconds(release_summary.dataset_std_frame, release_summary.dataset_std_seconds)}",
        "",
        "## Ghost Timing Coaching",
        to_blockquote(ghost_report) if ghost_report else "Ghost timing analysis not available.",
        "",
        "## Spine Straightness",
        to_blockquote(spine_section),
        "",
        f"![Spine straightness]({spine_copy.name})",
        "",
        "## Draw-Force Line",
        f"- DFL angle: {dfl_summary.geometry.angle_deg:.2f}°",
        f"- Draw length: {draw_length_text} (raw {dfl_summary.geometry.draw_length_px:.1f}px)",
        f"- Other angles ({dfl_summary.angle_count} files): {angle_stats_text}",
        f"- Other draw lengths ({dfl_summary.length_count} files): {length_stats_text}",
        "",
        f"![Draw-force line]({dfl_copy.name})",
        "",
        "## Draw-Force Coaching",
        to_blockquote(dfl_report) if dfl_report else "Draw-force analysis not available.",
        "",
        "## Post-Release Follow-Through",
        f"- Release frame analyzed: {post_summary.payload.get('release_frame', 'N/A')}",
        f"- Frames considered: {post_summary.payload.get('analyzed_frames', 'N/A')}",
        f"- Draw-hand status: {'⚠️' if post_summary.payload.get('draw_warning') else '✅'} {draw_ratio_text}",
        f"- Bow-arm status: {'⚠️' if post_summary.payload.get('bow_warning') else '✅'} {bow_drop_text}",
        "",
        f"![Post-release]({post_copy.name})",
        "",
        "## Post-Release Coaching",
        to_blockquote(post_report) if post_report else "Post-release analysis not available.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STRAIGHT video analysis pipeline.")
    parser.add_argument("--input-video", required=True, type=Path, help="Path to the source video.")
    parser.add_argument(
        "--video-id",
        help="Slug to use for intermediate files (defaults to the input video stem).",
    )
    parser.add_argument(
        "--reference-id",
        default="nfst008",
        help="Reference dataset for the ghost overlay (defaults to nfst008).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STREAM_OUTPUT_DIR,
        help="Directory that will hold the final ghost video and report.",
    )
    parser.add_argument(
        "--final-video-name",
        help="Optional filename for the copied ghost video (defaults to <video_id>_ghost.mp4).",
    )
    parser.add_argument(
        "--report-name",
        help="Optional filename for the Markdown report (defaults to <video_id>_analysis_report.md).",
    )
    parser.add_argument(
        "--use-st-prompt",
        action="store_true",
        help="Use the ST-specific VLM prompt (otherwise you'll be prompted).",
    )
    parser.add_argument("--enable-vlm-debug", action="store_true", help="Enable VLM debug confirmations.")
    parser.add_argument("--skip-remote", action="store_true", help="Reuse the existing inference JSON.")
    parser.add_argument("--skip-vlm", action="store_true", help="Reuse the existing estimated label JSON.")
    parser.add_argument("--skip-smoothing", action="store_true", help="Reuse the existing smoothed pose JSON.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every artifact even if files already exist.",
    )
    args = parser.parse_args()

    video_id = normalize_video_id(args.video_id or args.input_video.stem)
    reference_id = normalize_video_id(args.reference_id)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_video = transcode_to_training_video(video_id, args.input_video, args.force)
    inference_json, remote_video = run_remote_inference(video_id, args.force, args.skip_remote)
    use_st_prompt = args.use_st_prompt
    label_path = run_vlm_phase_estimation(
        video_id,
        use_st_prompt,
        args.enable_vlm_debug,
        args.force,
        args.skip_vlm,
    )
    smoothed_path = run_kalman_smoothing(video_id, args.force, args.skip_smoothing)
    processed_video, crop_offset = crop_and_banner_video(video_id, training_video, smoothed_path, label_path, args.force)
    window_start = 0
    clip_frame_count = 0
    total_frames = 0
    meta_path = processed_video.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            clip_frame_count = int(meta.get("clip_frame_count") or 0)
            total_frames = int(meta.get("source_frame_count") or 0)
        except Exception:
            pass
    if clip_frame_count == 0 or total_frames == 0:
        try:
            import cv2

            cap = cv2.VideoCapture(str(processed_video))
            if cap.isOpened():
                total_frames = total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            pass
    clip_frame_count = clip_frame_count or total_frames
    pose_for_video = adjust_pose_for_crop(smoothed_path, crop_offset, window_start, clip_frame_count)
    label_cropped = Path("estimated_labels") / f"vlm_estimated_label_{pose_for_video.stem.replace('results_', '')}.json"

    skeleton_overlay_path = run_skeleton_overlay(pose_for_video, processed_video, args.force)
    skeleton_copy = copy_artifact(skeleton_overlay_path, args.output_dir / f"{video_id}_skeleton.mp4")

    reference_pose = SMOOTHED_DIR / f"results_{reference_id}.json"
    ghost_video = ensure_ghost_overlay(
        pose_for_video,
        reference_pose,
        args.force,
        actual_video=processed_video,
        crop_offset=None,
        asset_video_id=video_id,
        report_video_id=smoothed_path.stem,
    )

    fps = detect_video_fps(processed_video)
    release_summary = compute_release_summary(smoothed_path, fps)
    ghost_report = read_ghost_report(smoothed_path)
    spine_path, spine_response = run_spine_analysis(pose_for_video, processed_video, args.force)
    dfl_summary = run_draw_force_analysis(pose_for_video, processed_video, args.force)
    dfl_angle_report_path, dfl_length_report_path = write_dfl_reports(
        pose_for_video.stem,
        dfl_summary.geometry,
        dfl_summary.angle_stats,
        dfl_summary.length_stats,
        dfl_summary.image_path,
        dfl_summary.angle_count,
        dfl_summary.length_count,
    )
    dfl_report = read_dfl_report(pose_for_video)
    post_summary = run_post_release_analysis(pose_for_video, processed_video, args.force)
    post_draw_report_path, post_bow_report_path, post_report_path = write_post_release_reports(
        pose_for_video.stem, post_summary, post_summary.image_path
    )
    post_summary.draw_markdown_path = post_draw_report_path
    post_summary.bow_markdown_path = post_bow_report_path
    if post_report_path:
        post_summary.markdown_path = post_report_path
    post_report = read_post_release_report(pose_for_video)

    final_video_name = args.final_video_name or f"{video_id}_ghost.mp4"
    final_video_copy = copy_artifact(ghost_video, output_dir / final_video_name)
    spine_copy = copy_artifact(spine_path, output_dir / spine_path.name)
    dfl_copy = copy_artifact(dfl_summary.image_path, output_dir / dfl_summary.image_path.name)
    post_copy = copy_artifact(post_summary.image_path, output_dir / post_summary.image_path.name)
    ghost_report_path = GHOST_ANALYSIS_DIR / f"{smoothed_path.stem}.md"
    if ghost_report_path.exists():
        copy_artifact(ghost_report_path, output_dir / ghost_report_path.name)
    angle_report_path = DFL_ANGLE_ANALYSIS_DIR / f"{smoothed_path.stem}.md"
    if angle_report_path.exists():
        copy_artifact(angle_report_path, output_dir / angle_report_path.name)
    length_report_path = DFL_LENGTH_ANALYSIS_DIR / f"{smoothed_path.stem}.md"
    if length_report_path.exists():
        copy_artifact(length_report_path, output_dir / length_report_path.name)
    post_analysis_report_path = post_summary.markdown_path or POST_ANALYSIS_DIR / f"{pose_for_video.stem}.md"
    if post_analysis_report_path and post_analysis_report_path.exists():
        copy_artifact(post_analysis_report_path, output_dir / post_analysis_report_path.name)
    if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists():
        copy_artifact(post_summary.draw_markdown_path, output_dir / post_summary.draw_markdown_path.name)
    if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists():
        copy_artifact(post_summary.bow_markdown_path, output_dir / post_summary.bow_markdown_path.name)

    report_name = args.report_name or f"{video_id}_analysis_report.md"
    report_path = output_dir / report_name
    write_report(
        report_path,
        video_id,
        processed_video,
        inference_json,
        remote_video,
        label_path,
        smoothed_path,
        reference_id,
        final_video_copy,
        release_summary,
        ghost_report,
        spine_copy,
        spine_response,
        dfl_summary,
        dfl_copy,
        dfl_summary.sequence_paths,
        dfl_report,
        post_summary,
        post_copy,
        post_report,
    )

    spine_report_path = None
    if spine_response:
        try:
            spine_report_path = WEB_ASSETS_RUNS_DIR / video_id / "spine.md"
            spine_report_path.parent.mkdir(parents=True, exist_ok=True)
            spine_report_path.write_text(spine_response.strip(), encoding="utf-8")
        except Exception as exc:
            print(f"⚠️  Failed to write spine markdown to assets: {exc}")

    export_run_assets(
        video_id,
        processed_video,
        final_video_copy,
        skeleton_copy,
        spine_copy,
        dfl_copy,
        dfl_summary.sequence_paths,
        dfl_summary.draw_length_image,
        dfl_summary.draw_length_sequence,
        post_summary.release_image,
        post_summary.release_sequence,
        post_summary.follow_image,
        post_summary.follow_sequence,
        post_copy,
        report_path,
        ghost_report_path if ghost_report_path.exists() else None,
        spine_report_path if spine_report_path and spine_report_path.exists() else None,
        dfl_angle_report_path if dfl_angle_report_path and dfl_angle_report_path.exists() else None,
        dfl_length_report_path if dfl_length_report_path and dfl_length_report_path.exists() else None,
        post_summary.markdown_path if post_summary.markdown_path and post_summary.markdown_path.exists() else None,
        post_summary.draw_markdown_path if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists() else None,
        post_summary.bow_markdown_path if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists() else None,
    )
    meta_path = processed_video.with_suffix(".meta.json") if processed_video.with_suffix(".meta.json").exists() else None
    update_runs_manifest(
        video_id,
        label_path,
        meta_path,
        ghost_report_path if ghost_report_path.exists() else None,
        spine_report_path if spine_report_path and spine_report_path.exists() else None,
        dfl_angle_report_path if dfl_angle_report_path and dfl_angle_report_path.exists() else None,
        dfl_length_report_path if dfl_length_report_path and dfl_length_report_path.exists() else None,
        post_summary.markdown_path if post_summary.markdown_path and post_summary.markdown_path.exists() else None,
        post_summary.draw_markdown_path if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists() else None,
        post_summary.bow_markdown_path if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists() else None,
        dfl_summary.sequence_paths,
        dfl_summary.draw_length_sequence,
        post_summary.release_sequence,
        post_summary.follow_sequence,
        post_summary.release_image,
        post_summary.follow_image,
    )

    bundle_run_outputs(
        video_id,
        training_video,
        processed_video,
        inference_json,
        smoothed_path,
        pose_for_video,
        label_path,
        label_cropped,
        final_video_copy,
        skeleton_copy,
        spine_copy,
        dfl_copy,
        dfl_summary.sequence_paths,
        dfl_summary.draw_length_image,
        dfl_summary.draw_length_sequence,
        post_summary.release_image,
        post_summary.release_sequence,
        post_summary.follow_image,
        post_summary.follow_sequence,
        post_copy,
        report_path,
        ghost_report_path if ghost_report_path.exists() else None,
        post_summary.markdown_path if post_summary.markdown_path and post_summary.markdown_path.exists() else None,
        post_summary.draw_markdown_path if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists() else None,
        post_summary.bow_markdown_path if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists() else None,
        [p for p in (angle_report_path if 'angle_report_path' in locals() else None, length_report_path if 'length_report_path' in locals() else None) if p and p.exists()],
        processed_video.with_suffix(".meta.json") if processed_video.with_suffix(".meta.json").exists() else None,
    )


if __name__ == "__main__":
    main()
