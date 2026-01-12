#!/usr/bin/env python3
"""Overlay skeleton keypoints back onto the source video for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV (cv2) is required. Install it via `pip install opencv-python`."
    ) from exc

POSE_DIR = Path("inference_data")
SMOOTHED_DIR = Path("smoothed_inference_data")
VIDEO_DIR = Path("training_videos")
LABEL_DIR = Path("estimated_labels")
OUTPUT_DIR = Path("skeleton_visualizations")
PRE_DRAW_FRAME_BUFFER = 30
POST_RELEASE_FRAME_BUFFER = 30


def list_pose_files(extra_dirs: Optional[Iterable[Path]] = None) -> List[Path]:
    dirs = [POSE_DIR]
    if extra_dirs:
        dirs.extend(extra_dirs)
    files: List[Path] = []
    for folder in dirs:
        if folder.exists():
            files.extend(folder.glob("*.json"))
    return sorted({path.resolve() for path in files})


def prompt_for_pose_file() -> Path:
    files = list_pose_files(extra_dirs=[SMOOTHED_DIR])
    if not files:
        raise FileNotFoundError("No inference JSONs found.")
    print("📁 Available inference JSON files:")
    for idx, path in enumerate(files, 1):
        location = path.parent.name
        print(f"  [{idx:02d}] ({location}) {path.name}")
    while True:
        raw = input("Select file by number or enter a path: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(files):
                return files[idx - 1]
            print("❌ Invalid selection. Try again.")
            continue
        candidate = Path(raw)
        if candidate.is_file():
            return candidate.resolve()
        candidate = POSE_DIR / raw
        if candidate.is_file():
            return candidate.resolve()
        candidate = SMOOTHED_DIR / raw
        if candidate.is_file():
            return candidate.resolve()
        print("❌ Could not find that file. Try again.\n")


def infer_video_path(pose_path: Path) -> Optional[Path]:
    stem = pose_path.stem
    if stem.startswith("results_"):
        stem = stem.split("results_", 1)[1]
    video_candidate = VIDEO_DIR / f"{stem}.mp4"
    if video_candidate.exists():
        return video_candidate
    return None


def infer_dataset_name(pose_path: Path) -> Optional[str]:
    stem = pose_path.stem
    if stem.startswith("results_"):
        stem = stem.split("results_", 1)[1]
    return stem or None


def _phase_start_idx0(phase: dict, label_data: dict) -> Optional[int]:
    if "start_frame" in phase:
        return max(0, int(phase["start_frame"]))
    fps = float(label_data.get("fps", 60.0))
    start = float(phase.get("start", 0.0))
    return max(0, int(round(start * fps)))


def load_phase_bounds(dataset: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    label_path = LABEL_DIR / f"vlm_estimated_label_{dataset}.json"
    if not label_path.exists() and dataset.endswith("_cropped"):
        fallback = dataset.rsplit("_cropped", 1)[0]
        fallback_path = LABEL_DIR / f"vlm_estimated_label_{fallback}.json"
        if fallback_path.exists():
            label_path = fallback_path
    if not label_path.exists():
        print(f"⚠️  Missing estimated label JSON: {label_path}")
        return None, None, None
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    draw_idx: Optional[int] = None
    release_idx: Optional[int] = None
    for phase in data.get("phases", []):
        name = str(phase.get("name", "")).lower()
        if draw_idx is None and "draw" in name:
            draw_idx = _phase_start_idx0(phase, data)
        if release_idx is None and "release" in name:
            release_idx = _phase_start_idx0(phase, data)
    frame_count = None
    if "frame_count" in data:
        try:
            frame_count = max(0, int(data["frame_count"]))
        except (TypeError, ValueError):
            frame_count = None
    if release_idx is None:
        print(f"⚠️  No release phase found in {label_path.name}")
    return draw_idx, release_idx, frame_count


def compute_frame_window(
    pose_path: Path,
    frames: Dict[int, dict],
) -> Tuple[Optional[int], Optional[int]]:
    dataset = infer_dataset_name(pose_path)
    if not dataset:
        return None, None
    draw_idx0, release_idx0, total_frames = load_phase_bounds(dataset)
    if total_frames is None and frames:
        total_frames = max(frames.keys())
    start_frame_id: Optional[int] = None
    end_frame_id: Optional[int] = None
    if draw_idx0 is not None:
        start_frame_id = max(0, draw_idx0 - PRE_DRAW_FRAME_BUFFER) + 1
    if release_idx0 is not None:
        end_frame_id = release_idx0 + POST_RELEASE_FRAME_BUFFER + 1
    elif total_frames is not None:
        end_frame_id = total_frames
    if start_frame_id is not None and start_frame_id < 1:
        start_frame_id = 1
    if end_frame_id is not None and total_frames is not None:
        end_frame_id = min(end_frame_id, total_frames)
    return start_frame_id, end_frame_id


def apply_frame_window(
    frames: Dict[int, dict],
    start_frame_id: Optional[int],
    end_frame_id: Optional[int],
) -> Dict[int, dict]:
    if start_frame_id is None and end_frame_id is None:
        return frames
    filtered: Dict[int, dict] = {}
    for frame_id, frame in frames.items():
        if start_frame_id is not None and frame_id < start_frame_id:
            continue
        if end_frame_id is not None and frame_id > end_frame_id:
            continue
        filtered[frame_id] = frame
    if not filtered:
        print("⚠️  No frames remain after applying the window; using original frames instead.")
        return frames
    return filtered


def extract_color_table(entry, expected_len: int, default_color: Tuple[int, int, int]):
    if entry is None:
        return [default_color for _ in range(expected_len)]
    if isinstance(entry, dict) and "__ndarray__" in entry:
        data = entry.get("__ndarray__", [])
    else:
        data = entry
    colors: List[Tuple[int, int, int]] = []
    for row in data:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            colors.append(tuple(int(c) for c in row[:3]))
        else:
            colors.append(default_color)
    if len(colors) < expected_len:
        colors.extend([default_color] * (expected_len - len(colors)))
    return colors[:expected_len]


def load_pose_json(pose_path: Path):
    with open(pose_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta_info", {})
    keypoints = int(meta.get("num_keypoints", 0))
    skeleton_links = meta.get("skeleton_links", [])
    skeleton_links = [tuple(int(i) for i in link) for link in skeleton_links]
    keypoint_colors = extract_color_table(
        meta.get("keypoint_colors"), keypoints, (0, 255, 0)
    )
    link_colors = extract_color_table(
        meta.get("skeleton_link_colors"), len(skeleton_links), (255, 128, 0)
    )
    frames = {}
    for frame in data.get("instance_info", []):
        frame_id = int(frame.get("frame_id", 1))
        frames[frame_id] = frame
    return meta, frames, keypoint_colors, link_colors, skeleton_links


def draw_instances(
    image,
    instances,
    keypoint_colors,
    link_colors,
    skeleton_links,
    score_threshold: float,
    point_radius: int,
):
    for inst in instances:
        keypoints = inst.get("smoothed_keypoints") or inst.get("keypoints") or []
        scores = inst.get("keypoint_scores") or []
        for idx, point in enumerate(keypoints):
            if len(point) != 2:
                continue
            score = scores[idx] if idx < len(scores) else 1.0
            if score < score_threshold:
                continue
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(image, (x, y), point_radius, keypoint_colors[idx % len(keypoint_colors)], -1)
        for link_idx, (start, end) in enumerate(skeleton_links):
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            start_score = scores[start] if start < len(scores) else 1.0
            end_score = scores[end] if end < len(scores) else 1.0
            if start_score < score_threshold or end_score < score_threshold:
                continue
            x1, y1 = keypoints[start]
            x2, y2 = keypoints[end]
            cv2.line(
                image,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                link_colors[link_idx % len(link_colors)],
                2,
            )


def visualize(
    pose_path: Path,
    video_path: Path,
    out_path: Path,
    score_threshold: float,
    point_radius: int,
) -> None:
    meta, frames, keypoint_colors, link_colors, skeleton_links = load_pose_json(pose_path)
    if not frames:
        raise ValueError("Pose JSON has no frames to visualize.")
    start_frame_id = min(frames.keys())
    end_frame_id = max(frames.keys())
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    OUTPUT_DIR.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("⚠️  AVC1 codec unavailable, falling back to MP4V.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video at {out_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(
        f"🎥 {video_path.name}: {fps:.2f} fps | {width}x{height} | {total_frames} frames"
    )
    sorted_frame_ids = sorted(frames.keys())
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame_id - 1))
    current_frame_id = start_frame_id
    pose_idx = 0
    while current_frame_id <= end_frame_id:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Ran out of video frames before finishing pose window.")
            break
        pose_frame = frames.get(current_frame_id)
        if pose_frame:
            draw_instances(
                frame,
                pose_frame.get("instances", []),
                keypoint_colors,
                link_colors,
                skeleton_links,
                score_threshold,
                point_radius,
            )
        writer.write(frame)
        if pose_frame:
            pose_idx += 1
            if pose_idx % 250 == 0:
                print(f"  Processed {pose_idx} pose frames...")
        current_frame_id += 1
    cap.release()
    writer.release()
    print(f"✅ Saved overlay video → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 2D skeletons back onto a video.")
    parser.add_argument("--pose", required=True, type=Path, help="Pose JSON path.")
    parser.add_argument(
        "--video",
        type=Path,
        help="Source video path (defaults to training_videos/<stem>.mp4 if available).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path (defaults to skeleton_visualizations/<pose_stem>.mp4).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.2,
        help="Minimum keypoint score required to draw points/segments.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=4,
        help="Radius (in pixels) for keypoint circles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_path = args.pose.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    candidate_video = args.video.expanduser() if args.video else infer_video_path(pose_path)
    if candidate_video is None or not candidate_video.exists():
        raise FileNotFoundError(
            "Source video not found. Provide --video explicitly or ensure training_videos/<stem>.mp4 exists."
        )
    video_path = candidate_video
    if args.output is None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"{pose_path.stem}.mp4"
    else:
        out_path = args.output.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    visualize(
        pose_path,
        video_path,
        out_path,
        score_threshold=max(0.0, args.score_threshold),
        point_radius=max(1, args.point_radius),
    )


if __name__ == "__main__":
    print("🎯 Skeleton overlay visualizer")
    main()
