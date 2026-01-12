#!/usr/bin/env python3
"""Visualize foot alignment vs head position on the pre-release frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("OpenCV (cv2) is required. Install it via `pip install opencv-python`.") from exc

from rag_gpt_client import call_rag_assistant
from visualize_skeleton import infer_video_path, load_pose_json

OUTPUT_DIR = Path("spine_visualizations")
DEBUG_DIR = OUTPUT_DIR / "debug"
ANALYSIS_DIR = Path("spine_straight_analysis")
NOTES_PATH = Path("config/notes/spine_straight_notes.txt")
PROMPT_SPINE = Path("config/prompts/spine.md")
SPINE_SYSTEM_PROMPT = "You are an archery technique coach."
LINE_COLOR = (0, 255, 0)
HEAD_COLOR = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw foot-aligned guides on the release frame.")
    parser.add_argument("--pose", required=True, type=Path, help="Path to smoothed pose JSON.")
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional source video path (defaults to training_videos/<pose_stem>.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (defaults to spine_visualizations/<pose_stem>_spine.png).",
    )
    parser.add_argument("--head-radius", type=int, default=80, help="Approximate head circle radius in pixels.")
    return parser.parse_args()


def get_release_frame(path: Path) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    info = data.get("smoothing_info") or {}
    release = info.get("release_frame")
    if isinstance(release, int):
        return max(1, release)
    release = info.get("window_end_frame")
    if isinstance(release, int):
        return max(1, release)
    frames = data.get("instance_info") or []
    if frames:
        return int(frames[-1].get("frame_id", 1))
    return None


def extract_keypoints(
    meta, frames, frame_id: int
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[float]]:
    frame = frames.get(frame_id)
    if not frame:
        return None, None, None, None
    instances = frame.get("instances", [])
    if not instances:
        return None, None, None, None
    points = instances[0].get("smoothed_keypoints") or instances[0].get("keypoints") or []
    name2id = meta.get("keypoint_name2id") or {}
    left = name2id.get("left_ankle") or name2id.get("left_foot")
    right = name2id.get("right_ankle") or name2id.get("right_foot")
    nose_idx = name2id.get("nose")
    left_eye_idx = name2id.get("left_eye")
    right_eye_idx = name2id.get("right_eye")
    head = nose_idx if nose_idx is not None else name2id.get("left_eye") or name2id.get("right_eye")
    hips = None
    head_pt = None
    head_scale = None
    nose_pt = None
    if left is not None and right is not None and left < len(points) and right < len(points):
        hips = (
            (float(points[left][0]), float(points[left][1])),
            (float(points[right][0]), float(points[right][1])),
        )
    if head is not None and head < len(points):
        head_pt = (float(points[head][0]), float(points[head][1]))
    eye_pts = []
    if nose_idx is not None and nose_idx < len(points):
        if len(points[nose_idx]) >= 2:
            nose_pt = (float(points[nose_idx][0]), float(points[nose_idx][1]))
    for idx in (left_eye_idx, right_eye_idx):
        if idx is not None and idx < len(points) and len(points[idx]) >= 2:
            eye_pts.append((float(points[idx][0]), float(points[idx][1])))
    if nose_pt and eye_pts:
        # Use average eye position for stability if both are present.
        ex = sum(pt[0] for pt in eye_pts) / len(eye_pts)
        ey = sum(pt[1] for pt in eye_pts) / len(eye_pts)
        head_scale = math.hypot(nose_pt[0] - ex, nose_pt[1] - ey)
    return hips, head_pt, nose_pt, head_scale


def save_head_debug(frame, center_x: int, center_y: int, radius: int, stem: str) -> Optional[Path]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    zoom = max(radius, 1) * 2
    x1 = max(0, center_x - zoom)
    x2 = min(frame.shape[1], center_x + zoom)
    y1 = max(0, center_y - zoom)
    y2 = min(frame.shape[0], center_y + zoom)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    debug_path = DEBUG_DIR / f"{stem}_head_zoom.png"
    cv2.imwrite(str(debug_path), crop)
    return debug_path


def load_spine_prompt() -> str:
    if not PROMPT_SPINE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_SPINE}")
    text = PROMPT_SPINE.read_text().strip()
    if not text:
        raise ValueError(f"Prompt file {PROMPT_SPINE} is empty.")
    return text


def load_spine_notes() -> str:
    if not NOTES_PATH.exists():
        return ""
    return NOTES_PATH.read_text(encoding="utf-8").strip()


def append_spine_note(note: str) -> None:
    note = note.strip()
    if not note:
        return
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{note}\n")


def classify_spine_severity(head_pt: Tuple[float, float], center_x: int, center_y: int, radius: int) -> Tuple[int, str, float]:
    dx = head_pt[0] - float(center_x)
    dy = head_pt[1] - float(center_y)
    distance = math.hypot(dx, dy)
    effective_radius = max(float(radius), 1.0)
    normalized = distance / effective_radius
    straight_threshold = 1.15
    if normalized <= straight_threshold:
        return 1, "spine is straight", normalized
    return 2, "spine is not straight", normalized


def main() -> None:
    args = parse_args()
    pose_path = args.pose.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)

    video_path = args.video.expanduser() if args.video else infer_video_path(pose_path)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Video not found. Provide --video explicitly.")

    meta, frames, *_ = load_pose_json(pose_path)
    release_frame = get_release_frame(pose_path)
    if release_frame is None:
        raise RuntimeError("Could not determine release frame from pose JSON.")
    target_frame = max(1, release_frame - 1)

    hips, head_pt, nose_pt, head_scale = extract_keypoints(meta, frames, target_frame)
    if hips is None or head_pt is None or nose_pt is None:
        raise RuntimeError("Frame lacks required foot/head keypoints (need feet, head, and nose).")
    (left_foot, right_foot) = hips

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = args.output.expanduser() if args.output else OUTPUT_DIR / f"{pose_path.stem}_spine.png"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read the requested frame from the video.")

    line_left = int(round(left_foot[0]))
    line_right = int(round(right_foot[0]))
    if line_left > line_right:
        line_left, line_right = line_right, line_left
    height = frame.shape[0]
    cv2.line(frame, (line_left, 0), (line_left, height), LINE_COLOR, 2)
    cv2.line(frame, (line_right, 0), (line_right, height), LINE_COLOR, 2)

    circle_x = (line_left + line_right) // 2
    anchor_pt = nose_pt
    circle_y = int(round(anchor_pt[1]))
    dynamic_radius = args.head_radius
    if head_scale and head_scale > 0:
        dynamic_radius = max(1, int(round(head_scale * 3.9)))
    cv2.circle(frame, (circle_x, circle_y), dynamic_radius, HEAD_COLOR, 3)

    cv2.putText(
        frame,
        f"frame {target_frame}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"frame {target_frame}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), frame)
    debug_path = save_head_debug(frame, circle_x, circle_y, dynamic_radius, pose_path.stem)
    print(f"✅ Saved spine visualization → {out_path}")
    if debug_path:
        print(f"🧪 Head zoom snapshot → {debug_path}")
    severity_score, severity_desc, normalized_distance = classify_spine_severity(anchor_pt, circle_x, circle_y, dynamic_radius)
    print(
        f"Spine severity ({pose_path.stem}): {severity_score} → {severity_desc} "
        f"(normalized distance {normalized_distance:.2f})"
    )
    prompt_text = load_spine_prompt().rstrip()
    if prompt_text.endswith(":"):
        prompt_headline = f"{prompt_text} {severity_desc}"
    else:
        prompt_headline = f"{prompt_text}\n{severity_desc}"
    prompt_payload = (
        f"{prompt_headline}\n\n"
        f"Video ID: {pose_path.stem}\n"
        f"Analyzed frame: {target_frame}"
    )
    notes_text = load_spine_notes()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ANALYSIS_DIR / f"{pose_path.stem}.md"
    rel_image = Path("..") / OUTPUT_DIR.name / out_path.name

    if severity_score <= 1:
        content = (
            f"Result is good: {severity_desc} (normalized distance {normalized_distance:.2f}).\n\n"
            f"![Spine visualization]({rel_image.as_posix()})\n"
        )
        report_path.write_text(content, encoding="utf-8")
        print(f"📝 Spine analysis saved to {report_path} (no assistant call; posture acceptable)")
    else:
        prompt_payload += "\n\nIssue flagged above. Provide concise corrective guidance (<120 words)."
        note_after_call: Optional[str] = f"{pose_path.stem} - spine is not straight"
        response = call_rag_assistant(
            prompt_payload,
            image_path=debug_path,
            system_prompt=SPINE_SYSTEM_PROMPT,
            notes_text=notes_text,
        )
        if response:
            content = response.strip()
            content += f"\n\n![Spine visualization]({rel_image.as_posix()})\n"
            report_path.write_text(content, encoding="utf-8")
            print(f"📝 Spine analysis saved to {report_path}")
        else:
            print("⚠️  Spine analysis assistant response unavailable.")
        append_spine_note(note_after_call)


if __name__ == "__main__":
    main()
