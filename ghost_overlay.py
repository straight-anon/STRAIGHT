#!/usr/bin/env python3
"""Overlay skeletons from two shots, aligning the reference as a ghost."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("OpenCV (cv2) is required. Install it via `pip install opencv-python`.") from exc

from phase_estimation import PhaseEstimationConfig, estimate_draw_start
from rag_gpt_client import call_rag_assistant
from visualize_skeleton import SMOOTHED_DIR, draw_instances, infer_video_path, load_pose_json

OUTPUT_DIR = Path("ghost_visualizations")
ANALYSIS_DIR = Path("ghost_overlay_analysis")
WEB_ASSETS_RUNS_DIR = Path("web/public/assets/runs")
NOTES_PATH = Path("config/notes/ghost_overlay_notes.txt")
PROMPT_GHOST = Path("config/prompts/ghost_timing.md")
GHOST_SYSTEM_PROMPT = "You are an archery timing coach."
EXCLUDED_RUNS_PATH = Path("config/excluded_runs.json")
MIN_DATASET_SAMPLES = 5

GHOST_COLOR = (0, 0, 255)
ANCHOR_KEYPOINTS = ["left_hip", "right_hip"]


PoseDict = Dict[str, Tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay a reference skeleton as a ghost.")
    parser.add_argument("--actual-pose", required=True, type=Path, help="Path to the smoothed pose JSON.")
    parser.add_argument(
        "--reference-pose",
        required=True,
        type=Path,
        help="Path to the reference smoothed pose JSON.",
    )
    parser.add_argument("--actual-video", type=Path, help="Optional explicit video path for the actual clip.")
    parser.add_argument("--reference-video", type=Path, help="Optional explicit video path for the reference clip.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output video path (defaults to ghost_visualizations/<actual>_ghost.mp4).",
    )
    return parser.parse_args()


def build_pose_sequence(
    frames: Dict[int, dict], keypoint_names: List[str], offset: Tuple[int, int] | None = None
) -> Tuple[List[PoseDict], List[int]]:
    order = sorted(frames.keys())
    sequence: List[PoseDict] = []
    dx, dy = offset or (0, 0)
    for frame_id in order:
        entry = frames[frame_id]
        instances = entry.get("instances", [])
        if not instances:
            sequence.append({})
            continue
        keypoints = instances[0].get("smoothed_keypoints") or instances[0].get("keypoints") or []
        pose: PoseDict = {}
        for idx, name in enumerate(keypoint_names):
            if idx < len(keypoints) and len(keypoints[idx]) >= 2:
                pose[name] = (float(keypoints[idx][0]) - dx, float(keypoints[idx][1]) - dy)
        sequence.append(pose)
    return sequence, order


def mean_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return sx / len(points), sy / len(points)


def compute_transform(
    ghost_points: List[Tuple[float, float]],
    actual_pose: PoseDict,
    keypoint_names: List[str],
    anchor_names: List[str],
):
    actual_pts = [actual_pose.get(name) for name in anchor_names if actual_pose.get(name) is not None]
    actual_pts = [p for p in actual_pts if p is not None]
    if len(actual_pts) < 2:
        return None
    ghost_pts = []
    for name in anchor_names:
        idx = keypoint_names.index(name) if name in keypoint_names else None
        if idx is not None and idx < len(ghost_points):
            ghost_pts.append(ghost_points[idx])
    if len(ghost_pts) < 2:
        return None
    actual_center = mean_point(actual_pts)
    ghost_center = mean_point(ghost_pts)
    actual_vec = (actual_pts[1][0] - actual_pts[0][0], actual_pts[1][1] - actual_pts[0][1])
    ghost_vec = (ghost_pts[1][0] - ghost_pts[0][0], ghost_pts[1][1] - ghost_pts[0][1])
    ghost_len = math.hypot(*ghost_vec)
    actual_len = math.hypot(*actual_vec)
    if ghost_len < 1e-6 or actual_len < 1e-6:
        return None
    scale = actual_len / ghost_len
    angle_ghost = math.atan2(ghost_vec[1], ghost_vec[0])
    angle_actual = math.atan2(actual_vec[1], actual_vec[0])
    angle_delta = angle_actual - angle_ghost
    cos_a = math.cos(angle_delta)
    sin_a = math.sin(angle_delta)
    return ghost_center, actual_center, scale, cos_a, sin_a


def apply_transform(points: List[Tuple[float, float]], transform):
    if transform is None:
        return None
    ghost_center, actual_center, scale, cos_a, sin_a = transform
    aligned: List[Tuple[int, int]] = []
    for x, y in points:
        dx = x - ghost_center[0]
        dy = y - ghost_center[1]
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        rx = rx * scale + actual_center[0]
        ry = ry * scale + actual_center[1]
        aligned.append((int(round(rx)), int(round(ry))))
    return aligned


def draw_ghost(image, ghost_points, transform, skeleton_links, point_radius: int):
    aligned = apply_transform(ghost_points, transform)
    if not aligned:
        return
    for x, y in aligned:
        cv2.circle(image, (x, y), max(1, point_radius - 1), GHOST_COLOR, -1)
    for start, end in skeleton_links:
        if start >= len(aligned) or end >= len(aligned):
            continue
        x1, y1 = aligned[start]
        x2, y2 = aligned[end]
        cv2.line(image, (x1, y1), (x2, y2), GHOST_COLOR, 2, cv2.LINE_AA)


def load_pose_frames(path: Path):
    meta, frames, keypoint_colors, link_colors, skeleton_links = load_pose_json(path)
    if not frames:
        raise ValueError(f"No frames available in {path} for visualization.")
    return meta, frames, keypoint_colors, link_colors, skeleton_links


def normalize_run_id(path: Path | str) -> str:
    """Normalize pose filenames so cropped/uncropped variants count as one sample."""
    name = path.name if isinstance(path, Path) else str(path)
    base = name.strip()
    if base.endswith(".json"):
        base = base[: -len(".json")]
    if base.startswith("results_"):
        base = base.split("results_", 1)[1]
    if base.endswith("_cropped"):
        base = base[: -len("_cropped")]
    parts = base.split("-")
    fixed_parts = []
    for part in parts:
        if part.count(".") == 2 and len(part.split(".")[-1]) < 3:
            left, last = part.rsplit(".", 1)
            fixed_parts.append(f"{left}.{last.zfill(3)}")
        elif part.count(".") == 3 and len(part.split(".")[-1]) < 3:
            segs = part.split(".")
            segs[-1] = segs[-1].zfill(3)
            fixed_parts.append(".".join(segs))
        else:
            fixed_parts.append(part)
    return "-".join(fixed_parts)


def get_draw_release_frames(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    info = data.get("smoothing_info") or {}
    draw = info.get("draw_frame") if isinstance(info.get("draw_frame"), int) else None
    release = info.get("release_frame") if isinstance(info.get("release_frame"), int) else None
    if release is None:
        window_end = info.get("window_end_frame")
        if isinstance(window_end, int):
            release = window_end
    if release is None and draw is not None:
        frames = data.get("instance_info") or []
        end = draw + 240
        if frames:
            last = frames[-1].get("frame_id")
            if isinstance(last, int):
                end = min(end, last)
        release = end
    return draw, release


def release_stats(exclude_path: Path) -> Tuple[Optional[float], Optional[float], int]:
    releases: List[int] = []
    seen: set[str] = set()
    excluded = load_excluded_run_ids()
    print(f"ℹ️  Excluded runs for timing: {sorted(excluded) or 'none'}")
    exclude_key = normalize_run_id(exclude_path)
    for json_path in sorted(SMOOTHED_DIR.glob("*.json")):
        if not json_path.stem.endswith("_cropped"):
            continue
        key = normalize_run_id(json_path)
        if key == exclude_key:
            print(f"  ↪︎ skipping {json_path.name} (current run)")
            continue
        if key in seen:
            print(f"  ↪︎ skipping {json_path.name} (duplicate)")
            continue
        if key in excluded:
            print(f"  ↪︎ skipping {json_path.name} (excluded)")
            continue
        seen.add(key)
        _, release = get_draw_release_frames(json_path)
        if isinstance(release, int):
            releases.append(release)
            print(f"  • {json_path.name}: release_frame={release}")
    if not releases:
        return None, None, 0
    mean_val = statistics.mean(releases)
    std_val = statistics.pstdev(releases) if len(releases) > 1 else 0.0
    return mean_val, std_val, len(releases)


def draw_release_window_stats(exclude_path: Path) -> Tuple[Optional[float], Optional[float], int]:
    windows: List[int] = []
    seen: set[str] = set()
    excluded = load_excluded_run_ids()
    print(f"ℹ️  Excluded runs for window stats: {sorted(excluded) or 'none'}")
    exclude_key = normalize_run_id(exclude_path)
    for json_path in sorted(SMOOTHED_DIR.glob("*.json")):
        if not json_path.stem.endswith("_cropped"):
            continue
        key = normalize_run_id(json_path)
        if key == exclude_key:
            print(f"  ↪︎ skipping {json_path.name} (current run)")
            continue
        if key in seen:
            print(f"  ↪︎ skipping {json_path.name} (duplicate)")
            continue
        if key in excluded:
            print(f"  ↪︎ skipping {json_path.name} (excluded)")
            continue
        seen.add(key)
        draw, release = get_draw_release_frames(json_path)
        if isinstance(draw, int) and isinstance(release, int):
            if release >= draw:
                windows.append(release - draw)
                seconds = (release - draw) / 60.0
                print(f"  • {json_path.name}: draw={draw}, release={release}, window_frames={release - draw}, window_seconds={seconds:.2f}")
    if not windows:
        return None, None, 0
    mean_val = statistics.mean(windows)
    std_val = statistics.pstdev(windows) if len(windows) > 1 else 0.0
    return mean_val, std_val, len(windows)


def load_ghost_prompt() -> str:
    if not PROMPT_GHOST.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_GHOST}")
    text = PROMPT_GHOST.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file {PROMPT_GHOST} is empty.")
    return text


def load_excluded_run_ids() -> set[str]:
    if not EXCLUDED_RUNS_PATH.exists():
        return set()
    try:
        data = json.loads(EXCLUDED_RUNS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {normalize_run_id(str(item)) for item in data if str(item).strip()}
    except Exception:
        pass
    return set()


def load_ghost_notes() -> str:
    if not NOTES_PATH.exists():
        return ""
    return NOTES_PATH.read_text(encoding="utf-8").strip()


def append_ghost_note(note: str) -> None:
    clean = note.strip()
    if not clean:
        return
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(clean + "\n")


def classify_release_timing(
    window_frames: Optional[int],
    avg_window: Optional[float],
    std_window: Optional[float],
) -> Tuple[str, str, Optional[float]]:
    if (
        window_frames is None
        or avg_window is None
        or std_window is None
        or std_window <= 0
    ):
        return "unknown", "Release timing comparison unavailable.", None
    diff = float(window_frames) - float(avg_window)
    if abs(diff) <= std_window:
        return "within", "Release timing is within one standard deviation of the reference mean.", diff
    if diff > 0:
        return "too_long", "Release timing is slower than the reference window.", diff
    return "too_short", "Release timing is faster than the reference window.", diff


def write_ghost_report(content: str, video_id: str, primary_path: Path, asset_video_id: Optional[str] = None) -> None:
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_text(content, encoding="utf-8")
    asset_dir = WEB_ASSETS_RUNS_DIR / (asset_video_id or video_id)
    try:
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset_path = asset_dir / "ghost_overlay.md"
        asset_path.write_text(content, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - asset publishing is best-effort
        print(f"⚠️  Unable to save ghost report to web assets: {exc}")


def format_frame_and_time(frame_val: Optional[float], fps: float) -> str:
    if frame_val is None:
        return "N/A"
    seconds = frame_val / max(fps, 1e-6)
    return f"{frame_val:.1f} frames (~{seconds:.2f}s)"


def format_int_frame(frame_val: Optional[int], fps: float) -> str:
    if frame_val is None:
        return "N/A"
    seconds = frame_val / max(fps, 1e-6)
    return f"{frame_val} frames (~{seconds:.2f}s)"


def run_timing_analysis(
    video_id: str,
    video_path: Path,
    release_frame: Optional[int],
    draw_to_release_frames: Optional[int],
    fps: float,
    avg_release_frame: Optional[float],
    std_release_frame: Optional[float],
    dataset_count: int,
    avg_window_frames: Optional[float],
    std_window_frames: Optional[float],
    window_count: int,
    asset_video_id: Optional[str] = None,
) -> None:
    prompt_text = load_ghost_prompt()
    min_samples = MIN_DATASET_SAMPLES
    sample_count = max(window_count or 0, dataset_count or 0)
    has_window_stats = avg_window_frames is not None and std_window_frames is not None and std_window_frames > 0
    limited_dataset = sample_count < min_samples
    status = "unknown"
    status_desc = "Not enough dataset stats available for timing."
    diff_frames = None
    note_after_call: Optional[str] = None
    notes_text = load_ghost_notes()

    if has_window_stats and not limited_dataset:
        status, status_desc, diff_frames = classify_release_timing(
            draw_to_release_frames, avg_window_frames, std_window_frames
        )
        if status in {"too_long", "too_short"}:
            direction = "too long" if status == "too_long" else "too short"
            note_after_call = f"{video_id} - release timing {direction}"

    draw_seconds = (
        draw_to_release_frames / max(fps, 1e-6) if draw_to_release_frames is not None else None
    )

    if limited_dataset:
        summary_line = (
            f"Timing measured at {draw_seconds:.2f}s; dataset too small for judgment (n={sample_count}, need ≥{min_samples})."
            if draw_seconds is not None
            else f"Dataset too small for judgment (n={sample_count}, need ≥{min_samples})."
        )
        status = "unknown"
    elif has_window_stats and status in {"too_long", "too_short"} and diff_frames is not None:
        diff_seconds = abs(diff_frames) / max(fps, 1e-6)
        direction = "late" if diff_frames > 0 else "early"
        summary_line = f"Release timing deviated by {diff_seconds:.2f}s ({direction}) relative to the dataset average."
    elif has_window_stats:
        summary_line = "Release timing is acceptable and within the reference window."
    else:
        summary_line = status_desc

    if sample_count < min_samples:
        summary_line += f" Dataset size is small (n={sample_count}, need ≥{min_samples})."

    lines = [
        prompt_text,
        "",
        f"Video ID: {video_id}",
        f"Timing summary: {summary_line}",
        f"Status: {status_desc}",
    ]

    question = "\n".join(lines).strip()

    def fmt(num: Optional[float], decimals: int = 2) -> str:
        if num is None:
            return "NA"
        return f"{num:.{decimals}f}"

    avg_window_seconds = avg_window_frames / max(fps, 1e-6) if avg_window_frames is not None else None
    std_window_seconds = std_window_frames / max(fps, 1e-6) if std_window_frames is not None else None

    data_lines = [
        "DATA_START",
        f"draw_to_release_frames={fmt(float(draw_to_release_frames)) if draw_to_release_frames is not None else 'NA'}",
        f"draw_to_release_seconds={fmt(draw_seconds) if draw_seconds is not None else 'NA'}",
        f"dataset_draw_to_release_avg_frames={fmt(avg_window_frames) if avg_window_frames is not None else 'NA'}",
        f"dataset_draw_to_release_avg_seconds={fmt(avg_window_seconds) if avg_window_seconds is not None else 'NA'}",
        f"dataset_draw_to_release_std_frames={fmt(std_window_frames) if std_window_frames is not None else 'NA'}",
        f"dataset_draw_to_release_std_seconds={fmt(std_window_seconds) if std_window_seconds is not None else 'NA'}",
        f"dataset_draw_to_release_sample_count={window_count}",
        f"dataset_release_sample_count={dataset_count}",
        f"dataset_min_samples_required={min_samples}",
        f"timing_status={status}",
        "DATA_END",
    ]
    data_block = "\n".join(data_lines) + "\n\n"

    is_bad = status in {"too_long", "too_short"}
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ANALYSIS_DIR / f"{video_id}.md"
    asset_id = asset_video_id or video_id

    if is_bad:
        question = (
            question
            + "\n\nIssue flagged above. Provide concise guidance to correct the timing (actionable, <120 words)."
        )
        response = call_rag_assistant(
            question,
            system_prompt=GHOST_SYSTEM_PROMPT,
            notes_text=notes_text,
        )
        if not response:
            print("⚠️  Ghost overlay analysis unavailable.")
        else:
            content = data_block + response.strip()
            write_ghost_report(content, video_id, report_path, asset_video_id=asset_id)
            print(f"📝 Ghost timing analysis saved to {report_path} (and web assets)")
    elif limited_dataset:
        content = f"{data_block}Timing note: {summary_line}\n"
        write_ghost_report(content, video_id, report_path, asset_video_id=asset_id)
        print(f"ℹ️  Ghost timing analysis saved to {report_path} (limited dataset)")
    else:
        content = f"{data_block}Result is good: {summary_line}\n"
        write_ghost_report(content, video_id, report_path, asset_video_id=asset_id)
        print(f"📝 Ghost timing analysis saved to {report_path} (and web assets)")
    if note_after_call:
        append_ghost_note(note_after_call)


def visualize_ghost(
    actual_pose_path: Path,
    reference_pose_path: Path,
    actual_video_path: Optional[Path] = None,
    reference_video_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    crop_offset: Optional[Tuple[int, int]] = None,
    asset_video_id: Optional[str] = None,
    report_video_id: Optional[str] = None,
):
    actual_pose_path = actual_pose_path.expanduser().resolve()
    reference_pose_path = reference_pose_path.expanduser().resolve()

    meta_actual, frames_actual, colors_actual, links_actual, skeleton_links = load_pose_frames(actual_pose_path)
    meta_ref, frames_ref, _, _, _ = load_pose_frames(reference_pose_path)

    keypoint_names = [meta_actual.get("keypoint_id2name", {}).get(str(i), str(i)) for i in range(len(colors_actual))]
    actual_sequence, actual_order = build_pose_sequence(frames_actual, keypoint_names, crop_offset)
    ref_sequence, ref_order = build_pose_sequence(frames_ref, keypoint_names)
    config = PhaseEstimationConfig()
    fps = 60.0
    draw_actual = estimate_draw_start(actual_sequence, fps, config) or 0
    draw_ref = estimate_draw_start(ref_sequence, fps, config) or 0
    stored_draw_actual, stored_release_actual = get_draw_release_frames(actual_pose_path)
    stored_draw_ref, stored_release_ref = get_draw_release_frames(reference_pose_path)
    draw_for_print_actual = stored_draw_actual if stored_draw_actual is not None else draw_actual
    draw_for_print_ref = stored_draw_ref if stored_draw_ref is not None else draw_ref
    release_actual = stored_release_actual
    release_ref = stored_release_ref
    if release_actual is None and actual_order:
        release_actual = actual_order[-1]
    if release_ref is None and ref_order:
        release_ref = ref_order[-1]
    draw_to_release_actual = (
        release_actual - draw_for_print_actual if release_actual is not None and draw_for_print_actual is not None else None
    )
    draw_to_release_ref = (
        release_ref - draw_for_print_ref if release_ref is not None and draw_for_print_ref is not None else None
    )
    avg_release, std_release, count_release = release_stats(actual_pose_path)
    avg_window, std_window, count_windows = draw_release_window_stats(actual_pose_path)

    video_candidate = actual_video_path or infer_video_path(actual_pose_path)
    if video_candidate is None:
        raise FileNotFoundError("Could not infer actual video path; provide --actual-video.")
    video_path = video_candidate.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    ref_candidate = reference_video_path or infer_video_path(reference_pose_path)
    if ref_candidate is None:
        raise FileNotFoundError("Could not infer reference video path; provide --reference-video.")
    ref_video_path = ref_candidate.expanduser().resolve()
    if not ref_video_path.exists():
        raise FileNotFoundError(ref_video_path)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = output_path.expanduser().resolve() if output_path else OUTPUT_DIR / f"{actual_pose_path.stem}_ghost.mp4"

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    ref_video = cv2.VideoCapture(str(ref_video_path))
    if not ref_video.isOpened():
        video.release()
        raise RuntimeError(f"Failed to open reference video {ref_video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("⚠️  AVC1 codec unavailable; falling back to MP4V.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        video.release()
        ref_video.release()
        raise RuntimeError(f"Failed to create {out_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"🎥 {video_path.name}: {fps:.2f} fps | {width}x{height} | {total_frames} frames")

    pose_index_map = {fid: idx for idx, fid in enumerate(actual_order)}
    start_id = actual_order[0]
    end_id = actual_order[-1]
    video.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_id - 1))
    current_frame_id = start_id

    ref_sequence_points = []
    for frame in ref_order:
        ref_entry = frames_ref.get(frame, {})
        instances = ref_entry.get("instances", [])
        if not instances:
            ref_sequence_points.append([])
            continue
        points = instances[0].get("smoothed_keypoints") or instances[0].get("keypoints") or []
        ref_sequence_points.append([(float(x), float(y)) for x, y in points])

    ref_transform = None
    while current_frame_id <= end_id:
        ret, frame = video.read()
        if not ret:
            print("⚠️  Ran out of video frames before finishing window.")
            break
        pose_frame = frames_actual.get(current_frame_id)
        if pose_frame:
            draw_instances(
                frame,
                pose_frame.get("instances", []),
                colors_actual,
                links_actual,
                skeleton_links,
                score_threshold=0.2,
                point_radius=4,
            )
        seq_idx = pose_index_map.get(current_frame_id)
        if seq_idx is not None and ref_sequence_points:
            relative = seq_idx - draw_actual
            ref_idx = draw_ref + relative
            ref_idx = max(0, min(ref_idx, len(ref_sequence_points) - 1))
            if ref_transform is None:
                ref_transform = compute_transform(
                    ref_sequence_points[ref_idx],
                    actual_sequence[seq_idx],
                    keypoint_names,
                    ANCHOR_KEYPOINTS,
                )
            draw_ghost(
                frame,
                ref_sequence_points[ref_idx],
                ref_transform,
                skeleton_links,
                point_radius=4,
            )
        writer.write(frame)
        current_frame_id += 1

    video.release()
    ref_video.release()
    writer.release()
    print(f"✅ Saved ghost overlay video → {out_path}")
    if draw_to_release_actual is not None:
        seconds_actual = draw_to_release_actual / fps
        print(f"🎯 Actual draw→release: {draw_to_release_actual} frames ({seconds_actual:.2f}s)")
    else:
        print("🎯 Actual draw→release: N/A")
    if draw_to_release_ref is not None:
        seconds_ref = draw_to_release_ref / fps
        print(f"👻 Reference draw→release: {draw_to_release_ref} frames ({seconds_ref:.2f}s)")
    else:
        print("👻 Reference draw→release: N/A")

    if release_actual is not None:
        print(f"🎯 Actual release frame: {release_actual} ({release_actual / fps:.2f}s)")
    else:
        print("🎯 Actual release frame: N/A")

    if avg_release is not None:
        std_text = f" ± {std_release:.2f}" if std_release is not None else ""
        avg_seconds = avg_release / fps
        std_seconds = std_release / fps if std_release is not None else None
        seconds_text = f" ({avg_seconds:.2f}s"
        if std_seconds is not None:
            seconds_text += f" ± {std_seconds:.2f}s"
        seconds_text += ")"
        print(
            f"📊 Avg release frame (others, n={count_release}): {avg_release:.2f}{std_text}{seconds_text}"
        )
    else:
        print("📊 Avg release frame (others): N/A")
    run_timing_analysis(
        video_id=report_video_id or actual_pose_path.stem,
        video_path=out_path,
        release_frame=release_actual,
        draw_to_release_frames=draw_to_release_actual,
        fps=fps,
        avg_release_frame=avg_release,
        std_release_frame=std_release,
        dataset_count=count_release,
        avg_window_frames=avg_window,
        std_window_frames=std_window,
        window_count=count_windows,
        asset_video_id=asset_video_id or report_video_id or actual_pose_path.stem,
    )


def main() -> None:
    args = parse_args()
    visualize_ghost(
        args.actual_pose.expanduser().resolve(),
        args.reference_pose.expanduser().resolve(),
        actual_video_path=args.actual_video.expanduser().resolve() if args.actual_video else None,
        reference_video_path=args.reference_video.expanduser().resolve() if args.reference_video else None,
        output_path=args.output.expanduser().resolve() if args.output else None,
        asset_video_id=None,
        report_video_id=None,
    )


if __name__ == "__main__":
    main()
