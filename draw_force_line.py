#!/usr/bin/env python3
"""Visualize the draw-force line (DFL) on the pre-release frame."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("OpenCV (cv2) is required. Install it via `pip install opencv-python`.") from exc

from rag_gpt_client import call_rag_assistant

VIDEO_DIR = Path("training_videos")
OUTPUT_DIR = Path("draw_force_visualizations")
DEFAULT_DATA_DIR = Path("smoothed_inference_data")
LABEL_DIR = Path("estimated_labels")
ANGLE_ANALYSIS_DIR = Path("draw_force_angle_analysis")
LENGTH_ANALYSIS_DIR = Path("draw_length_analysis")
NOTES_PATH = Path("config/notes/draw_force_notes.txt")
PROMPT_DFL_ANGLE = Path("config/prompts/draw_force_angle.md")
PROMPT_DFL_LENGTH = Path("config/prompts/draw_force_length.md")
DFL_SYSTEM_PROMPT = "You are an archery technique coach."
DFL_COLOR = (0, 255, 0)
DRAW_ARM_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2
MIN_DATASET_SAMPLES = 5
EXCLUDED_RUNS_PATH = Path("config/excluded_runs.json")

Point = Tuple[float, float]


@dataclass
class PoseFrame:
    """Container for the frame that precedes release."""

    pose_path: Path
    frame_id: int
    keypoints: Dict[str, Point]
    all_points: List[Point]


@dataclass
class DFLGeometry:
    """Computed geometry for the DFL visualization."""

    bow_wrist: Point
    draw_wrist: Point
    draw_elbow: Point
    adjusted_bow: Point
    dfl_end: Point
    draw_line_end: Point
    draw_line_start: Point
    angle_deg: float
    draw_length_px: float
    normalized_draw_length: Optional[float]
    hip_center: Optional[Point]
    hip_width: Optional[float]
    left_wrist_raw: Point
    right_wrist_raw: Point

    def line_points(self) -> List[Point]:
        """Return all line endpoints needed for cropping."""
        return [
            self.adjusted_bow,
            self.dfl_end,
            self.draw_wrist,
            self.draw_line_end,
            self.draw_line_start,
            self.draw_elbow,
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the DFL and draw-arm lines on the pre-release frame.")
    parser.add_argument("--pose", required=True, type=Path, help="Path to the smoothed pose JSON.")
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional explicit video path (defaults to training_videos/<pose_stem>.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to draw_force_visualizations/<pose_stem>_force.png).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to scan for comparison pose JSONs when computing dataset stats.",
    )
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file {path} is empty.")
    return text


def load_dfl_notes() -> str:
    if not NOTES_PATH.exists():
        return ""
    return NOTES_PATH.read_text(encoding="utf-8").strip()


def normalize_run_id(raw: str) -> str:
    base = raw.strip()
    if base.endswith(".json"):
        base = base[: -len(".json")]
    if base.startswith("results_"):
        base = base.split("results_", 1)[1]
    if base.endswith("_cropped"):
        base = base[: -len("_cropped")]
    parts = base.split("-")
    fixed_parts: List[str] = []
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


def load_excluded_run_ids() -> Set[str]:
    if not EXCLUDED_RUNS_PATH.exists():
        return set()
    try:
        data = json.loads(EXCLUDED_RUNS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {normalize_run_id(str(item)) for item in data if str(item).strip()}
    except Exception:
        pass
    return set()


def append_dfl_note(note: str) -> None:
    clean = note.strip()
    if not clean:
        return
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(clean + "\n")


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
    if not stats:
        return "unknown", f"No dataset stats available for {label}."
    avg, std = stats
    if std is None or std <= 0:
        return "unknown", f"No variability data for {label}."
    window = std + grace
    diff = value - avg
    if abs(diff) <= window:
        return "within", f"{label.capitalize()} stays within the reference window."
    direction = "higher" if diff > 0 else "lower"
    return (
        "outside",
        f"{label.capitalize()} is {abs(diff):.2f}{unit} {direction} than the reference mean."
    )


def infer_video_path(pose_path: Path) -> Optional[Path]:
    """Infer the accompanying video by matching the pose stem."""
    stem = pose_path.stem
    if stem.startswith("results_"):
        stem = stem.split("results_", 1)[1]
    candidate = VIDEO_DIR / f"{stem}.mp4"
    if candidate.exists():
        return candidate
    return None


def infer_dataset_name(pose_path: Path) -> Optional[str]:
    """Return the dataset identifier used for video/label lookups."""
    stem = pose_path.stem
    if stem.startswith("results_"):
        stem = stem.split("results_", 1)[1]
    if stem.endswith("_cropped"):
        stem = stem[: -len("_cropped")]
    parts = stem.split("-")
    fixed_parts = []
    for part in parts:
        if part.count(".") == 2 and len(part.split(".")[-1]) < 3:
            base, last = part.rsplit(".", 1)
            fixed_parts.append(f"{base}.{last.zfill(3)}")
        else:
            fixed_parts.append(part)
    stem = "-".join(fixed_parts)
    return stem or None


def resolve_label_path(dataset: str) -> Optional[Path]:
    """Locate the estimated-label JSON that exactly matches the dataset name."""
    path = LABEL_DIR / f"vlm_estimated_label_{dataset}.json"
    return path if path.exists() else None


def release_frame_from_labels(pose_path: Path) -> int:
    """Read the release-frame boundary from the estimated label JSON."""
    dataset = infer_dataset_name(pose_path)
    if not dataset:
        raise RuntimeError(f"Could not infer dataset name from pose file {pose_path}")
    label_path = resolve_label_path(dataset)
    if not label_path:
        raise FileNotFoundError(
            f"No estimated label JSON found for dataset '{dataset}'. "
            "Please run the label generation pipeline first."
        )
    try:
        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        raise RuntimeError(f"Failed to parse estimated label JSON: {label_path}") from None
    fps = float(data.get("fps", 60.0))
    release_idx: Optional[int] = None
    for phase in data.get("phases", []):
        name = str(phase.get("name", "")).lower()
        if "release" not in name:
            continue
        start_frame = phase.get("start_frame")
        if isinstance(start_frame, int):
            release_idx = start_frame
        elif "start" in phase:
            try:
                release_idx = int(round(float(phase["start"]) * fps))
            except (TypeError, ValueError):
                pass
        if release_idx is not None:
            break
    if release_idx is None:
        raise RuntimeError(f"No release phase found in estimated label JSON: {label_path}")
    return max(1, release_idx)


def load_pose_frame(pose_path: Path) -> PoseFrame:
    """Load the smoothed frame that occurs immediately before the release."""
    with pose_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta_info", {})
    frames = {}
    for frame in data.get("instance_info", []):
        frame_id = int(frame.get("frame_id", 1))
        frames[frame_id] = frame
    if not frames:
        raise RuntimeError(f"No frames found in {pose_path}.")
    smoothing = data.get("smoothing_info") or {}
    target_id = determine_pre_release_frame(frames, smoothing, pose_path)
    frame_entry = select_frame(frames, target_id)
    keypoints, all_points = extract_keypoints(meta, frame_entry)
    return PoseFrame(pose_path=pose_path, frame_id=target_id, keypoints=keypoints, all_points=all_points)


def determine_pre_release_frame(frames: Dict[int, dict], smoothing: dict, pose_path: Path) -> int:
    """Return the frame index that immediately precedes the release phase."""
    label_release = release_frame_from_labels(pose_path)
    return max(1, label_release - 1)


def select_frame(frames: Dict[int, dict], frame_id: int) -> dict:
    """Select the requested frame or the closest preceding frame."""
    if frame_id in frames:
        return frames[frame_id]
    sorted_ids = sorted(frames.keys())
    for idx in reversed(sorted_ids):
        if idx <= frame_id:
            return frames[idx]
    return frames[sorted_ids[0]]


def extract_keypoints(meta: dict, frame_entry: dict) -> Tuple[Dict[str, Point], List[Point]]:
    """Extract smoothed keypoints and keep the raw array for cropping."""
    instances = frame_entry.get("instances", [])
    if not instances:
        raise RuntimeError("Frame has no instances to visualize.")
    source = instances[0]
    raw_points = source.get("smoothed_keypoints") or source.get("keypoints") or []
    all_points: List[Point] = []
    for coords in raw_points:
        if isinstance(coords, Sequence) and len(coords) >= 2:
            x, y = float(coords[0]), float(coords[1])
            if math.isfinite(x) and math.isfinite(y):
                all_points.append((x, y))
    name2id = meta.get("keypoint_name2id") or {}
    named: Dict[str, Point] = {}
    for name, idx in name2id.items():
        try:
            kp_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= kp_idx < len(raw_points):
            coords = raw_points[kp_idx]
            if isinstance(coords, Sequence) and len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if math.isfinite(x) and math.isfinite(y):
                    named[name] = (x, y)
    return named, all_points


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def adjust_bow_wrist(
    bow_wrist: Point,
    draw_wrist: Point,
    nose: Point,
    eye: Point,
    forward_multiplier: float = 3.0,
    lift_multiplier: float = 2.0,
) -> Point:
    """
    Extend the bow wrist away from the archer so the DFL originates ahead of the bow hand.

    The forward offset and vertical lift both scale with the nose→eye distance to keep the
    adjustment proportional to the archer's size.
    """
    eye_gap = distance(nose, eye)
    if eye_gap <= 0:
        return bow_wrist
    direction = (bow_wrist[0] - draw_wrist[0], bow_wrist[1] - draw_wrist[1])
    length = math.hypot(direction[0], direction[1])
    if length < 1e-6:
        return bow_wrist
    scale = (eye_gap * forward_multiplier) / length
    extended = (bow_wrist[0] + direction[0] * scale, bow_wrist[1] + direction[1] * scale)
    lifted = (extended[0], extended[1] - eye_gap * lift_multiplier)
    return lifted


def extend_line(start: Point, through: Point, extra_distance: float) -> Point:
    """
    Extend the line segment defined by ``start`` and ``through`` by ``extra_distance`` beyond ``through``.
    """
    direction = (through[0] - start[0], through[1] - start[1])
    length = math.hypot(direction[0], direction[1])
    if length < 1e-6:
        return through
    unit = (direction[0] / length, direction[1] / length)
    return (through[0] + unit[0] * extra_distance, through[1] + unit[1] * extra_distance)


def compute_internal_angle_deg(vec_a: Point, vec_b: Point) -> float:
    """
    Compute the internal angle in degrees between two vectors sharing the same origin.
    """
    dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
    len_a = math.hypot(vec_a[0], vec_a[1])
    len_b = math.hypot(vec_b[0], vec_b[1])
    if len_a < 1e-6 or len_b < 1e-6:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (len_a * len_b)))
    return math.degrees(math.acos(cos_theta))


def compute_crop_box(
    frame_shape: Tuple[int, int, int],
    coords: Iterable[Point],
    padding_ratio: float = 0.2,
    min_size: int = 320,
) -> Tuple[int, int, int, int]:
    """
    Build a crop that keeps the archer and the DFL lines centered within the saved frame.
    """
    height, width = frame_shape[:2]
    xs: List[float] = []
    ys: List[float] = []
    for x, y in coords:
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return (0, 0, width, height)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    pad_x = max(min_size * 0.1, span_x * padding_ratio)
    pad_y = max(min_size * 0.1, span_y * padding_ratio)
    x0 = max(0, int(round(min_x - pad_x)))
    y0 = max(0, int(round(min_y - pad_y)))
    x1 = min(width, int(round(max_x + pad_x)))
    y1 = min(height, int(round(max_y + pad_y)))
    if x1 - x0 < min_size:
        extra = (min_size - (x1 - x0)) // 2
        x0 = max(0, x0 - extra)
        x1 = min(width, x1 + extra)
    if y1 - y0 < min_size:
        extra = (min_size - (y1 - y0)) // 2
        y0 = max(0, y0 - extra)
        y1 = min(height, y1 + extra)
    return x0, y0, x1, y1


def compute_dfl_geometry(keypoints: Dict[str, Point]) -> DFLGeometry:
    """Derive all geometry required for the DFL visualization."""
    left_wrist = keypoints.get("left_wrist")
    right_wrist = keypoints.get("right_wrist")
    nose = keypoints.get("nose")
    left_elbow = keypoints.get("left_elbow")
    right_elbow = keypoints.get("right_elbow")
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")
    if not (left_wrist and right_wrist and nose and left_elbow and right_elbow):
        raise RuntimeError("Missing required wrist/elbow/nose keypoints in the selected frame.")
    nose_dist_left = distance(left_wrist, nose)
    nose_dist_right = distance(right_wrist, nose)
    if nose_dist_left <= nose_dist_right:
        draw_wrist = left_wrist
        draw_elbow = left_elbow
        bow_wrist = right_wrist
    else:
        draw_wrist = right_wrist
        draw_elbow = right_elbow
        bow_wrist = left_wrist
    eye_candidates = []
    if left_eye:
        eye_candidates.append((distance(draw_wrist, left_eye), left_eye))
    if right_eye:
        eye_candidates.append((distance(draw_wrist, right_eye), right_eye))
    if not eye_candidates:
        raise RuntimeError("Missing both eyes; cannot compute the DFL offset.")
    _, draw_eye = min(eye_candidates, key=lambda entry: entry[0])
    adjusted_bow = adjust_bow_wrist(bow_wrist, draw_wrist, nose, draw_eye)
    wrist_to_elbow = distance(draw_wrist, draw_elbow)
    dfl_extra = max(wrist_to_elbow, distance(nose, draw_eye) * 2.0)
    dfl_end = extend_line(adjusted_bow, draw_wrist, wrist_to_elbow + dfl_extra)
    draw_line_extra = max(wrist_to_elbow * 1.2, distance(nose, draw_eye) * 2.5)
    draw_line_end = extend_line(draw_wrist, draw_elbow, draw_line_extra)
    draw_line_start = extend_line(draw_elbow, draw_wrist, draw_line_extra)
    vec_dfl = (dfl_end[0] - draw_wrist[0], dfl_end[1] - draw_wrist[1])
    vec_draw = (draw_line_end[0] - draw_wrist[0], draw_line_end[1] - draw_wrist[1])
    angle_deg = compute_internal_angle_deg(vec_dfl, vec_draw)
    draw_length = distance(draw_wrist, bow_wrist)
    hips = [pt for pt in (left_hip, right_hip) if pt]
    hip_center = None
    hip_width = None
    if hips:
        hip_center = (
            sum(pt[0] for pt in hips) / len(hips),
            sum(pt[1] for pt in hips) / len(hips),
        )
    if left_hip and right_hip:
        hip_width = distance(left_hip, right_hip)
        if hip_width < 1e-6:
            hip_width = None
    draw_length_norm = None
    if hip_width:
        draw_length_norm = draw_length / hip_width
    return DFLGeometry(
        bow_wrist=bow_wrist,
        draw_wrist=draw_wrist,
        draw_elbow=draw_elbow,
        adjusted_bow=adjusted_bow,
        dfl_end=dfl_end,
        draw_line_end=draw_line_end,
        draw_line_start=draw_line_start,
        angle_deg=angle_deg,
        draw_length_px=draw_length,
        normalized_draw_length=draw_length_norm,
        hip_center=hip_center,
        hip_width=hip_width,
        left_wrist_raw=left_wrist,
        right_wrist_raw=right_wrist,
    )


def read_frame(video_path: Path, frame_id: int):
    """Return the requested frame (1-indexed) from the source video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
    return frame


def crop_frame(frame, crop_box: Tuple[int, int, int, int]):
    """Crop the frame to the provided bounding box."""
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1].copy(), (x0, y0)


def draw_lines_and_text(image, offset: Tuple[int, int], geometry: DFLGeometry) -> None:
    """Render the DFL line, draw-arm line, and text annotation."""
    ox, oy = offset

    def to_int(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0] - ox)), int(round(pt[1] - oy)))

    cv2.line(image, to_int(geometry.adjusted_bow), to_int(geometry.dfl_end), DFL_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    cv2.line(
        image,
        to_int(geometry.draw_line_start),
        to_int(geometry.draw_line_end),
        DRAW_ARM_COLOR,
        LINE_THICKNESS,
        cv2.LINE_AA,
    )
    # Draw a dashed orange connector between wrists.
    dash_color = (0, 165, 255)  # orange BGR
    start = to_int(geometry.left_wrist_raw)
    end = to_int(geometry.right_wrist_raw)
    total_len = math.hypot(end[0] - start[0], end[1] - start[1])
    if total_len >= 1:
        dash_length = 12
        gap_length = 8
        dx = (end[0] - start[0]) / total_len
        dy = (end[1] - start[1]) / total_len
        covered = 0.0
        draw = True
        curr = start
        while covered < total_len:
            segment_len = dash_length if draw else gap_length
            next_pt = (
                curr[0] + dx * segment_len,
                curr[1] + dy * segment_len,
            )
            if draw:
                cv2.line(
                    image,
                    (int(round(curr[0])), int(round(curr[1]))),
                    (int(round(next_pt[0])), int(round(next_pt[1]))),
                    dash_color,
                    3,
                    cv2.LINE_AA,
                )
            curr = next_pt
            covered += segment_len
            draw = not draw
    # no text overlay – metrics only printed to console


def compute_dataset_stats(
    current_pose: Path, data_dir: Path
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    int,
    int,
    List[Tuple[str, float]],
    List[Tuple[str, float]],
]:
    """Compute dataset-wide stats (mean, std) for DFL angle and normalized draw length."""
    current_dataset = infer_dataset_name(current_pose)
    if data_dir.is_file():
        candidates = [data_dir]
    else:
        candidates = sorted(data_dir.glob("*.json"))
    angles: List[float] = []
    lengths: List[float] = []
    angle_details: List[Tuple[str, float]] = []
    length_details: List[Tuple[str, float]] = []
    seen_datasets: Set[str] = set()
    excluded = load_excluded_run_ids()
    if excluded:
        print(f"ℹ️  Excluding runs from DFL stats: {sorted(excluded)}")
    for path in candidates:
        if path.resolve() == current_pose.resolve():
            continue
        ds_name = infer_dataset_name(path)
        if current_dataset and ds_name == current_dataset:
            continue
        if ds_name and ds_name in excluded:
            print(f"  ↪︎ skipping {path.name} (excluded)")
            continue
        if ds_name:
            if ds_name in seen_datasets:
                continue
            seen_datasets.add(ds_name)
        try:
            frame = load_pose_frame(path)
            geometry = compute_dfl_geometry(frame.keypoints)
        except Exception as exc:  # pragma: no cover - log and continue
            print(f"⚠️  Skipping {path.name}: {exc}")
            continue
        angles.append(geometry.angle_deg)
        angle_details.append((path.name, geometry.angle_deg))
        if geometry.normalized_draw_length is not None:
            lengths.append(geometry.normalized_draw_length)
            length_details.append((path.name, geometry.normalized_draw_length))
        else:
            print(f"⚠️  Skipping draw length for {path.name}: missing hip width.")
    angle_stats = None
    length_stats = None
    if angles:
        avg = statistics.mean(angles)
        std = statistics.pstdev(angles)
        angle_stats = (avg, std)
    if lengths:
        avg = statistics.mean(lengths)
        std = statistics.pstdev(lengths)
        length_stats = (avg, std)
    return angle_stats, length_stats, len(angles), len(lengths), angle_details, length_details


def main() -> None:
    args = parse_args()
    pose_path = args.pose.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    pose_frame = load_pose_frame(pose_path)
    geometry = compute_dfl_geometry(pose_frame.keypoints)

    video_path = args.video.expanduser() if args.video else infer_video_path(pose_path)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Video not found. Provide --video explicitly.")
    frame = read_frame(video_path, pose_frame.frame_id)

    coords_for_crop = list(pose_frame.all_points) + geometry.line_points()
    crop_box = compute_crop_box(frame.shape, coords_for_crop)
    cropped, offset = crop_frame(frame, crop_box)
    draw_lines_and_text(cropped, offset, geometry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output.expanduser().resolve() if args.output else OUTPUT_DIR / f"{pose_path.stem}_force.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cropped)

    print(f"🎯 DFL angle (frame {pose_frame.frame_id}): {geometry.angle_deg:.2f}°")
    if geometry.normalized_draw_length is not None:
        print(
            f"📏 Draw length: {geometry.normalized_draw_length:.2f} hip-widths "
            f"(raw {geometry.draw_length_px:.1f}px)"
        )
    else:
        print("📏 Draw length: unavailable (missing hip keypoints).")
    (
        angle_stats,
        length_stats,
        angle_count,
        length_count,
        angle_details,
        length_details,
    ) = compute_dataset_stats(pose_path, args.data_dir)
    if angle_count:
        if angle_stats:
            print(f"📊 Other DFL angles ({angle_count} files): avg={angle_stats[0]:.2f}°, σ={angle_stats[1]:.2f}°")
        if angle_details:
            print("📂 Individual DFL angles:")
            for name, val in sorted(angle_details, key=lambda entry: entry[1], reverse=True):
                print(f"   {name}: {val:.2f}°")
    else:
        print("ℹ️  No other pose JSONs found for DFL angle statistics.")
    if length_count:
        if length_stats:
            print(
                f"📊 Other draw lengths ({length_count} files): "
                f"avg={length_stats[0]:.2f} hip-widths, σ={length_stats[1]:.2f} hip-widths"
            )
        if length_details:
            print("📂 Individual draw lengths:")
            for name, val in sorted(length_details, key=lambda entry: entry[1], reverse=True):
                print(f"   {name}: {val:.2f} hip-widths")
    else:
        print("ℹ️  No other pose JSONs with reliable hip widths for draw-length statistics.")
    angle_status, angle_desc = classify_metric(
        geometry.angle_deg,
        angle_stats,
        "dfl angle",
        "°",
        angle_count,
    )
    length_status, length_desc = classify_metric(
        geometry.normalized_draw_length,
        length_stats,
        "draw length",
        " hip-widths",
        length_count,
        grace=0.03,
    )
    notes_to_append: List[str] = []
    if angle_status == "outside":
        notes_to_append.append(f"{pose_path.stem} - {angle_desc}")
    if length_status == "outside":
        notes_to_append.append(f"{pose_path.stem} - {length_desc}")
    notes_text = load_dfl_notes()
    def fmt(num: Optional[float]) -> str:
        if num is None:
            return "NA"
        return f"{num:.2f}"

    angle_avg = angle_stats[0] if angle_stats else None
    angle_std = angle_stats[1] if angle_stats else None
    length_avg = length_stats[0] if length_stats else None
    length_std = length_stats[1] if length_stats else None
    length_value = geometry.normalized_draw_length

    angle_data_block = "\n".join(
        [
            "DATA_START",
            f"dfl_angle_deg={fmt(geometry.angle_deg)}",
            f"dfl_angle_avg_deg={fmt(angle_avg)}",
            f"dfl_angle_std_deg={fmt(angle_std)}",
            f"draw_force_angle_sample_count={angle_count}",
            f"draw_force_min_samples_required={MIN_DATASET_SAMPLES}",
            "DATA_END",
            "",
        ]
    )
    length_data_lines = [
        "DATA_START",
    ]
    if length_value is not None:
        length_data_lines.append(f"draw_length_hipwidths={fmt(length_value)}")
    else:
        length_data_lines.append("draw_length_hipwidths=NA")
    length_data_lines.extend(
        [
            f"draw_length_avg_hipwidths={fmt(length_avg)}",
            f"draw_length_std_hipwidths={fmt(length_std)}",
            f"draw_length_sample_count={length_count}",
            f"draw_length_min_samples_required={MIN_DATASET_SAMPLES}",
            "DATA_END",
            "",
        ]
    )
    length_data_block = "\n".join(length_data_lines)

    sections_written = False
    rel_image = Path("..") / OUTPUT_DIR.name / out_path.name

    # Angle analysis: only query the assistant when outside the window; otherwise log a good/unavailable result.
    ANGLE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    angle_report = ANGLE_ANALYSIS_DIR / f"{pose_path.stem}.md"
    angle_prompt = load_prompt(PROMPT_DFL_ANGLE)
    if angle_status == "within":
        angle_content = (
            f"{angle_data_block}## DFL Angle\nResult is good: {angle_desc}\n\n"
            f"![Draw-force visualization]({rel_image.as_posix()})\n"
        )
        angle_report.write_text(angle_content, encoding="utf-8")
        print(f"📝 Draw-force angle analysis saved to {angle_report} (no assistant call; angle within window)")
        sections_written = True
    elif angle_status == "outside":
        angle_question = "\n".join(
            [
                angle_prompt,
                "",
                f"Video ID: {pose_path.stem}",
                f"DFL angle: {geometry.angle_deg:.2f}°",
                f"Assessment: {angle_desc}",
                "Issue flagged: provide concise corrective guidance (<120 words).",
            ]
        ).strip()
        angle_response = call_rag_assistant(
            angle_question,
            image_path=out_path,
            system_prompt=DFL_SYSTEM_PROMPT,
            notes_text=notes_text,
        )
        if angle_response:
            angle_content = (
                f"{angle_data_block}## DFL Angle\n{angle_response.strip()}\n\n"
                f"![Draw-force visualization]({rel_image.as_posix()})\n"
            )
            angle_report.write_text(angle_content, encoding="utf-8")
            print(f"📝 Draw-force angle analysis saved to {angle_report}")
            sections_written = True
    else:
        angle_content = (
            f"{angle_data_block}## DFL Angle\nAnalysis unavailable: {angle_desc}\n\n"
            f"![Draw-force visualization]({rel_image.as_posix()})\n"
        )
        angle_report.write_text(angle_content, encoding="utf-8")
        print(f"ℹ️  Draw-force angle analysis saved to {angle_report} (no assistant call; insufficient data)")
        sections_written = True

    # Length analysis: only query when outside the window.
    if geometry.normalized_draw_length is not None:
        LENGTH_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        length_report = LENGTH_ANALYSIS_DIR / f"{pose_path.stem}.md"
        if length_status == "within":
            length_content = (
                f"{length_data_block}## Draw Length\nResult is good: {length_desc}\n\n"
                f"![Draw-force visualization]({rel_image.as_posix()})\n"
            )
            length_report.write_text(length_content, encoding="utf-8")
            print(f"📝 Draw length analysis saved to {length_report} (no assistant call; length within window)")
            sections_written = True
        elif length_status == "outside":
            direction = "too long" if "higher" in length_desc else "too short"
            length_prompt = load_prompt(PROMPT_DFL_LENGTH)
            length_question = "\n".join(
                [
                    length_prompt,
                    "",
                    f"Video ID: {pose_path.stem}",
                    f"Draw length is {direction} compared to the reference window.",
                    "Issue flagged: provide concise corrective guidance (<120 words).",
                ]
            ).strip()
            length_response = call_rag_assistant(
                length_question,
                image_path=out_path,
                system_prompt=DFL_SYSTEM_PROMPT,
                notes_text=notes_text,
            )
            if length_response:
                length_content = (
                    f"{length_data_block}## Draw Length\n{length_response.strip()}\n\n"
                    f"![Draw-force visualization]({rel_image.as_posix()})\n"
                )
                length_report.write_text(length_content, encoding="utf-8")
                print(f"📝 Draw-force length analysis saved to {length_report}")
                sections_written = True
        else:
            length_content = (
                f"{length_data_block}## Draw Length\nAnalysis unavailable: {length_desc}\n\n"
                f"![Draw-force visualization]({rel_image.as_posix()})\n"
            )
            length_report.write_text(length_content, encoding="utf-8")
            print(f"ℹ️  Draw length analysis saved to {length_report} (no assistant call; insufficient data)")
            sections_written = True
    else:
        print("ℹ️  Normalized draw length unavailable; skipping length analysis.")

    if not sections_written:
        print("⚠️  Draw-force assistant analysis unavailable.")
    for note in notes_to_append:
        append_dfl_note(note)
    print(f"🖼️  Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
