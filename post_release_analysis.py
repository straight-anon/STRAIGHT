"""
1. download all the nfst videos of my w/ shot trainer (I sent you the link to google drive)
2. store them in training_videos/
3. their respective smoothed inference data is already in smoothed_inference_data/
4. their respective estimated label data is already in estimated_labels/

Post-release analysis outline.

This module will:
1. Use the estimated-label JSON (same naming scheme as draw_force_line.py) to locate
   the release frame and define a follow-through window (~20 frames, depending on your choosing, after release).
2. Load the matching smoothed pose JSON (same format as draw_force_line.py relies on) so the
   follow-through analysis uses the stabilized keypoints rather than raw detections.
3. output a visual analysis of two key metrics:

Deliverable:
    * Command-line tool similar to draw_force_line.py that prints a concise textual report
      (and optionally writes a JSON summary) describing the follow-through quality for the
      requested shot.
    * Also emits a visual (PNG) for the release frame: draw guide lines that illustrate the
      draw-hand follow-through and bow-arm alignment so the warnings have a visual reference.

Within the follow-through window:
    A. Check the draw wrist vs. head position.
        - Measure whether the draw wrist stays behind the head (x-position relative to nose/ears).
        - If the wrist moves forward of the head, emit a warning message.
        - Basically, did the archer's draw hand "follow through" properly?

    B. Check for bow-arm collapse.
        - Track the bow wrist/elbow/shoulder alignment across the follow-through frames.
        - Quantify “collapse” via the change in shoulder-elbow-wrist angle or forward drift
          of the bow wrist relative to release.
        - If the forward drift/angle change exceeds a threshold, emit a warning.

CLI structure (to be implemented later):
    * Arguments:
        --pose <smoothed inference JSON>
        --video (optional, if frame inspection is desired)
        --output (optional report path)
    * Steps:
        1. Resolve release frame via estimated labels; gather follow-through frames.
        2. Compute the two metrics above.
        3. Print a concise textual report (and optionally save JSON/PNG snippets).
        (Current implementation inspects only the single frame that occurs 28 frames after release.)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("OpenCV (cv2) is required. Install it via `pip install opencv-python`.") from exc


VIDEO_DIR = Path("training_videos")
OUTPUT_DIR = Path("post_release_visualizations")
LABEL_DIR = Path("estimated_labels")
FOLLOW_FRAME_OFFSET = 17
DRAW_THRESHOLD_PX = 20.0
DRAW_GOOD_RATIO = 0.7
BOW_COLLAPSE_THRESHOLD_DEG = 5.0

Point = Tuple[float, float]


@dataclass
class FrameSample:
    frame_id: int
    keypoints: Dict[str, Point]
    all_points: List[Point]


@dataclass
class FollowThroughAssessment:
    release_frame: int
    analyzed_frames: int
    draw_ratio: Optional[float]
    draw_warning: bool
    bow_angle_drop: Optional[float]
    bow_warning: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate follow-through quality after release.")
    parser.add_argument("--pose", required=True, type=Path, help="Path to smoothed pose JSON.")
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional explicit video path (defaults to training_videos/<pose_stem>.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path (defaults to post_release_visualizations/<pose>_post.png).",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path to save the numeric assessment as JSON.",
    )
    return parser.parse_args()


def infer_dataset_name(pose_path: Path) -> Optional[str]:
    stem = pose_path.stem
    if stem.startswith("results_"):
        stem = stem.split("results_", 1)[1]
    return stem or None


def infer_video_path(pose_path: Path) -> Optional[Path]:
    dataset = infer_dataset_name(pose_path)
    if not dataset:
        return None
    candidate = VIDEO_DIR / f"{dataset}.mp4"
    if candidate.exists():
        return candidate
    return None


def read_estimated_label(dataset: str) -> dict:
    def load(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    primary = LABEL_DIR / f"vlm_estimated_label_{dataset}.json"
    base = None
    if dataset.endswith("_cropped"):
        base_name = dataset.removesuffix("_cropped")
        base = LABEL_DIR / f"vlm_estimated_label_{base_name}.json"

    if primary.exists() and base and base.exists():
        # Prefer the freshest label between cropped and base (avoids stale copies).
        chosen = primary if primary.stat().st_mtime >= base.stat().st_mtime else base
        return load(chosen)
    if primary.exists():
        return load(primary)
    if base and base.exists():
        return load(base)

    raise FileNotFoundError(
        f"Estimated-label JSON not found for dataset '{dataset}' (checked {primary}"
        + (f", {base}" if base else "")
        + ")."
    )


def locate_release_frame(label_data: dict) -> int:
    fps = float(label_data.get("fps", 60.0))
    for phase in label_data.get("phases", []):
        name = str(phase.get("name", "")).lower()
        if "release" not in name:
            continue
        if isinstance(phase.get("start_frame"), int):
            return max(1, int(phase["start_frame"]))
        if "start" in phase:
            try:
                timestamp = float(phase["start"])
                return max(1, int(round(timestamp * fps)))
            except (TypeError, ValueError):
                continue
    raise RuntimeError("Release phase not found in estimated-label data.")


def load_pose_frames(pose_path: Path) -> Tuple[dict, Dict[int, dict]]:
    with pose_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta_info", {})
    frames: Dict[int, dict] = {}
    for frame in data.get("instance_info", []):
        frame_id = int(frame.get("frame_id", 1))
        frames[frame_id] = frame
    if not frames:
        raise RuntimeError(f"No frames found in {pose_path}.")
    return meta, frames


def extract_keypoints(meta: dict, frame_entry: dict) -> Tuple[Dict[str, Point], List[Point]]:
    instances = frame_entry.get("instances", [])
    if not instances:
        return {}, []
    source = instances[0]
    raw_points = source.get("smoothed_keypoints") or source.get("keypoints") or []
    named: Dict[str, Point] = {}
    all_points: List[Point] = []
    for coords in raw_points:
        if isinstance(coords, Sequence) and len(coords) >= 2:
            x = float(coords[0])
            y = float(coords[1])
            if math.isfinite(x) and math.isfinite(y):
                all_points.append((x, y))
    name2id = meta.get("keypoint_name2id") or {}
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


def get_head_point(keypoints: Dict[str, Point]) -> Optional[Point]:
    if keypoints.get("nose"):
        return keypoints["nose"]
    eyes = [keypoints.get("left_eye"), keypoints.get("right_eye")]
    eyes = [pt for pt in eyes if pt]
    if eyes:
        sx = sum(pt[0] for pt in eyes)
        sy = sum(pt[1] for pt in eyes)
        return (sx / len(eyes), sy / len(eyes))
    return None


def identify_roles(keypoints: Dict[str, Point]) -> Dict[str, str]:
    nose = keypoints.get("nose")
    left_wrist = keypoints.get("left_wrist")
    right_wrist = keypoints.get("right_wrist")
    if not (nose and left_wrist and right_wrist):
        raise RuntimeError("Frame is missing nose and/or wrist keypoints required for analysis.")
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
        "draw_elbow": f"{draw_side}_elbow",
        "draw_shoulder": f"{draw_side}_shoulder",
        "bow_wrist": f"{bow_side}_wrist",
        "bow_elbow": f"{bow_side}_elbow",
        "bow_shoulder": f"{bow_side}_shoulder",
    }


def get_frame_sample(frames: Dict[int, dict], meta: dict, frame_id: int) -> Optional[FrameSample]:
    if frame_id not in frames:
        return None
    keypoints, all_points = extract_keypoints(meta, frames[frame_id])
    if not keypoints:
        return None
    return FrameSample(frame_id=frame_id, keypoints=keypoints, all_points=all_points)


def find_nearest_sample(frames: Dict[int, dict], meta: dict, target_frame: int, max_offset: int = 6) -> Optional[FrameSample]:
    """Return the closest available frame sample near target_frame, scanning outward."""
    for delta in range(0, max_offset + 1):
        for candidate in (target_frame + delta, target_frame - delta):
            if candidate < 0:
                continue
            sample = get_frame_sample(frames, meta, candidate)
            if sample:
                return sample
    return None


def evaluate_draw_wrist_follow(
    release_head: Point,
    release_draw_wrist: Point,
    samples: List[FrameSample],
    draw_wrist_name: str,
    threshold_px: float = DRAW_THRESHOLD_PX,
    good_ratio: float = DRAW_GOOD_RATIO,
) -> Tuple[Optional[float], bool]:
    sign = 1.0 if (release_draw_wrist[0] - release_head[0]) >= 0 else -1.0
    total = 0
    good = 0
    for sample in samples:
        head = get_head_point(sample.keypoints)
        draw = sample.keypoints.get(draw_wrist_name)
        if not head or not draw:
            continue
        total += 1
        delta = draw[0] - head[0]
        if delta * sign > 0 and abs(delta) >= threshold_px:
            good += 1
    if total == 0:
        return None, True
    ratio = good / total
    return ratio, ratio < good_ratio


def compute_joint_angle(a: Point, b: Point, c: Point) -> Optional[float]:
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    len1 = math.hypot(*v1)
    len2 = math.hypot(*v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_theta = max(-1.0, min(1.0, dot / (len1 * len2)))
    return math.degrees(math.acos(cos_theta))


def evaluate_bow_arm_collapse(
    release_points: Dict[str, Point],
    samples: List[FrameSample],
    roles: Dict[str, str],
    drop_threshold: float = BOW_COLLAPSE_THRESHOLD_DEG,
) -> Tuple[Optional[float], bool]:
    shoulder = release_points.get(roles["bow_shoulder"])
    elbow = release_points.get(roles["bow_elbow"])
    wrist = release_points.get(roles["bow_wrist"])
    if not (shoulder and elbow and wrist):
        return None, True
    release_angle = compute_joint_angle(shoulder, elbow, wrist)
    if release_angle is None:
        return None, True
    min_angle = release_angle
    for sample in samples:
        sh = sample.keypoints.get(roles["bow_shoulder"])
        el = sample.keypoints.get(roles["bow_elbow"])
        wr = sample.keypoints.get(roles["bow_wrist"])
        if not (sh and el and wr):
            continue
        angle = compute_joint_angle(sh, el, wr)
        if angle is None:
            continue
        if angle < min_angle:
            min_angle = angle
    drop = release_angle - min_angle
    if drop < 0:
        drop = 0.0
    warn = drop > drop_threshold
    return drop, warn


def compute_angle_between_vectors(v1: Point, v2: Point) -> Optional[float]:
    len1 = math.hypot(*v1)
    len2 = math.hypot(*v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_theta = max(-1.0, min(1.0, dot / (len1 * len2)))
    return math.degrees(math.acos(cos_theta))


def compute_draw_to_vertical_angle(head_point: Point, draw_point: Point) -> Optional[float]:
    vector = (draw_point[0] - head_point[0], draw_point[1] - head_point[1])
    return compute_angle_between_vectors(vector, (0.0, -1.0))


def get_hip_point(keypoints: Dict[str, Point], side: str) -> Optional[Point]:
    direct = keypoints.get(f"{side}_hip")
    if direct:
        return direct
    left = keypoints.get("left_hip")
    right = keypoints.get("right_hip")
    if left and right:
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    return None


def compute_bow_arm_torso_angle(keypoints: Dict[str, Point], roles: Dict[str, str]) -> Optional[float]:
    shoulder = keypoints.get(roles["bow_shoulder"])
    wrist = keypoints.get(roles["bow_wrist"])
    bow_side = roles["bow_wrist"].split("_", 1)[0]
    hip = get_hip_point(keypoints, bow_side)
    if not (shoulder and wrist and hip):
        return None
    arm_vec = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
    torso_vec = (hip[0] - shoulder[0], hip[1] - shoulder[1])
    return compute_angle_between_vectors(arm_vec, torso_vec)


def compute_distance(p1: Optional[Point], p2: Optional[Point]) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_crop_box(
    frame_shape: Tuple[int, int, int], coords: Iterable[Point], padding_ratio: float = 0.2, min_size: int = 320
) -> Tuple[int, int, int, int]:
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


def crop_frame(frame, crop_box: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1].copy(), (x0, y0)


def read_frame(video_path: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, frame_id - 1)
    if total_frames > 0:
        target = min(target, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if (not ok or frame is None) and total_frames > 0:
        # Retry the last available frame before failing.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
    return frame


def draw_visual_guides(
    image,
    offset: Tuple[int, int],
    frame_points: Dict[str, Point],
    head_point: Optional[Point],
    roles: Dict[str, str],
    assessment: FollowThroughAssessment,
):
    ox, oy = offset

    def to_int(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0] - ox)), int(round(pt[1] - oy)))

    draw_wrist = frame_points.get(roles["draw_wrist"])
    bow_shoulder = frame_points.get(roles["bow_shoulder"])
    bow_elbow = frame_points.get(roles["bow_elbow"])
    bow_wrist = frame_points.get(roles["bow_wrist"])

    if head_point and draw_wrist:
        cv2.line(image, to_int(head_point), to_int(draw_wrist), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(image, to_int(draw_wrist), 8, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(image, to_int(head_point), 6, (255, 255, 255), -1, cv2.LINE_AA)
    if bow_shoulder and bow_elbow and bow_wrist:
        cv2.line(image, to_int(bow_shoulder), to_int(bow_elbow), (0, 255, 0), 4, cv2.LINE_AA)
        cv2.line(image, to_int(bow_elbow), to_int(bow_wrist), (255, 0, 0), 4, cv2.LINE_AA)
        for pt in (bow_shoulder, bow_elbow, bow_wrist):
            cv2.circle(image, to_int(pt), 6, (255, 255, 255), -1, cv2.LINE_AA)

    text_lines = [
        f"Draw hand: {'⚠️ needs follow-through' if assessment.draw_warning else '✅ behind head'}",
    ]
    if assessment.draw_ratio is not None:
        text_lines[-1] += f" ({assessment.draw_ratio*100:.0f}% of frames)"
    else:
        text_lines[-1] += " (insufficient data)"
    bow_line = (
        f"Bow arm: {'⚠️ collapse detected' if assessment.bow_warning else '✅ stable'}"
    )
    if assessment.bow_angle_drop is not None:
        bow_line += f" (Δ {assessment.bow_angle_drop:.1f}°)"
    else:
        bow_line += " (insufficient data)"
    text_lines.append(bow_line)
    y = 30
    for line in text_lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 35


def draw_dashed_line(
    image,
    start_pt: Tuple[int, int],
    end_pt: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int = 12,
    gap_len: int = 8,
):
    x1, y1 = start_pt
    x2, y2 = end_pt
    total_len = math.hypot(x2 - x1, y2 - y1)
    if total_len < 1e-3:
        return
    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len
    dist = 0.0
    while dist < total_len:
        start = dist
        end = min(dist + dash_len, total_len)
        sx = int(round(x1 + dx * start))
        sy = int(round(y1 + dy * start))
        ex = int(round(x1 + dx * end))
        ey = int(round(y1 + dy * end))
        cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        dist += dash_len + gap_len


def draw_draw_length_overlay(
    image,
    offset: Tuple[int, int],
    head_point: Optional[Point],
    draw_point: Optional[Point],
    label: str,
    y: int = 35,
) -> int:
    ox, oy = offset

    def to_int(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0] - ox)), int(round(pt[1] - oy)))

    if head_point and draw_point:
        nose_px = to_int(head_point)
        draw_px = to_int(draw_point)
        draw_dashed_line(image, nose_px, draw_px, (0, 200, 255), 4)
        cv2.circle(image, nose_px, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, draw_px, 8, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.putText(image, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return y + 40


def draw_bow_torso_overlay(
    image,
    offset: Tuple[int, int],
    shoulder: Optional[Point],
    wrist: Optional[Point],
    hip: Optional[Point],
    angle_deg: Optional[float],
    y: int = 35,
) -> int:
    ox, oy = offset

    def to_int(pt: Point) -> Tuple[int, int]:
        return (int(round(pt[0] - ox)), int(round(pt[1] - oy)))

    if shoulder and wrist and hip:
        shoulder_px = to_int(shoulder)
        wrist_px = to_int(wrist)
        hip_px = to_int(hip)
        cv2.line(image, shoulder_px, hip_px, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(image, shoulder_px, wrist_px, (0, 180, 0), 5, cv2.LINE_AA)
        for pt in (shoulder_px, wrist_px, hip_px):
            cv2.circle(image, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
    label = (
        f"Bow arm vs torso: {angle_deg:.1f} deg"
        if angle_deg is not None
        else "Bow arm vs torso: insufficient data"
    )
    cv2.putText(image, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return y + 40


def concat_side_by_side(img_left, img_right):
    left_h, left_w = img_left.shape[:2]
    right_h, right_w = img_right.shape[:2]
    target_h = max(left_h, right_h)

    def resize_to_height(img, height):
        h, w = img.shape[:2]
        if h == height:
            return img
        scale = height / float(h)
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

    left_resized = resize_to_height(img_left, target_h)
    right_resized = resize_to_height(img_right, target_h)
    return cv2.hconcat([left_resized, right_resized])


def save_json_report(path: Path, assessment: FollowThroughAssessment, extra: Optional[dict] = None) -> None:
    payload = {
        "release_frame": assessment.release_frame,
        "analyzed_frames": assessment.analyzed_frames,
        "draw_ratio": assessment.draw_ratio,
        "draw_warning": assessment.draw_warning,
        "bow_angle_drop": assessment.bow_angle_drop,
        "bow_warning": assessment.bow_warning,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pose_path = args.pose.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    dataset = infer_dataset_name(pose_path)
    if not dataset:
        raise RuntimeError("Unable to infer dataset name from pose path.")
    label_data = read_estimated_label(dataset)
    release_frame = locate_release_frame(label_data)
    meta, frames = load_pose_frames(pose_path)
    if release_frame not in frames:
        raise RuntimeError(f"Pose JSON lacks release frame {release_frame}.")
    release_points, _ = extract_keypoints(meta, frames[release_frame])
    if not release_points:
        raise RuntimeError("Release frame lacks keypoints for analysis.")
    roles = identify_roles(release_points)
    head_point = get_head_point(release_points)
    draw_wrist = release_points.get(roles["draw_wrist"])
    if not head_point or not draw_wrist:
        raise RuntimeError("Release frame is missing head/draw wrist keypoints.")
    pre_frame_id = max(1, release_frame - 1)
    follow_frame_id = release_frame + FOLLOW_FRAME_OFFSET

    pre_sample = get_frame_sample(frames, meta, pre_frame_id)
    if not pre_sample:
        pre_sample = find_nearest_sample(frames, meta, pre_frame_id)
    if not pre_sample:
        raise RuntimeError(f"Pre-release frame {pre_frame_id} missing or lacks keypoints.")
    follow_sample = get_frame_sample(frames, meta, follow_frame_id)
    if not follow_sample:
        follow_sample = find_nearest_sample(frames, meta, follow_frame_id)
    if not follow_sample:
        raise RuntimeError(f"Follow-through frame {follow_frame_id} missing or lacks keypoints.")

    pre_head = get_head_point(pre_sample.keypoints)
    pre_draw = pre_sample.keypoints.get(roles["draw_wrist"])
    pre_draw_len = compute_distance(pre_head, pre_draw)
    pre_bow_torso_angle = compute_bow_arm_torso_angle(pre_sample.keypoints, roles)

    draw_ratio, draw_warning = evaluate_draw_wrist_follow(
        head_point, draw_wrist, [follow_sample], roles["draw_wrist"]
    )
    bow_drop, bow_warning = evaluate_bow_arm_collapse(release_points, [follow_sample], roles)
    assessment = FollowThroughAssessment(
        release_frame=release_frame,
        analyzed_frames=1,
        draw_ratio=draw_ratio,
        draw_warning=draw_warning,
        bow_angle_drop=bow_drop,
        bow_warning=bow_warning,
    )

    follow_head = get_head_point(follow_sample.keypoints)
    follow_draw = follow_sample.keypoints.get(roles["draw_wrist"])
    follow_draw_len = compute_distance(follow_head, follow_draw)
    draw_length_pct_change = None
    if pre_draw_len and pre_draw_len > 1e-6 and follow_draw_len is not None:
        draw_length_pct_change = ((follow_draw_len - pre_draw_len) / pre_draw_len) * 100.0

    bow_torso_angle_follow = compute_bow_arm_torso_angle(follow_sample.keypoints, roles)
    shoulder = follow_sample.keypoints.get(roles["bow_shoulder"])
    wrist = follow_sample.keypoints.get(roles["bow_wrist"])
    hip = get_hip_point(follow_sample.keypoints, roles["bow_wrist"].split("_", 1)[0])
    print(f"Release frame: {release_frame}")
    print(f"Pre-release frame: {pre_sample.frame_id}")
    if pre_sample.frame_id != pre_frame_id:
        print(f"  (requested {pre_frame_id}, using nearest available {pre_sample.frame_id})")
    print(f"Follow-through frame: {follow_sample.frame_id} (release + {FOLLOW_FRAME_OFFSET})")
    if follow_sample.frame_id != follow_frame_id:
        print(f"  (requested {follow_frame_id}, using nearest available {follow_sample.frame_id})")
    if pre_draw_len is None:
        print("⚠️  Unable to compute pre-release nose-to-draw length (missing data).")
    else:
        print(f"Pre-release nose-to-draw length: {pre_draw_len:.1f} px")
    if pre_bow_torso_angle is None:
        print("⚠️  Unable to compute pre-release bow-arm vs torso angle (missing data).")
    else:
        print(f"Pre-release bow-arm vs torso angle: {pre_bow_torso_angle:.1f}°")
    if follow_draw_len is None:
        print("⚠️  Unable to compute follow-through nose-to-draw length (missing data).")
    else:
        print(f"Follow-through nose-to-draw length: {follow_draw_len:.1f} px")
    if draw_length_pct_change is None:
        print("⚠️  Unable to compute draw-length change (missing data).")
    else:
        sign = "+" if draw_length_pct_change >= 0 else ""
        print(f"Nose-to-draw length change: {sign}{draw_length_pct_change:.1f}%")
    if bow_torso_angle_follow is None:
        print("⚠️  Unable to compute follow-through bow-arm vs torso angle (missing data).")
    else:
        print(f"Follow-through bow-arm vs torso angle: {bow_torso_angle_follow:.1f}°")

    video_path = args.video.expanduser() if args.video else infer_video_path(pose_path)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Video not found; provide --video explicitly.")
    pre_frame = read_frame(video_path, pre_frame_id)
    follow_frame = read_frame(video_path, follow_frame_id)

    pre_crop_box = compute_crop_box(pre_frame.shape, pre_sample.all_points)
    pre_cropped, pre_offset = crop_frame(pre_frame, pre_crop_box)
    pre_label = (
        f"Nose-to-draw length: {pre_draw_len:.1f} px" if pre_draw_len is not None else "Nose-to-draw length: insufficient data"
    )
    next_y = draw_draw_length_overlay(pre_cropped, pre_offset, pre_head, pre_draw, pre_label, y=35)
    draw_bow_torso_overlay(pre_cropped, pre_offset, pre_sample.keypoints.get(roles["bow_shoulder"]), pre_sample.keypoints.get(roles["bow_wrist"]), get_hip_point(pre_sample.keypoints, roles["bow_wrist"].split("_", 1)[0]), pre_bow_torso_angle, y=next_y)

    follow_crop_box = compute_crop_box(follow_frame.shape, follow_sample.all_points)
    follow_cropped, follow_offset = crop_frame(follow_frame, follow_crop_box)
    if draw_length_pct_change is None:
        follow_label = (
            f"Nose-to-draw length: {follow_draw_len:.1f} px"
            if follow_draw_len is not None
            else "Nose-to-draw length: insufficient data"
        )
    else:
        sign = "+" if draw_length_pct_change >= 0 else ""
        follow_label = (
            f"Draw length change: {sign}{draw_length_pct_change:.1f}% (pre {pre_draw_len:.1f}px → post {follow_draw_len:.1f}px)"
            if pre_draw_len is not None and follow_draw_len is not None
            else f"Draw length change: {sign}{draw_length_pct_change:.1f}%"
        )
    next_y = draw_draw_length_overlay(follow_cropped, follow_offset, follow_head, follow_draw, follow_label, y=35)
    draw_bow_torso_overlay(follow_cropped, follow_offset, shoulder, wrist, hip, bow_torso_angle_follow, y=next_y)

    combined = concat_side_by_side(pre_cropped, follow_cropped)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output.expanduser().resolve() if args.output else OUTPUT_DIR / f"{pose_path.stem}_pre_follow.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    print(f"🖼️  Saved side-by-side visualization to {out_path}")

    if args.json_report:
        report_path = args.json_report.expanduser().resolve()
        save_json_report(
            report_path,
            assessment,
            extra={
                "nose_draw_length_pre": pre_draw_len,
                "nose_draw_length_follow": follow_draw_len,
                "nose_draw_length_change_pct": draw_length_pct_change,
                "bow_torso_angle_pre": pre_bow_torso_angle,
                "bow_torso_angle_follow": bow_torso_angle_follow,
                "pre_frame": pre_frame_id,
                "follow_frame": follow_frame_id,
            },
        )
        print(f"📝  Saved JSON report to {report_path}")


if __name__ == "__main__":
    main()
