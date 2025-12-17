#!/usr/bin/env python3
"""Simple Kalman smoothing pass for mmpose JSON outputs.

Pick an inference JSON under ``inference_data``, smooth its primary instance
with fixed Kalman settings (including a post-release boost), and write the
result to ``smoothed_inference_data``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

POSE_DIR = Path("inference_data")
OUTPUT_DIR = Path("smoothed_inference_data")
# all frames other than post-release
BASE_SCORE_THRESHOLD = 0.05
BASE_PROCESS_VARIANCE = 0.1
BASE_MEASUREMENT_VARIANCE = 35.0
PRE_DRAW_FRAME_BUFFER = 30
POST_RELEASE_FRAME_BUFFER = 30
# post release
POST_RELEASE_FRAME_WINDOW = 20
POST_RELEASE_PROCESS_VARIANCE = 0.5
POST_RELEASE_MEASUREMENT_VARIANCE = 12.5


class KalmanFilter1D:
    """Constant-velocity Kalman filter for a single dimension."""

    def __init__(self, process_var: float, measurement_var: float):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.configure(process_var, measurement_var)
        self.state: Optional[Tuple[float, float]] = None  # position, velocity
        self.P: Optional[List[List[float]]] = None  # 2x2 covariance matrix

    def configure(self, process_var: float, measurement_var: float) -> None:
        self.process_var = process_var
        self.measurement_var = measurement_var

    def _predict(self, dt: float) -> None:
        if self.state is None:
            return
        x, v = self.state
        x_pred = x + v * dt
        v_pred = v
        if self.P is None:
            self.P = [[1.0, 0.0], [0.0, 1.0]]
        p00, p01 = self.P[0]
        p10, p11 = self.P[1]
        p00_pred = p00 + dt * (p10 + p01) + (dt * dt) * p11 + self.process_var
        p01_pred = p01 + dt * p11
        p10_pred = p10 + dt * p11
        p11_pred = p11 + self.process_var
        self.state = (x_pred, v_pred)
        self.P = [[p00_pred, p01_pred], [p10_pred, p11_pred]]

    def _update(self, measurement: float) -> None:
        if self.state is None or self.P is None:
            self.state = (measurement, 0.0)
            self.P = [[1.0, 0.0], [0.0, 1.0]]
            return
        x_pred, v_pred = self.state
        p00, p01 = self.P[0]
        p10, p11 = self.P[1]
        s = p00 + self.measurement_var
        k0 = p00 / s
        k1 = p10 / s
        residual = measurement - x_pred
        x_new = x_pred + k0 * residual
        v_new = v_pred + k1 * residual
        p00_new = (1.0 - k0) * p00
        p01_new = (1.0 - k0) * p01
        p10_new = p10 - k1 * p00
        p11_new = p11 - k1 * p01
        sym = 0.5 * (p01_new + p10_new)
        self.P = [[p00_new, sym], [sym, p11_new]]
        self.state = (x_new, v_new)

    def step(self, measurement: Optional[float], dt: float = 1.0) -> Optional[float]:
        if self.state is None and measurement is None:
            return None
        if self.state is None:
            self.state = (float(measurement), 0.0)
            self.P = [[1.0, 0.0], [0.0, 1.0]]
            return float(measurement)
        self._predict(dt)
        if measurement is not None:
            self._update(float(measurement))
        return self.state[0]


def list_pose_files() -> List[Path]:
    return sorted(POSE_DIR.glob("*.json"))


def smooth_instances(
    data: dict,
    score_threshold: float,
    process_var: float,
    measurement_var: float,
    release_frame_idx0: Optional[int],
    post_frames: int,
    post_process_var: float,
    post_measurement_var: float,
) -> int:
    filters: Dict[int, Dict[int, Dict[str, KalmanFilter1D]]] = {}
    frames = data.get("instance_info", [])
    total_updates = 0
    for frame in frames:
        instances = frame.get("instances", [])
        if not instances:
            continue
        frame_idx0 = int(frame.get("frame_id", 1)) - 1
        in_post_window = False
        if release_frame_idx0 is not None and post_frames > 0:
            start_idx = release_frame_idx0
            end_idx = release_frame_idx0 + post_frames
            in_post_window = start_idx <= frame_idx0 < end_idx

        if in_post_window:
            frame_process_var = post_process_var
            frame_measurement_var = post_measurement_var
        else:
            frame_process_var = process_var
            frame_measurement_var = measurement_var
        # Keep only the primary detection to avoid smoothing noisy secondaries.
        primary = instances[0]
        frame["instances"] = [primary]
        keypoints = primary.get("keypoints", [])
        scores = primary.get("keypoint_scores") or []
        smoothed: List[List[float]] = []
        inst_filters = filters.setdefault(0, {})
        for kp_idx, point in enumerate(keypoints):
            if len(point) != 2:
                smoothed.append([float(point[0]), float(point[1])])
                continue
            score = scores[kp_idx] if kp_idx < len(scores) else 1.0
            use_measurement = score >= score_threshold
            fx = inst_filters.setdefault(kp_idx, {}).setdefault(
                "x", KalmanFilter1D(process_var, measurement_var)
            )
            fy = inst_filters.setdefault(kp_idx, {}).setdefault(
                "y", KalmanFilter1D(process_var, measurement_var)
            )
            fx.configure(frame_process_var, frame_measurement_var)
            fy.configure(frame_process_var, frame_measurement_var)
            mx = float(point[0]) if use_measurement else None
            my = float(point[1]) if use_measurement else None
            sx = fx.step(mx)
            sy = fy.step(my)
            if sx is None:
                sx = float(point[0])
            if sy is None:
                sy = float(point[1])
            smoothed.append([sx, sy])
            total_updates += 1
        primary["keypoints"] = smoothed
    return total_updates


def infer_dataset_name(pose_path: Path) -> Optional[str]:
    stem = pose_path.stem
    if stem.startswith("results_"):
        stem = stem[len("results_") :]
    return stem or None


def _phase_start_index(phase: dict, label_data: dict) -> Optional[int]:
    if "start_frame" in phase:
        return max(0, int(phase["start_frame"]))
    fps = float(label_data.get("fps", 60.0))
    start = float(phase.get("start", 0.0))
    return max(0, int(round(start * fps)))


def load_phase_bounds(dataset: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    label_path = Path("estimated_labels") / f"vlm_estimated_label_{dataset}.json"
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
            draw_idx = _phase_start_index(phase, data)
        if release_idx is None and "release" in name:
            release_idx = _phase_start_index(phase, data)
    frame_count = None
    if "frame_count" in data:
        frame_count = max(0, int(data["frame_count"]))
    if release_idx is None:
        print(f"⚠️  No release phase found in {label_path.name}")
    return draw_idx, release_idx, frame_count


def limit_frames_to_window(
    data: dict,
    start_idx0: Optional[int],
    end_idx0: Optional[int],
) -> None:
    if start_idx0 is None and end_idx0 is None:
        return
    frames = data.get("instance_info", [])
    trimmed = []
    for frame in frames:
        frame_idx0 = int(frame.get("frame_id", 1)) - 1
        if start_idx0 is not None and frame_idx0 < start_idx0:
            continue
        if end_idx0 is not None and frame_idx0 > end_idx0:
            continue
        trimmed.append(frame)
    data["instance_info"] = trimmed


def trim_label_file(dataset: Optional[str], start_idx0: Optional[int], end_idx0: Optional[int]) -> None:
    if not dataset:
        return
    label_path = Path("estimated_labels") / f"vlm_estimated_label_{dataset}.json"
    if not label_path.exists():
        return
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠️  Failed to read label file {label_path}: {exc}")
        return
    offset = start_idx0 or 0
    new_frames = max(0, (end_idx0 or data.get("frame_count", 0) - 1) - offset + 1)
    phases = data.get("phases", [])
    for phase in phases:
        for key in ("start_frame", "end_frame"):
            if key in phase:
                try:
                    phase[key] = max(0, int(phase[key]) - offset)
                except (TypeError, ValueError):
                    pass
    data["phases"] = phases
    if new_frames:
        data["frame_count"] = new_frames
    label_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✂️  Trimmed labels to window [{offset}, {offset + new_frames - 1}] → {label_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Kalman smoothing to pose JSON files.")
    parser.add_argument("--pose", required=True, type=Path, help="Path to the inference JSON to smooth.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (defaults to smoothed_inference_data/<pose_name>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_path = args.pose.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    dataset_name = infer_dataset_name(pose_path)
    draw_frame_idx0: Optional[int] = None
    release_frame_idx0: Optional[int] = None
    total_frames: Optional[int] = None
    if dataset_name:
        draw_frame_idx0, release_frame_idx0, total_frames = load_phase_bounds(dataset_name)
    else:
        print(f"⚠️  Could not infer dataset name from {pose_path.name}; skipping release window logic.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = args.output.expanduser().resolve() if args.output else OUTPUT_DIR / pose_path.name

    with open(pose_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if total_frames is None:
        frames = data.get("instance_info", [])
        if frames:
            total_frames = max(int(frame.get("frame_id", 0)) for frame in frames)

    # Base (pre-release) noise settings
    score_threshold = BASE_SCORE_THRESHOLD
    base_process_var = BASE_PROCESS_VARIANCE
    base_measurement_var = BASE_MEASUREMENT_VARIANCE

    # Post-release defaults assume faster motion + higher confidence
    post_frames = POST_RELEASE_FRAME_WINDOW
    post_process_var = POST_RELEASE_PROCESS_VARIANCE
    post_measurement_var = POST_RELEASE_MEASUREMENT_VARIANCE

    updates = smooth_instances(
        data,
        score_threshold=score_threshold,
        process_var=base_process_var,
        measurement_var=base_measurement_var,
        release_frame_idx0=release_frame_idx0,
        post_frames=post_frames,
        post_process_var=post_process_var,
        post_measurement_var=post_measurement_var,
    )

    # No trimming; keep all frames to avoid desync.
    window_start_idx0 = None
    window_end_idx0 = None

    data["smoothing_info"] = {
        "method": "kalman_constant_velocity",
        "score_threshold": score_threshold,
        "process_variance": BASE_PROCESS_VARIANCE,
        "measurement_variance": BASE_MEASUREMENT_VARIANCE,
        "source_file": str(pose_path),
        "instances": "primary_only",
        "draw_frame": draw_frame_idx0,
        "release_frame": release_frame_idx0,
        "post_release_frames": post_frames,
        "post_release_process_variance": post_process_var,
        "post_release_measurement_variance": post_measurement_var,
        "window_start_frame": window_start_idx0,
        "window_end_frame": window_end_idx0,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Smoothed {updates} keypoints → {out_path}")


if __name__ == "__main__":
    print("🔧 Kalman smoothing (per-keypoint, constant velocity)")
    main()
