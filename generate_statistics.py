#!/usr/bin/env python3
"""Regenerate statistics.md using pre-release pose measurements."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

DATASETS: List[str] = [
    "nfcore_experiment",
    *[f"nfst{idx:03d}" for idx in range(1, 16)],
]

INFERENCE_DIR = Path("inference_data")
LABEL_DIR = Path("estimated_labels")
OUTPUT_PATH = Path("statistics.md")


Point = Tuple[float, float]
PointMap = Dict[str, Point]
MetricFn = Callable[[PointMap], float]


@dataclass
class Metric:
    key: str
    title: str
    description: str
    compute: MetricFn


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def require(points: PointMap, *names: str) -> List[Point]:
    missing = [name for name in names if name not in points]
    if missing:
        raise KeyError(
            ", ".join(missing)
        )
    return [points[name] for name in names]


METRICS: List[Metric] = [
    Metric(
        key="hip_width",
        title="Hip Width Statistics",
        description=(
            "Distance between left/right hip keypoints measured on the frame immediately "
            "before release."
        ),
        compute=lambda pts: distance(*require(pts, "left_hip", "right_hip")),
    ),
    Metric(
        key="foot_distance",
        title="Foot Distance Statistics",
        description=(
            "Distance between left/right ankles measured right before release."
        ),
        compute=lambda pts: distance(*require(pts, "left_ankle", "right_ankle")),
    ),
    Metric(
        key="torso_length",
        title="Torso Length Statistics",
        description=(
            "Absolute vertical distance between the shoulder midpoint and hip midpoint "
            "on the pre-release frame."
        ),
        compute=lambda pts: abs(
            midpoint(*require(pts, "left_shoulder", "right_shoulder"))[1]
            - midpoint(*require(pts, "left_hip", "right_hip"))[1]
        ),
    ),
    Metric(
        key="left_forearm",
        title="Left Forearm Length Statistics",
        description="Left elbow→left wrist distance right before release.",
        compute=lambda pts: distance(*require(pts, "left_elbow", "left_wrist")),
    ),
    Metric(
        key="right_forearm",
        title="Right Forearm Length Statistics",
        description="Right elbow→right wrist distance right before release.",
        compute=lambda pts: distance(*require(pts, "right_elbow", "right_wrist")),
    ),
    Metric(
        key="left_upper_arm",
        title="Left Upper Arm Length Statistics",
        description="Left shoulder→left elbow distance right before release.",
        compute=lambda pts: distance(*require(pts, "left_shoulder", "left_elbow")),
    ),
    Metric(
        key="right_upper_arm",
        title="Right Upper Arm Length Statistics",
        description="Right shoulder→right elbow distance right before release.",
        compute=lambda pts: distance(*require(pts, "right_shoulder", "right_elbow")),
    ),
]


def load_release_frame(dataset: str) -> Tuple[Optional[int], Optional[str]]:
    label_path = LABEL_DIR / f"vlm_estimated_label_{dataset}.json"
    if not label_path.exists():
        return None, f"Missing estimated label {label_path}"
    with open(label_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    release_phase = None
    for phase in label_data.get("phases", []):
        name = str(phase.get("name", "")).lower()
        if "release" in name:
            release_phase = phase
            break
    if not release_phase:
        return None, f"No release phase in {label_path.name}"
    if "start_frame" in release_phase:
        release_frame = int(release_phase["start_frame"])
    else:
        fps = float(label_data.get("fps", 60.0))
        start = float(release_phase.get("start", 0.0))
        release_frame = int(round(start * fps))
    if release_frame <= 0:
        return None, f"Release frame <= 0 in {label_path.name}"
    return release_frame, None


def load_keypoints_for_frame(dataset: str, frame_idx0: int) -> Tuple[Optional[PointMap], Optional[str]]:
    inference_path = INFERENCE_DIR / f"results_{dataset}.json"
    if not inference_path.exists():
        return None, f"Missing inference JSON {inference_path}"
    with open(inference_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta_info", {})
    name2id = {
        name: int(idx)
        for name, idx in (meta.get("keypoint_name2id") or {}).items()
    }
    frames = {int(item.get("frame_id", 0)): item for item in data.get("instance_info", [])}
    target_frame = frame_idx0 + 1  # inference frame_ids are 1-based
    while target_frame > 0:
        frame_entry = frames.get(target_frame)
        if frame_entry:
            instances = frame_entry.get("instances", [])
            if instances:
                points = build_point_map(name2id, instances[0].get("keypoints", []))
                if points:
                    return points, None
        target_frame -= 1
    return None, f"No detections before frame {frame_idx0} in {inference_path.name}"


def build_point_map(name2id: Dict[str, int], keypoints: List[List[float]]) -> PointMap:
    points: PointMap = {}
    for name, idx in name2id.items():
        if 0 <= idx < len(keypoints):
            coords = keypoints[idx]
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                points[name] = (float(coords[0]), float(coords[1]))
    return points


def format_table(metric: Metric, values: Dict[str, Optional[float]]) -> List[str]:
    lines = [f"## {metric.title}", "", metric.description, ""]
    lines.append("| Dataset | Value (px) |")
    lines.append("|---------|------------|")
    for dataset in DATASETS:
        value = values.get(dataset)
        if value is None:
            formatted = "N/A"
        else:
            formatted = f"{value:.2f}"
        lines.append(f"| {dataset} | {formatted} |")
    numeric = [v for v in values.values() if isinstance(v, (int, float))]
    if numeric:
        mean = statistics.mean(numeric)
        std = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
        lines.append("")
        lines.append(f"Mean across datasets: **{mean:.2f} px**  ")
        lines.append(f"Standard deviation: **{std:.2f} px**")
    lines.append("")
    return lines


def main() -> None:
    dataset_points: Dict[str, Optional[PointMap]] = {}
    dataset_errors: Dict[str, str] = {}
    for dataset in DATASETS:
        release_frame, err = load_release_frame(dataset)
        if err or release_frame is None:
            dataset_points[dataset] = None
            dataset_errors[dataset] = err or "Unknown error"
            continue
        pre_release_frame = release_frame - 1
        points, err = load_keypoints_for_frame(dataset, pre_release_frame)
        if err or points is None:
            dataset_points[dataset] = None
            dataset_errors[dataset] = err or "Unknown error"
            continue
        dataset_points[dataset] = points

    metric_values: Dict[str, Dict[str, Optional[float]]] = {
        metric.key: {} for metric in METRICS
    }
    metric_notes: Dict[str, Dict[str, str]] = {metric.key: {} for metric in METRICS}

    for dataset, points in dataset_points.items():
        if points is None:
            for metric in METRICS:
                metric_values[metric.key][dataset] = None
                metric_notes[metric.key][dataset] = dataset_errors.get(dataset, "")
            continue
        for metric in METRICS:
            try:
                metric_values[metric.key][dataset] = metric.compute(points)
            except KeyError as exc:
                metric_values[metric.key][dataset] = None
                metric_notes[metric.key][dataset] = f"Missing keypoints: {exc}"

    lines: List[str] = []
    lines.append("# Pre-release Pose Statistics")
    lines.append("")
    lines.append(
        "All measurements below are taken from the frame immediately before release, "
        "where the release frame is defined by `estimated_labels/vlm_estimated_label_<dataset>.json`."
    )
    lines.append("")
    missing_datasets = {
        dataset: reason
        for dataset, reason in dataset_errors.items()
        if reason
    }
    if missing_datasets:
        lines.append("## Notes")
        lines.append("")
        for dataset, reason in missing_datasets.items():
            lines.append(f"- {dataset}: {reason} - excluded from numeric aggregates.")
        lines.append("")

    for metric in METRICS:
        lines.extend(format_table(metric, metric_values[metric.key]))

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
