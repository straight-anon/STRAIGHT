"""
Structured phase estimation helpers for archery sequences.

Stages (chronological):
  • rest → draw → release

Currently:
  • rest→draw uses a simple left-arm-angle heuristic.
  • draw→release is found via binary search that asks an external labeler
    (e.g., GPT-4o) whether a frame is still “not released” or already “release”.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Type aliases keep things explicit without forcing a concrete pose structure.
# Each frame is expected to provide 2D coordinates for keypoints we care about.
# ---------------------------------------------------------------------------
Keypoint = Tuple[float, float]
PoseFrame = Dict[str, Keypoint]
PoseSequence = Sequence[PoseFrame]
FrameLabelFn = Callable[[int], "PhaseLabel | str"]


class PhaseLabel(str, Enum):
    """Discrete phase outputs expected from the labeler / VLM."""

    REST = "rest"
    DRAW = "draw"  # includes “full draw” responses
    RELEASE = "release"


@dataclass
class PhaseEstimationConfig:
    """Tunables for the heuristics that drive phase transitions."""

    # Rest → draw: minimum arm angle (deg) between torso and left arm.
    left_arm_draw_angle_deg: float = 35.0
    # Number of repeated VLM queries per frame to smooth noisy answers (>=1).
    label_confirmation_rounds: int = 1


@dataclass
class PhaseEstimationResult:
    """Container for phase transition boundaries expressed in frame indices."""

    rest_start_frame: int
    draw_start_frame: Optional[int]
    release_start_frame: Optional[int]

    def as_phase_list(self, total_frames: int) -> List[Dict[str, int]]:
        """
        Build a frame-based timeline list:
            • start_frame/end_frame delimit half-open [start, end) intervals.
            • Missing transition times gracefully collapse the corresponding segments.
        """
        timeline: List[Dict[str, int]] = []
        cursor = self.rest_start_frame

        if self.draw_start_frame is not None:
            timeline.append(
                {"name": "rest", "start_frame": cursor, "end_frame": self.draw_start_frame}
            )
            cursor = self.draw_start_frame

        if self.release_start_frame is not None:
            timeline.append(
                {"name": "draw", "start_frame": cursor, "end_frame": self.release_start_frame}
            )
            cursor = self.release_start_frame

        # Everything after the last known transition is treated as release.
        timeline.append({"name": "release", "start_frame": cursor, "end_frame": total_frames})
        return timeline


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def estimate_phase_transitions(
    poses: PoseSequence,
    fps: float,
    release_label_frame: Optional[FrameLabelFn] = None,
    config: PhaseEstimationConfig | None = None,
) -> PhaseEstimationResult:
    """
    Estimate when each phase begins.

    Notes:
        • All clips are assumed to start in the rest phase (t = 0).
        • The rest→draw heuristic is angle-based.
        • draw→release relies on a binary search that queries a labeler
          (e.g., GPT-4o) for “release” vs “not release”.
    """
    config = config or PhaseEstimationConfig()
    num_frames = len(poses)

    draw_start = estimate_draw_start(poses, fps, config)
    release_start = estimate_release_start(
        poses,
        fps,
        config,
        draw_start,
        release_label_frame,
        num_frames,
    )

    return PhaseEstimationResult(
        rest_start_frame=0,
        draw_start_frame=draw_start,
        release_start_frame=release_start,
    )


def estimate_draw_start(
    poses: PoseSequence, fps: float, config: PhaseEstimationConfig
) -> Optional[int]:
    """
    Detect the moment the archer exits rest and enters the draw phase.

    Strategy:
        Find the first frame where the angle between the torso vector
        (left shoulder → left hip) and the arm vector (left shoulder → left wrist)
        exceeds the configurable threshold.
    """
    _ = fps  # kept for API compatibility
    threshold = config.left_arm_draw_angle_deg

    for idx, frame in enumerate(poses):
        arm_angle = compute_left_arm_angle_deg(frame)
        if arm_angle is None:
            continue  # Skip frames missing keypoints
        if arm_angle >= threshold:
            return idx

    return None


def estimate_release_start(
    poses: PoseSequence,
    fps: float,
    config: PhaseEstimationConfig,
    draw_start: Optional[int],
    label_frame: Optional[FrameLabelFn],
    num_frames: int,
) -> Optional[int]:
    """
    Locate the draw→release transition via binary search.
    """
    _ = poses
    _ = fps  # API compatibility
    if label_frame is None or num_frames == 0:
        return None

    left = 0
    if draw_start is not None:
        left = min(num_frames - 1, max(0, draw_start))
    right = num_frames - 1

    while left < right:
        mid = (left + right) // 2
        label = classify_frame(label_frame, mid, config.label_confirmation_rounds)
        if label == PhaseLabel.RELEASE:
            right = mid
        else:
            left = mid + 1

    return left


def classify_frame(
    label_fn: FrameLabelFn,
    frame_idx: int,
    confirmations: int,
) -> PhaseLabel:
    """Query external labeler and optionally vote across repeated calls."""
    rounds = max(1, confirmations)
    votes: Dict[PhaseLabel, int] = {}

    for _ in range(rounds):
        raw_label = label_fn(frame_idx)
        label = normalize_phase_label(raw_label)
        votes[label] = votes.get(label, 0) + 1

    return max(votes.items(), key=lambda item: item[1])[0]


def normalize_phase_label(label: PhaseLabel | str) -> PhaseLabel:
    """Convert arbitrary (case-insensitive) strings into PhaseLabel enums."""
    if isinstance(label, PhaseLabel):
        return label

    normalized = label.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized in {"yes", "string", "holding"}:
        normalized = "draw"
    if normalized in {"no", "no_string"}:
        normalized = "release"
    if normalized in {"released"}:
        normalized = "release"
    if normalized in {"not_release", "not_released"}:
        normalized = "draw"
    if normalized == "full_draw":
        normalized = "draw"
    try:
        return PhaseLabel(normalized)
    except ValueError as exc:
        raise ValueError(f"Unrecognized phase label: '{label}'") from exc



# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def compute_left_arm_angle_deg(frame: PoseFrame) -> Optional[float]:
    """
    Return the angle (degrees) between the torso and left arm.

    Required keypoints inside `frame`:
        left_shoulder, left_hip, left_wrist
    """
    try:
        shoulder = frame["left_shoulder"]
        hip = frame["left_hip"]
        wrist = frame["left_wrist"]
    except KeyError:
        return None

    torso_vec = vector_from_points(shoulder, hip)
    arm_vec = vector_from_points(shoulder, wrist)

    return angle_between_deg(arm_vec, torso_vec)


def vector_from_points(a: Keypoint, b: Keypoint) -> Tuple[float, float]:
    """Return vector pointing from point a to point b."""
    return (b[0] - a[0], b[1] - a[1])


def angle_between_deg(vec_a: Tuple[float, float], vec_b: Tuple[float, float]) -> Optional[float]:
    """Compute the angle between two vectors in degrees."""
    mag_a = math.hypot(*vec_a)
    mag_b = math.hypot(*vec_b)
    if mag_a == 0 or mag_b == 0:
        return None

    dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_theta))


__all__ = [
    "PhaseEstimationConfig",
    "PhaseEstimationResult",
    "FrameLabelFn",
    "PhaseLabel",
    "normalize_phase_label",
    "estimate_phase_transitions",
    "estimate_draw_start",
    "estimate_release_start",
]
