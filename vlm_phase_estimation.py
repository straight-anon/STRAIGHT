#!/usr/bin/env python3
"""
Interactive VLM-driven phase estimator.

Workflow
--------
1. Prompt the user for a video number (e.g., "1" → nf0001).
2. Load the corresponding video + mmpose inference JSON.
3. Use the deterministic rest→draw heuristic plus a binary search (draw→release)
   that asks GPT-4o if a single frame is released or not.
4. Once close to release, switch to a two-frame confirmation (prompt_two.txt)
   that sends consecutive frames to ensure we captured the exact transition.
5. Save the resulting timeline under estimated_labels as
   vlm_estimated_label_<video>.json.

Set OPENAI_API_KEY in your shell before running this script.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from openai import OpenAI

from phase_estimation import (
    PhaseEstimationConfig,
    PhaseLabel,
    estimate_phase_transitions,
    normalize_phase_label,
)

VIDEO_DIR = Path("training_videos")
POSE_DIR = Path("inference_data")
PROMPTS_DIR = Path("config/prompts")
PROMPT_PATH = PROMPTS_DIR / "vlm_default.txt"
PROMPT_BACKUP_PATH = PROMPTS_DIR / "vlm_backup.txt"
PROMPT_ST_PATH = PROMPTS_DIR / "vlm_st.txt"
OUTPUT_DIR = Path("estimated_labels")
DEBUG_DIR = Path("debug_vlm_calls")
DEFAULT_MODEL = "gpt-4o"
ENABLE_CROPPING = True
FACE_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
ANCHOR_ELBOW_THRESHOLD_DEG = 25.0
ANCHOR_STABILITY_WINDOW = 10
ANCHOR_STABILITY_MAX_DELTA = 1.0
VLM_EXPOSURE_OFFSET = 19  # corresponds to slider value 119 (offset -100..+100)
VLM_SATURATION_SCALE = 1.23
VLM_CONTRAST_SCALE = 1.25
VLM_SHARPNESS_SCALE = 2.19
VLM_BLACK_POINT = 42
DEFAULT_FACE_EXTRA_HORIZONTAL_PAD_PERCENT = 32.0  # percent of crop width
DEFAULT_FACE_EXTRA_DOWNWARD_PAD_PERCENT = 170.0  # percent of crop height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate draw-to-release phases using visual-language prompts."
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="Video identifier without extension (e.g., nfst001).",
    )
    parser.add_argument(
        "--use-st-prompt",
        action="store_true",
        help="Use the ST-specific prompt instead of the default prompt.",
    )
    parser.add_argument(
        "--enable-debug",
        action="store_true",
        help="Require confirmation before every VLM request.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help="Directory that stores training videos.",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=POSE_DIR,
        help="Directory containing inference JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write estimated label JSON files.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEBUG_DIR,
        help="Directory to store debugging crops and logs.",
    )
    return parser.parse_args()


def _env_percent(name: str, default_percent: float) -> float:
    """
    Read a percentage value from the environment and return it as a ratio.
    """
    raw = os.environ.get(name)
    if raw is None:
        return max(0.0, default_percent / 100.0)
    try:
        value = float(raw)
    except ValueError:
        print(f"⚠️  Expected numeric percent for {name}, falling back to {default_percent}%.")
        return max(0.0, default_percent / 100.0)
    return max(0.0, value / 100.0)


FACE_EXTRA_HORIZONTAL_PAD_RATIO = _env_percent(
    "FACE_EXTRA_HORIZONTAL_PAD_PERCENT", DEFAULT_FACE_EXTRA_HORIZONTAL_PAD_PERCENT
)
FACE_EXTRA_DOWNWARD_PAD_RATIO = _env_percent(
    "FACE_EXTRA_DOWNWARD_PAD_PERCENT", DEFAULT_FACE_EXTRA_DOWNWARD_PAD_PERCENT
)


# -----------------------------------------------------------------------------
# Video + pose utilities
# -----------------------------------------------------------------------------
@dataclass
class VideoContext:
    video_id: str
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float



def open_video(video_id: str) -> Tuple[cv2.VideoCapture, VideoContext]:
    path = VIDEO_DIR / f"{video_id}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    duration = frame_count / fps if fps > 0 else 0.0

    ctx = VideoContext(
        video_id=video_id,
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration,
    )
    return cap, ctx


def load_pose_sequence(pose_path: Path, frame_count: int) -> List[Dict[str, Tuple[float, float]]]:
    """
    Convert mmpose JSON output into a PhaseEstimation-friendly sequence.
    Frames without detections yield empty dictionaries to keep indexing simple.
    """
    if not pose_path.exists():
        raise FileNotFoundError(f"MMPose results not found: {pose_path}")

    with open(pose_path, "r") as f:
        data = json.load(f)

    meta = data.get("meta_info", {})
    num_keypoints = int(meta.get("num_keypoints", 0))
    id_to_name = meta.get("keypoint_id2name", {})
    keypoint_names: List[str] = [id_to_name.get(str(i), str(i)) for i in range(num_keypoints)]

    sequence: List[Dict[str, Tuple[float, float]]] = [dict() for _ in range(frame_count)]
    for frame in data.get("instance_info", []):
        frame_idx = max(0, int(frame.get("frame_id", 1)) - 1)
        if frame_idx >= frame_count:
            continue
        instances = frame.get("instances", [])
        if not instances:
            continue
        keypoints = instances[0].get("keypoints", [])
        if not keypoints:
            continue
        sequence[frame_idx] = {
            name: (float(x), float(y))
            for name, (x, y) in zip(keypoint_names, keypoints)
        }

    return sequence


# -----------------------------------------------------------------------------
# Cropping helpers
# -----------------------------------------------------------------------------
def extend_crop_down_and_horizontal(
    crop_box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    horizontal_ratio: float,
    downward_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    Expand the crop horizontally and downward without lifting the top edge.
    """
    if horizontal_ratio <= 0 and downward_ratio <= 0:
        return crop_box

    height, width = frame_shape[:2]
    x0, y0, x1, y1 = crop_box
    base_w = max(1, x1 - x0)
    base_h = max(1, y1 - y0)
    extra_horizontal = int(round(base_w * horizontal_ratio))
    extra_downward = int(round(base_h * downward_ratio))

    if extra_horizontal > 0:
        x0 = max(0, x0 - extra_horizontal)
        x1 = min(width, x1 + extra_horizontal)
    if extra_downward > 0:
        y1 = min(height, y1 + extra_downward)

    return x0, y0, x1, y1


def compute_person_crop(
    frame_shape: Tuple[int, int, int],
    keypoints: Dict[str, Tuple[float, float]],
    padding_ratio: float = 0.2,
    min_box: int = 240,
) -> Tuple[Tuple[int, int, int, int], bool]:
    """
    Build a crop around the full person by covering all detected keypoints.
    Returns ((x0, y0, x1, y1), used_fallback_flag).
    """
    height, width = frame_shape[:2]

    used_fallback = False
    coords: List[Tuple[float, float]] = list(keypoints.values())

    if not coords:
        # Fallback: focus on upper center where the archer is expected.
        used_fallback = True
        box_w = int(width * 0.45)
        box_h = int(height * 0.75)
        x0 = max(0, width // 2 - box_w // 2)
        y0 = max(0, int(height * 0.05))
        x1 = min(width, x0 + box_w)
        y1 = min(height, y0 + box_h)
    else:
        min_x = min(pt[0] for pt in coords)
        max_x = max(pt[0] for pt in coords)
        min_y = min(pt[1] for pt in coords)
        max_y = max(pt[1] for pt in coords)

        base_w = max(max_x - min_x, 1.0)
        base_h = max(max_y - min_y, 1.0)
        pad_x = max(min_box * 0.1, padding_ratio * base_w)
        pad_y = max(min_box * 0.1, padding_ratio * base_h)

        x0 = max(0, int(min_x - pad_x))
        y0 = max(0, int(min_y - pad_y))
        x1 = min(width, int(max_x + pad_x))
        y1 = min(height, int(max_y + pad_y))

        if x1 - x0 < min_box:
            extra = (min_box - (x1 - x0)) // 2
            x0 = max(0, x0 - extra)
            x1 = min(width, x1 + extra)
        if y1 - y0 < min_box:
            extra = (min_box - (y1 - y0)) // 2
            y0 = max(0, y0 - extra)
            y1 = min(height, y1 + extra)

    crop_box = extend_crop_down_and_horizontal(
        (x0, y0, x1, y1),
        frame_shape,
        FACE_EXTRA_HORIZONTAL_PAD_RATIO,
        FACE_EXTRA_DOWNWARD_PAD_RATIO,
    )
    return crop_box, used_fallback


def compute_face_crop(
    frame_shape: Tuple[int, int, int],
    keypoints: Dict[str, Tuple[float, float]],
    padding_ratio: float = 0.35,
    min_box: int = 100,
) -> Tuple[Tuple[int, int, int, int], bool]:
    """
    Crop tightly around the face region using facial keypoints.
    Falls back to the upper portion of the person crop when keypoints are missing.
    """
    height, width = frame_shape[:2]
    face_coords = [keypoints[name] for name in FACE_KEYPOINTS if name in keypoints]

    if face_coords:
        min_x = min(pt[0] for pt in face_coords)
        max_x = max(pt[0] for pt in face_coords)
        min_y = min(pt[1] for pt in face_coords)
        max_y = max(pt[1] for pt in face_coords)

        base_w = max(max_x - min_x, 1.0)
        base_h = max(max_y - min_y, 1.0)
        pad_x = max(min_box * 0.15, padding_ratio * base_w)
        pad_y = max(min_box * 0.15, padding_ratio * base_h)

        x0 = max(0, int(min_x - pad_x))
        y0 = max(0, int(min_y - pad_y))
        x1 = min(width, int(max_x + pad_x))
        y1 = min(height, int(max_y + pad_y))
    else:
        # No facial landmarks: shrink the person crop to its upper segment.
        (x0, y0, x1, y1), _ = compute_person_crop(frame_shape, keypoints)
        head_h = max(min_box, int((y1 - y0) * 0.45))
        y1 = min(height, y0 + head_h)

    if x1 - x0 < min_box:
        extra = (min_box - (x1 - x0)) // 2
        x0 = max(0, x0 - extra)
        x1 = min(width, x1 + extra)
    if y1 - y0 < min_box:
        extra = (min_box - (y1 - y0)) // 2
        y0 = max(0, y0 - extra)
        y1 = min(height, y1 + extra)

    crop_box = extend_crop_down_and_horizontal(
        (x0, y0, x1, y1),
        frame_shape,
        FACE_EXTRA_HORIZONTAL_PAD_RATIO,
        FACE_EXTRA_DOWNWARD_PAD_RATIO,
    )
    return crop_box, not bool(face_coords)


def crop_frame(frame, crop_box: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1].copy()


def apply_vlm_preprocessing(image):

    return image
    # """
    # Apply the fixed exposure/saturation/contrast/sharpness adjustments the user
    # requested before sending any crop to the VLM.
    # """
    # if image.size == 0:
    #     return image

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # hsv[..., 2] = np.clip(hsv[..., 2] + VLM_EXPOSURE_OFFSET, 0, 255)
    # hsv[..., 1] = np.clip(hsv[..., 1] * VLM_SATURATION_SCALE, 0, 255)
    # adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # adjusted = adjusted * VLM_CONTRAST_SCALE - VLM_BLACK_POINT
    # adjusted = np.clip(adjusted, 0, 255)

    # if VLM_SHARPNESS_SCALE > 1.0:
    #     amount = VLM_SHARPNESS_SCALE - 1.0
    #     blurred = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=1.0)
    #     adjusted = cv2.addWeighted(adjusted, 1.0 + amount, blurred, -amount, 0)
    #     adjusted = np.clip(adjusted, 0, 255)

    # return adjusted.astype(np.uint8)


def compute_right_elbow_angle_deg(frame: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """
    Return the internal angle at the right elbow using landmarks 12-14-16
    (right shoulder → right elbow ← right wrist). Missing keypoints return None.
    """
    try:
        shoulder = frame["right_shoulder"]
        elbow = frame["right_elbow"]
        wrist = frame["right_wrist"]
    except KeyError:
        return None

    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return None

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def detect_anchor_start_frame(
    poses: Sequence[Dict[str, Tuple[float, float]]],
    angle_threshold: float = ANCHOR_ELBOW_THRESHOLD_DEG,
    stability_window: int = ANCHOR_STABILITY_WINDOW,
    stability_delta: float = ANCHOR_STABILITY_MAX_DELTA,
) -> Optional[Tuple[int, float, float]]:
    """
    Locate the first frame where the right-elbow angle stays below the provided
    threshold with < stability_delta fluctuation across stability_window frames.
    """
    if stability_window <= 0:
        return None

    window: deque[float] = deque(maxlen=stability_window)
    for idx, frame in enumerate(poses):
        angle = compute_right_elbow_angle_deg(frame)
        if angle is None:
            window.clear()
            continue

        window.append(angle)
        if len(window) < stability_window:
            continue

        window_range = max(window) - min(window)
        if angle < angle_threshold and window_range < stability_delta:
            start_idx = idx - stability_window + 1
            return max(0, start_idx), angle, window_range

    return None


def build_debug_image_entry(
    *,
    tag: str,
    frame_idx: int,
    crop_path: str,
    crop_box: Tuple[int, int, int, int],
    used_fallback: bool,
) -> Dict[str, object]:
    return {
        "tag": tag,
        "frame_idx": frame_idx,
        "crop_path": crop_path,
        "crop_box": {
            "x0": crop_box[0],
            "y0": crop_box[1],
            "x1": crop_box[2],
            "y1": crop_box[3],
        },
        "used_fallback_crop": used_fallback,
    }


def save_debug_image(video_id: str, frame_idx: int, image, tag: Optional[str] = None) -> str:
    DEBUG_DIR.mkdir(exist_ok=True)
    video_dir = DEBUG_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    name = tag or f"frame_{frame_idx:05d}"
    path = video_dir / f"{name}.jpg"
    cv2.imwrite(str(path), image)
    return str(path)


# -----------------------------------------------------------------------------
# VLM-backed classifier
# -----------------------------------------------------------------------------
class VLMPhaseClassifier:
    """
    Adapter around GPT-4o that inspects face-only crops from the later half of a
    clip to decide when the bowstring leaves the archer's face.
    """

    def __init__(
        self,
        video_id: str,
        video_capture: cv2.VideoCapture,
        poses: Sequence[Dict[str, Tuple[float, float]]],
        prompt_text: str,
        backup_prompt_text: str,
        fps: float = 60.0,
        model: str = DEFAULT_MODEL,
        enable_cropping: bool = True,
        frame_count: int = 0,
        debug_mode: bool = False,
    ):
        self.video_id = video_id
        self.cap = video_capture
        self.poses = poses
        self.prompt_text = prompt_text
        self.backup_prompt_text = backup_prompt_text
        self.fps = fps if fps and fps > 0 else 60.0
        self.model = model
        self.enable_cropping = enable_cropping
        self.frame_count = frame_count
        self.client = OpenAI()
        self._label_cache: Dict[int, PhaseLabel] = {}
        self._debug_mode = debug_mode
        self._using_backup_prompt = False
        self._first_prompt_checked = False
        anchor_info = detect_anchor_start_frame(poses)
        if anchor_info is None:
            fallback = frame_count // 2 if frame_count else 0
            self._anchor_start_idx = fallback
            if frame_count:
                print("⚠️  Unable to detect anchor via elbow angle; defaulting to midpoint gating.")
        else:
            self._anchor_start_idx, anchor_angle, anchor_range = anchor_info
            seconds = self._anchor_start_idx / max(self.fps, 1e-6)
            print(
                f"🎯 Right-elbow anchor detected at frame {self._anchor_start_idx} "
                f"(~{seconds:.2f}s, angle={anchor_angle:.1f}°, Δ={anchor_range:.2f}°)."
            )
        self._late_half_start = self._anchor_start_idx

    def label_release_frame(self, frame_idx: int) -> PhaseLabel:
        if frame_idx < self._late_half_start:
            return PhaseLabel.DRAW

        return self._label_frame_with_vlm(frame_idx)

    def _label_frame_with_vlm(self, frame_idx: int) -> PhaseLabel:
        if frame_idx in self._label_cache:
            return self._label_cache[frame_idx]

        frame, keypoints = self._get_frame_and_pose(frame_idx)
        crop_box, _ = self._compute_crop(frame, keypoints)
        cropped = crop_frame(frame, crop_box)
        processed = apply_vlm_preprocessing(cropped)

        seq_num = len(self._label_cache) + 1
        path = save_debug_image(self.video_id, frame_idx, processed, f"{seq_num}_{frame_idx:05d}")
        self._confirm_before_network(frame_idx, path)

        b64_image = encode_jpeg_base64(processed)
        raw_default = self._query_model(b64_image, prompt_text=self.prompt_text)
        label_default = normalize_phase_label(raw_default)
        raw_backup = None
        label_backup = None

        if not self._first_prompt_checked:
            self._first_prompt_checked = True
            if label_default == PhaseLabel.RELEASE:
                self._using_backup_prompt = True
                print("↪️  First response was 'no'; entering dual requests (default + backup).")
                raw_backup = self._query_model(b64_image, prompt_text=self.backup_prompt_text)
                label_backup = normalize_phase_label(raw_backup)
                label = label_backup
            else:
                label = label_default
        elif self._using_backup_prompt:
            raw_backup = self._query_model(b64_image, prompt_text=self.backup_prompt_text)
            label_backup = normalize_phase_label(raw_backup)
            if label_default == PhaseLabel.DRAW:
                self._using_backup_prompt = False
                label = label_default
            else:
                label = label_backup
        else:
            label = label_default

        self._label_cache[frame_idx] = label
        log_parts = [f"default: '{raw_default.strip()}'"]
        if raw_backup is not None:
            log_parts.append(f"backup: '{raw_backup.strip()}'")
        print(f"🧠 Frame {frame_idx}: model returned { ' | '.join(log_parts) } -> {label.value}")
        return label

    def _get_frame_and_pose(self, frame_idx: int) -> Tuple:
        frame = self._get_frame_with_fallback(frame_idx)
        keypoints = self._pose_for_frame(frame_idx)
        return frame, keypoints

    def _get_frame_with_fallback(self, frame_idx: int):
        idx = max(0, min(frame_idx, self.frame_count - 1))
        for candidate in range(idx, -1, -1):
            frame = self._get_frame(candidate)
            if frame is not None:
                if candidate != frame_idx:
                    print(
                        f"↪️  Requested frame {frame_idx} missing pixels; "
                        f"using frame {candidate} instead."
                    )
                return frame
        raise ValueError(f"Unable to read frame {frame_idx} or any earlier frame.")

    def _compute_crop(
        self, frame, keypoints: Dict[str, Tuple[float, float]]
    ) -> Tuple[Tuple[int, int, int, int], bool]:
        if self.enable_cropping:
            crop_box, used_fallback = compute_face_crop(frame.shape, keypoints)
            if used_fallback:
                print("⚠️  Using fallback crop for face (no reliable keypoints).")
            return crop_box, used_fallback
        height, width = frame.shape[:2]
        return (0, 0, width, height), False

    def _pose_for_frame(self, frame_idx: int) -> Dict[str, Tuple[float, float]]:
        if 0 <= frame_idx < len(self.poses) and self.poses[frame_idx]:
            return self.poses[frame_idx]

        nearest_idx = self._find_nearest_pose(frame_idx)
        if nearest_idx is not None:
            print(f"↪️  Frame {frame_idx} missing pose; reusing frame {nearest_idx}")
            return self.poses[nearest_idx]
        return {}

    def _find_nearest_pose(self, frame_idx: int) -> Optional[int]:
        max_range = max(frame_idx, len(self.poses) - frame_idx - 1)
        for offset in range(1, max_range + 1):
            left = frame_idx - offset
            if 0 <= left < len(self.poses) and self.poses[left]:
                return left
            right = frame_idx + offset
            if 0 <= right < len(self.poses) and self.poses[right]:
                return right
        return None

    def _get_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def _confirm_before_network(self, frame_idx: int, crop_path: str) -> None:
        if not self._debug_mode:
            return

        print(f"🖼️  Crop saved to {crop_path} (frame {frame_idx}). Inspect before sending.")
        while True:
            ans = input("Send this crop to OpenAI? [y/N]: ").strip().lower()
            if ans in {"y", "yes"}:
                return
            if ans in {"n", "no", ""}:
                self._debug_mode = False
                print("🔕 Debug confirmations disabled for subsequent calls.")
                return
            print("Please reply with 'y' or 'n'.")

    def _query_model(self, image_b64: str, *, prompt_text: str) -> str:
        print("🔍 Querying VLM...")
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }
            ],
        )
        text = extract_response_text(response)
        return text


def encode_jpeg_base64(image) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("JPEG encoding failed.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def dump_response(response) -> Dict:
    """Convert OpenAI response object into a JSON-serializable dict."""
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:
            pass
    return {"repr": repr(response)}


def extract_response_text(response) -> str:
    """
    Responses API objects expose `.output` with typed segments, but also
    `.output_text`. Use whichever is available.
    """
    if hasattr(response, "output") and response.output:
        chunks: List[str] = []
        for item in response.output:
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    chunks.append(content.text)
        if chunks:
            return " ".join(chunks)

    if hasattr(response, "output_text"):
        text = "".join(response.output_text)
        if text:
            return text

    raise ValueError("Model response did not contain text.")


# -----------------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    global VIDEO_DIR, POSE_DIR, OUTPUT_DIR, DEBUG_DIR
    VIDEO_DIR = args.video_dir
    POSE_DIR = args.pose_dir
    OUTPUT_DIR = args.output_dir
    DEBUG_DIR = args.debug_dir

    video_id = args.video_id
    pose_path = POSE_DIR / f"results_{video_id}.json"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)

    cap, ctx = open_video(video_id)
    try:
        poses = load_pose_sequence(pose_path, ctx.frame_count)
        prompt_source = PROMPT_ST_PATH if args.use_st_prompt else PROMPT_PATH
        prompt_text = load_prompt_text(prompt_source)
        backup_prompt_text = load_prompt_text(PROMPT_BACKUP_PATH)
        DEBUG_DIR.mkdir(exist_ok=True)
        debug_mode = args.enable_debug
        classifier = VLMPhaseClassifier(
            video_id,
            cap,
            poses,
            prompt_text,
            backup_prompt_text,
            fps=ctx.fps,
            enable_cropping=ENABLE_CROPPING,
            frame_count=ctx.frame_count,
            debug_mode=debug_mode,
        )

        config = PhaseEstimationConfig()
        result = estimate_phase_transitions(
            poses=poses,
            fps=ctx.fps,
            release_label_frame=classifier.label_release_frame,
            config=config,
        )

        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / f"vlm_estimated_label_{video_id}.json"
        payload = {
            "video": f"{video_id}.mp4",
            "fps": ctx.fps,
            "frame_count": ctx.frame_count,
            "duration_sec": ctx.duration_sec,
            "phases": result.as_phase_list(ctx.frame_count),
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"✅ Saved estimates to {output_path}")
        print(f"🗂️  Debug artifacts stored in {DEBUG_DIR / video_id}")
    finally:
        cap.release()


def load_prompt_text(path: Path) -> str:
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"Prompt file not found: {path}")


if __name__ == "__main__":
    main()
