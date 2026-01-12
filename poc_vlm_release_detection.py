#!/usr/bin/env python3
"""
VLM release detection that uses YOLO face crops (no pose dependency).

For each frame in a video:
  1) Detect the face using face.py (YOLOv5-Face).
  2) Extend the crop 50% downward.
  3) Send the crop to the configured VLM to classify draw vs release.
Binary search over frames finds the earliest release frame.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import face as face_detector
from phase_estimation import (
    PhaseEstimationConfig,
    PhaseLabel,
    estimate_release_start,
    normalize_phase_label,
)

VIDEO_DIR = Path("training_videos")
PROMPTS_DIR = Path("config/prompts")
PROMPT_PATH = PROMPTS_DIR / "vlm_default.txt"
PROMPT_BACKUP_PATH = PROMPTS_DIR / "vlm_backup.txt"
PROMPT_ST_PATH = PROMPTS_DIR / "vlm_st.txt"
OUTPUT_DIR = Path("estimated_labels")
DEBUG_DIR = Path("debug_vlm_calls")
DEFAULT_MODEL = "gemini-2.5-flash"
TARGET_FPS = 30.0  # normalized timeline
YOLO_FACE_EXTRA_DOWNWARD_RATIO = 0.5  # add 50% of face height below detected box


# -----------------------------------------------------------------------------
# Video utilities
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


def open_video(video_id: str, base_dir: Path = VIDEO_DIR) -> Tuple[cv2.VideoCapture, VideoContext]:
    path = base_dir / f"{video_id}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Validate reported frame count by attempting to read the last frame.
    suspicious_count = False
    if frame_count_prop > 0:
        probe = cv2.VideoCapture(str(path))
        probe.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count_prop - 1))
        ok, _ = probe.read()
        probe.release()
        if not ok:
            suspicious_count = True

    if frame_count_prop == 0 or suspicious_count:
        frame_count = count_decodable_frames(path)
    else:
        frame_count = frame_count_prop

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


def count_decodable_frames(path: Path) -> int:
    """Sequentially decode frames to obtain a reliable count when metadata lies."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def review_flagged_frames(video_path: Path, feedback: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    When the composite is rejected, walk through each queried frame and
    ask the user to confirm whether the stored label was correct.
    Returns a list of per-frame confirmations.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  Unable to open video for review: {video_path}")
        cap = None

    confirmations: List[Dict[str, object]] = []
    for rec in sorted(feedback, key=lambda r: int(r.get("normalized_frame", 0))):
        nf = rec.get("normalized_frame")
        sf = rec.get("source_frame")
        auto_label = rec.get("auto_label")
        final_label = rec.get("final_label")
        raw_default = rec.get("raw_default")
        raw_backup = rec.get("raw_backup")
        prompt_text = f"n={nf} src={sf} auto={auto_label} final={final_label}"
        if raw_default:
            prompt_text += f" default='{raw_default}'"
        if raw_backup:
            prompt_text += f" backup='{raw_backup}'"

        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(sf or 0)))
            ok, frame = cap.read()
            if ok and frame is not None:
                display = compose_with_panel(
                    frame,
                    header=prompt_text,
                    footer_lines=["y=accept, n=reject, q=quit review"],
                )
                cv2.imshow("Frame Review", display)
                cv2.waitKey(1)
            else:
                frame = None
        else:
            frame = None

        while True:
            ans = input(f"{prompt_text} | Accept? [y/n/q]: ").strip().lower()
            if ans in {"y", "yes"}:
                confirmations.append(
                    {
                        "normalized_frame": nf,
                        "source_frame": sf,
                        "accepted": True,
                        "auto_label": auto_label,
                        "final_label": final_label,
                        "raw_default": raw_default,
                        "raw_backup": raw_backup,
                    }
                )
                break
            if ans in {"n", "no"}:
                confirmations.append(
                    {
                        "normalized_frame": nf,
                        "source_frame": sf,
                        "accepted": False,
                        "auto_label": auto_label,
                        "final_label": final_label,
                        "raw_default": raw_default,
                        "raw_backup": raw_backup,
                    }
                )
                break
            if ans in {"q", "quit"}:
                break
            print("Please reply with 'y', 'n', or 'q'.")

        if frame is not None:
            cv2.destroyWindow("Frame Review")

    return confirmations


def discover_experiment_videos(experiment_dirs: Sequence[Path]) -> List[Tuple[str, Path]]:
    vids: List[Tuple[str, Path]] = []
    for folder in experiment_dirs:
        if not folder.exists():
            print(f"⚠️  Experiment directory not found: {folder}")
            continue
        paths = list(folder.glob("*.mp4"))
        def _sort_key(p: Path):
            stem = p.stem
            if stem.startswith("deprecated_"):
                return float("inf")
            try:
                return int(stem)
            except ValueError:
                return stem
        for path in sorted(paths, key=_sort_key):
            stem = path.stem
            if stem.startswith("deprecated_"):
                continue
            vids.append((stem, folder))
    return vids


# -----------------------------------------------------------------------------
# Cropping helpers
# -----------------------------------------------------------------------------
def compute_detector_face_crop(
    frame,
    extra_down_ratio: float = YOLO_FACE_EXTRA_DOWNWARD_RATIO,
    conf: float = 0.5,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Use the YOLO-based face detector (face.py) to get a crop with extra space below.
    """
    try:
        x1, y1, x2, y2 = face_detector.detect_best_face(frame, conf=conf)
    except Exception as exc:
        print(f"⚠️  YOLO face detector failed ({exc}); skipping frame.")
        return None

    height, width = frame.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    x2_ext = min(width, x2 + int(face_w * 0.20))  # expand 20% to the right
    y2_ext = min(height, y2 + int(face_h * extra_down_ratio))

    if x2_ext <= x1 or y2_ext <= y1:
        print("⚠️  YOLO face detector returned invalid box; skipping frame.")
        return None

    return x1, y1, x2_ext, y2_ext


def crop_frame(frame, crop_box: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1].copy()


def save_debug_image(video_id: str, frame_idx: int, image, tag: Optional[str] = None) -> str:
    DEBUG_DIR.mkdir(exist_ok=True)
    video_dir = DEBUG_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    name = tag or f"frame_{frame_idx:05d}"
    path = video_dir / f"{name}.jpg"
    cv2.imwrite(str(path), image)
    return str(path)


def compose_with_panel(
    image,
    header: str,
    footer_lines: Sequence[str],
    bg_color=(0, 0, 0),
    header_color: Tuple[int, int, int] = (255, 255, 255),
    footer_colors: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Return a display image with the original frame on top and a separate info panel below.
    The original image pixels are not modified.
    """
    if image is None or image.size == 0:
        return image

    img = image.copy()
    lines = [header] + list(footer_lines)
    colors: List[Tuple[int, int, int]] = [header_color] + list(
        footer_colors or [(255, 255, 255)] * len(footer_lines)
    )
    if not lines:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    margin_x = 10
    margin_y = 10
    line_gap = 6

    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    max_w = max((w for w, _ in text_sizes), default=0)
    total_text_h = sum((h for _, h in text_sizes)) + line_gap * (len(lines) - 1)
    panel_h = total_text_h + margin_y * 2
    panel_w = max(img.shape[1], max_w + margin_x * 2)

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = bg_color

    y = margin_y
    for (line, (w, h)), color in zip(zip(lines, text_sizes), colors):
        cv2.putText(
            panel,
            line,
            (margin_x, y + h),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += h + line_gap

    # Pad the narrower side to align widths.
    if panel.shape[1] < img.shape[1]:
        pad_w = img.shape[1] - panel.shape[1]
        pad = np.zeros((panel.shape[0], pad_w, 3), dtype=np.uint8)
        pad[:] = bg_color
        panel = np.concatenate([panel, pad], axis=1)
    elif panel.shape[1] > img.shape[1]:
        pad_w = panel.shape[1] - img.shape[1]
        pad = np.zeros((img.shape[0], pad_w, 3), dtype=np.uint8)
        pad[:] = bg_color
        img = np.concatenate([img, pad], axis=1)

    return np.vstack([img, panel])


def apply_vlm_preprocessing(image):
    return image


def _overlay_label(image, text: str, color: Tuple[int, int, int]) -> np.ndarray:
    """Annotate an image with a simple label in the top-left corner."""
    if image is None or image.size == 0:
        return image
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    margin = 12
    cv2.putText(img, text, (margin, margin + 24), font, scale, color, thickness, cv2.LINE_AA)
    return img


def build_release_composite(
    before_frame,
    after_frame,
) -> np.ndarray:
    """
    Stack two frames horizontally with minimal labels for release review.
    """
    if before_frame is None or after_frame is None:
        raise ValueError("Missing frame data for composite.")

    labeled_before = _overlay_label(before_frame, "BEFORE", (0, 255, 0))
    labeled_after = _overlay_label(after_frame, "AFTER", (0, 0, 255))
    return np.hstack([labeled_before, labeled_after])


# -----------------------------------------------------------------------------
# VLM-backed classifier (no pose dependency)
# -----------------------------------------------------------------------------
class VLMReleaseClassifier:
    """
    Adapter around Gemini/OpenAI that inspects face-only crops to decide when
    the bowstring leaves the archer's face.
    """

    def __init__(
        self,
        video_id: str,
        video_capture: cv2.VideoCapture,
        prompt_text: str,
        backup_prompt_text: str,
        src_fps: float,
        src_frame_count: int,
        norm_fps: float = TARGET_FPS,
        norm_frame_count: int = 0,
        debug_mode: bool = False,
        interactive_labels: bool = True,
        enable_backup_prompt: bool = True,
    ):
        self.video_id = video_id
        self.cap = video_capture
        self.prompt_text = prompt_text
        self.backup_prompt_text = backup_prompt_text
        self.src_fps = src_fps if src_fps and src_fps > 0 else 60.0
        self.src_frame_count = max(1, src_frame_count)
        self.norm_fps = norm_fps if norm_fps and norm_fps > 0 else TARGET_FPS
        self.norm_frame_count = max(1, norm_frame_count or int(math.ceil(self.src_frame_count * self.norm_fps / self.src_fps)))
        self.model = DEFAULT_MODEL
        self._gemini_model = None
        self._label_cache: Dict[int, PhaseLabel] = {}
        self._debug_mode = debug_mode
        self._user_correct = 0
        self._user_incorrect = 0
        # Single shared feedback log so multiple videos append to the same file.
        self._feedback_log_path = DEBUG_DIR / "vlm_user_feedback.jsonl"
        self._feedback_records: List[Dict[str, object]] = []
        self._interactive_labels = interactive_labels
        self._enable_backup_prompt = enable_backup_prompt

    def label_release_frame(self, frame_idx: int) -> PhaseLabel:
        return self._label_frame_with_vlm(frame_idx)

    def _label_frame_with_vlm(self, frame_idx: int) -> PhaseLabel:
        if frame_idx in self._label_cache:
            return self._label_cache[frame_idx]

        src_idx = self._source_index(frame_idx)
        frame = self._get_frame(src_idx)
        if frame is None:
            raise ValueError(f"Unable to read source frame {src_idx} (normalized {frame_idx}).")

        crop_box = compute_detector_face_crop(frame)
        if not crop_box:
            raise RuntimeError(f"Face detection failed for normalized frame {frame_idx} (source {src_idx}).")

        cropped = crop_frame(frame, crop_box)
        processed = apply_vlm_preprocessing(cropped)

        seq_num = len(self._label_cache) + 1
        path = save_debug_image(self.video_id, src_idx, processed, f"{seq_num}_n{frame_idx:05d}_s{src_idx:05d}")
        self._confirm_before_network(frame_idx, path)

        b64_image = encode_jpeg_base64(processed)
        raw_default = self._query_model(b64_image, prompt_text=self.prompt_text)
        label_default = map_yes_no_to_phase(raw_default)
        raw_backup = None
        label_backup = None

        # If the default prompt already says "draw"/"yes", accept it and skip backup.
        if self._enable_backup_prompt and label_default == PhaseLabel.RELEASE:
            raw_backup = self._query_model(b64_image, prompt_text=self.backup_prompt_text)
            label_backup = map_yes_no_to_phase(raw_backup)
            # If either prompt says "yes"/draw, keep it as draw; release only if both agree.
            label = PhaseLabel.RELEASE if label_backup == PhaseLabel.RELEASE else PhaseLabel.DRAW
        else:
            label = label_default

        self._label_cache[frame_idx] = label
        log_parts = [f"default: '{raw_default.strip()}'"]
        if raw_backup is not None:
            log_parts.append(f"backup: '{raw_backup.strip()}'")
        print(f"🧠 Frame {frame_idx}: model returned { ' | '.join(log_parts) } -> {label.value}")
        auto_label = label
        if self._interactive_labels:
            label, user_action = self._confirm_label_gui(frame_idx, processed, label, [])
        else:
            user_action = "auto"
        self._append_user_feedback(
            norm_frame_idx=frame_idx,
            src_frame_idx=src_idx,
            auto_label=auto_label,
            final_label=label,
            user_action=user_action,
            raw_default=raw_default,
            raw_backup=raw_backup,
        )
        self._label_cache[frame_idx] = label
        return label

    def _source_index(self, norm_idx: int) -> int:
        ratio = self.src_fps / self.norm_fps
        src_idx = int(round(norm_idx * ratio))
        return max(0, min(self.src_frame_count - 1, src_idx))

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
            ans = input("Send this crop to the VLM? [y/N]: ").strip().lower()
            if ans in {"y", "yes"}:
                return
            if ans in {"n", "no", ""}:
                self._debug_mode = False
                print("🔕 Debug confirmations disabled for subsequent calls.")
                return
            print("Please reply with 'y' or 'n'.")

    def _query_model(self, image_b64: str, *, prompt_text: str) -> str:
        print(f"🔍 Querying VLM ({self.model})...")
        if not self.model.lower().startswith("gemini"):
            raise RuntimeError("Only Gemini models are supported; set a Gemini model (e.g., gemini-3.0-pro).")
        return self._query_gemini_model(image_b64, prompt_text=prompt_text)

    def _query_gemini_model(self, image_b64: str, *, prompt_text: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is required for Gemini models. Install via "
                "`pip install google-generativeai`."
            ) from exc

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY in your environment for Gemini access.")

        if self._gemini_model is None:
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                self.model, generation_config={"temperature": 0.0}
            )

        image_bytes = base64.b64decode(image_b64)
        try:
            response = self._gemini_model.generate_content(
                [prompt_text, {"mime_type": "image/jpeg", "data": image_bytes}]
            )
        except Exception as exc:  # pragma: no cover - passthrough to caller
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        return extract_gemini_response_text(response)

    def _confirm_label_gui(
        self, frame_idx: int, image, label: PhaseLabel, footer_lines: Sequence[str]
    ) -> Tuple[PhaseLabel, str]:
        """
        Ask user to confirm/flip the label for this frame and return the decision.
        Falls back to CLI prompt if the OpenCV window cannot capture key events.
        """
        window = "VLM Label Check"
        header = label.value.upper()
        label_color = (0, 255, 0) if label == PhaseLabel.DRAW else (0, 0, 255)
        display = compose_with_panel(image, header, [], header_color=label_color)
        cv2.imshow(window, display)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in {ord("y"), ord("Y"), 13, 10}:  # Enter also counts
                self._user_correct += 1
                cv2.destroyWindow(window)
                return label, "y"
            if key in {ord("n"), ord("N")}:
                self._user_incorrect += 1
                flipped = PhaseLabel.RELEASE if label == PhaseLabel.DRAW else PhaseLabel.DRAW
                cv2.destroyWindow(window)
                return flipped, "n"
            # If no key event is captured (e.g., headless / unfocused window), fall back to CLI.
            if key in {0xFF, 255}:
                cv2.destroyWindow(window)
                return self._confirm_label_cli(frame_idx, label)
            print("Press 'y' to accept or 'n' to flip.")

    def _confirm_label_cli(self, frame_idx: int, label: PhaseLabel) -> Tuple[PhaseLabel, str]:
        """Text-only confirmation fallback when GUI key capture fails."""
        prompt = f"[CLI] Frame {frame_idx}: model -> {label.value}. Accept? [y/n]: "
        while True:
            ans = input(prompt).strip().lower()
            if ans in {"y", "yes"}:
                self._user_correct += 1
                return label, "y"
            if ans in {"n", "no"}:
                self._user_incorrect += 1
                flipped = PhaseLabel.RELEASE if label == PhaseLabel.DRAW else PhaseLabel.DRAW
                return flipped, "n"
            print("Please reply with 'y' or 'n'.")

    def _append_user_feedback(
        self,
        *,
        norm_frame_idx: int,
        src_frame_idx: int,
        auto_label: PhaseLabel,
        final_label: PhaseLabel,
        user_action: str,
        raw_default: Optional[str],
        raw_backup: Optional[str],
    ) -> None:
        """Persist per-frame user confirmations to a JSONL file."""
        record = {
            "video_id": self.video_id,
            "normalized_frame": norm_frame_idx,
            "source_frame": src_frame_idx,
            "auto_label": auto_label.value,
            "final_label": final_label.value,
            "user_action": user_action,  # 'y' confirms, 'n' flips
            "raw_default": raw_default,
            "raw_backup": raw_backup,
        }
        path = self._feedback_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            json.dump(record, f)
            f.write("\n")
        self._feedback_records.append(record)

    @property
    def feedback_records(self) -> List[Dict[str, object]]:
        return list(self._feedback_records)


def review_release_frame(
    video_id: str,
    classifier: VLMReleaseClassifier,
    release_frame_norm: int,
    norm_frame_count: int,
    save_dir: Path,
) -> Tuple[int, Path, bool]:
    """
    Show a two-frame composite (before/after) to let the user confirm or
    adjust the release frame. Returns (norm_idx, composite_path, accepted_flag).
    """
    window = "Release Review"
    current = int(release_frame_norm)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("➡️  Release review: left/right (or A/D) to adjust, Y/Enter to accept, N to flag for later, Q to keep current and exit.")

    while True:
        before_norm = max(0, current - 1)
        after_norm = min(norm_frame_count - 1, current)
        before_src = classifier._source_index(before_norm)
        after_src = classifier._source_index(after_norm)
        before_frame = classifier._get_frame(before_src)
        after_frame = classifier._get_frame(after_src)

        composite = build_release_composite(before_frame, after_frame)
        comp_path = save_dir / f"{video_id}_release_composite_{current:05d}.jpg"
        cv2.imwrite(str(comp_path), composite)

        cv2.imshow(window, composite)
        key = cv2.waitKey(0) & 0xFF
        if key in {81, 2424832, ord("a"), ord("A")}:  # left arrow / A
            current = max(0, current - 1)
            continue
        if key in {83, 2555904, ord("d"), ord("D")}:  # right arrow / D
            current = min(norm_frame_count - 1, current + 1)
            continue
        if key in {ord("y"), ord("Y"), 13, 10}:  # accept
            cv2.destroyWindow(window)
            return current, comp_path, True
        if key in {ord("q"), ord("Q")}:  # quit, keep current
            cv2.destroyWindow(window)
            return current, comp_path, True
        if key in {ord("n"), ord("N")}:  # flag for later
            cv2.destroyWindow(window)
            return current, comp_path, False
        # Any other key: keep waiting.
        print("Use ←/→ (A/D) to move, Y to accept, N to flag, Q to quit.")


def encode_jpeg_base64(image) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("JPEG encoding failed.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


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


def extract_gemini_response_text(response) -> str:
    """
    Extract text from a google-generativeai response.
    """
    if hasattr(response, "text") and response.text:
        return response.text

    for cand in getattr(response, "candidates", []) or []:
        for part in getattr(cand, "content", getattr(cand, "parts", [])) or []:
            text = getattr(part, "text", None)
            if text:
                return text

    raise ValueError("Gemini response did not contain text.")


def load_prompt_text(path: Path) -> str:
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"Prompt file not found: {path}")


def map_yes_no_to_phase(label_text: str) -> PhaseLabel:
    """
    Interpret a binary yes/no answer where 'yes' means DRAW (string/arrow present)
    and 'no' means RELEASE. Fall back to normalize_phase_label for other phrasing.
    """
    if not label_text:
        return PhaseLabel.DRAW
    cleaned = label_text.strip().lower()
    # Prompts ask if the string/arrow is still visible on the face/hand.
    # Presence => still drawing; absence => release.
    if cleaned.startswith("yes"):
        return PhaseLabel.DRAW
    if cleaned.startswith("no"):
        return PhaseLabel.RELEASE
    return normalize_phase_label(label_text)


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def process_single_video(
    video_id: str,
    video_dir: Path,
    output_dir: Path,
    debug_dir: Path,
    output_subdir: str,
    prompt_text: str,
    backup_prompt_text: str,
    debug_mode: bool,
    force_interactive: bool,
    enable_backup_prompt: bool,
) -> None:
    experiment_output_dir = output_dir / output_subdir
    experiment_output_dir.mkdir(exist_ok=True, parents=True)
    output_path = experiment_output_dir / f"vlm_release_estimate_{video_id}.json"
    print(f"🎬 [{video_id}] Processing video (dir={video_dir})")
    if output_path.exists():
        print(f"⏭️  [{video_id}] Skipping; estimate already exists at {output_path}")
        return

    def _run_pass(interactive_labels: bool, force_debug: bool) -> Dict[str, object]:
        cap, ctx = open_video(video_id, base_dir=video_dir)
        try:
            norm_frame_count = max(
                1, int(math.ceil(ctx.frame_count * TARGET_FPS / max(ctx.fps, 1e-6)))
            )
            poses: List[Dict[str, Tuple[float, float]]] = [dict() for _ in range(norm_frame_count)]
            debug_dir.mkdir(exist_ok=True)
            classifier = VLMReleaseClassifier(
                video_id,
                cap,
                prompt_text,
                backup_prompt_text,
                src_fps=ctx.fps,
                src_frame_count=ctx.frame_count,
                norm_fps=TARGET_FPS,
                norm_frame_count=norm_frame_count,
                debug_mode=force_debug,
                interactive_labels=interactive_labels,
                enable_backup_prompt=enable_backup_prompt,
            )

            config = PhaseEstimationConfig()
            release_frame = estimate_release_start(
                poses=poses,
                fps=ctx.fps,
                config=config,
                draw_start=None,
                label_frame=classifier.label_release_frame,
                num_frames=norm_frame_count,
            )

            if release_frame is None:
                raise RuntimeError("Unable to locate release frame.")

            experiment_output_dir = output_dir / output_subdir
            composite_dir = experiment_output_dir / "composites"
            release_frame, composite_path, accepted = review_release_frame(
                video_id,
                classifier,
                release_frame,
                norm_frame_count,
                composite_dir,
            )

            release_time = release_frame / max(TARGET_FPS, 1e-6)
            src_release_frame = classifier._source_index(release_frame)

            return {
                "ctx": ctx,
                "norm_frame_count": norm_frame_count,
                "release_frame": release_frame,
                "src_release_frame": src_release_frame,
                "release_time": release_time,
                "composite_path": composite_path,
                "queries": classifier.feedback_records,
                "accepted": accepted,
                "user_correct": classifier._user_correct,
                "user_incorrect": classifier._user_incorrect,
            }
        finally:
            cap.release()

    # Single pass; composite review can flag for follow-up but does not rerun.
    if force_interactive:
        result = _run_pass(interactive_labels=True, force_debug=debug_mode)
    else:
        result = _run_pass(interactive_labels=False, force_debug=False)

    ctx = result["ctx"]
    frame_confirmations: List[Dict[str, object]] = []
    if not result["accepted"]:
        frame_confirmations = review_flagged_frames(ctx.path, result["queries"])

    payload = {
        "video": f"{video_id}.mp4",
        "fps": ctx.fps,
        "frame_count": ctx.frame_count,
        "duration_sec": ctx.duration_sec,
        "normalized_fps": TARGET_FPS,
        "normalized_frame_count": result["norm_frame_count"],
        "release": {
            "normalized_frame": int(result["release_frame"]),
            "source_frame": int(result["src_release_frame"]),
            "time_sec": result["release_time"],
        },
        "release_composite_path": str(result["composite_path"]),
        "composite_accepted": bool(result["accepted"]),
        "queries": result["queries"],
        "frame_confirmations": frame_confirmations,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"✅ [{video_id}] Release frame: n={result['release_frame']} "
        f"(src={result['src_release_frame']}) ~{result['release_time']:.2f}s @ {TARGET_FPS} fps"
    )
    print(f"📝 [{video_id}] Saved estimate to {output_path}")
    print(
        f"👀 [{video_id}] User confirmations: {result['user_correct']} ok / "
        f"{result['user_incorrect']} flipped"
    )
    print(f"🗂️  Debug artifacts stored in {debug_dir / video_id}")
    if not result["accepted"]:
        review_log = debug_dir / "composite_review_flags.jsonl"
        review_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "video_id": video_id,
            "release_frame": int(result["release_frame"]),
            "release_composite_path": str(result["composite_path"]),
            "frame_confirmations": frame_confirmations,
        }
        with review_log.open("a") as f:
            json.dump(entry, f)
            f.write("\n")
        print(f"🚩 [{video_id}] Composite flagged; per-frame confirmations logged to {review_log}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate release frame using VLM face crops (no pose dependency).")
    parser.add_argument(
        "--video-id",
        help="Optional single video identifier (stem without extension). If omitted, loops over experiment dirs.",
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
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write release estimate JSON files.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEBUG_DIR,
        help="Directory to store debugging crops and logs.",
    )
    parser.add_argument(
        "--experiment-dirs",
        type=Path,
        nargs="+",
        help="Directories to scan for experiment MP4s when --video-id is not provided.",
    )
    parser.add_argument(
        "--per-frame-confirm",
        action="store_true",
        help="Force per-frame confirmations (skip the fast auto-pass).",
    )
    parser.add_argument(
        "--use-release-trainer",
        action="store_true",
        help="Use the release-trainer prompt (yes=draw, no=release) and disable backup queries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global VIDEO_DIR, OUTPUT_DIR, DEBUG_DIR
    VIDEO_DIR = args.video_dir
    OUTPUT_DIR = args.output_dir
    DEBUG_DIR = args.debug_dir
    output_subdir = args.video_dir.name

    if args.use_release_trainer:
        prompt_source = PROMPT_ST_PATH
        enable_backup_prompt = False
    else:
        prompt_source = PROMPT_ST_PATH if args.use_st_prompt else PROMPT_PATH
        enable_backup_prompt = True

    prompt_text = load_prompt_text(prompt_source)
    backup_prompt_text = prompt_text if not enable_backup_prompt else load_prompt_text(PROMPT_BACKUP_PATH)
    debug_mode = args.enable_debug

    if args.video_id:
        videos = [(args.video_id, VIDEO_DIR)]
    else:
        scan_dirs = args.experiment_dirs if args.experiment_dirs else [VIDEO_DIR]
        videos = discover_experiment_videos(scan_dirs)
        if not videos:
            raise SystemExit("No experiment videos found.")

    for vid, vdir in videos:
        try:
            process_single_video(
                vid,
                vdir,
                OUTPUT_DIR,
                DEBUG_DIR,
                output_subdir,
                prompt_text,
                backup_prompt_text,
                debug_mode,
                force_interactive=args.per_frame_confirm,
                enable_backup_prompt=enable_backup_prompt,
            )
        except Exception as exc:
            print(f"❌ [{vid}] Failed: {exc}")


if __name__ == "__main__":
    main()