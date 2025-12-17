#!/usr/bin/env python3
"""
Pose-based release detection with VLM voting and super-resolution crops.

Process each video in a folder:
  - Use uniface RetinaFace detector to locate the face region (no fallback).
  - Crop with generous padding, enhance via super-resolution (when available) + edge boost.
  - Query the VLM with the experiment prompt using three images (first frame, target frame, last frame).
  - Binary search for the first release frame (assume frame 0=draw, last=release).
  - Save before/after crops side by side under release_exp_pairs/<video>_release_pair.jpg.
  - Save debug crops (enhanced only) as PNGs with ordered prefixes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import google.generativeai as genai

from phase_estimation import PhaseLabel, normalize_phase_label

PROMPT_ARROW_PATH = Path("expconfig/prompts/vlm_arrow.txt")
PROMPT_STRING_PATH = Path("expconfig/prompts/vlm_string.txt")
EXPERIMENT_PROMPT_PATH = Path("expconfig/prompts/experiment_prompt.txt")
DEBUG_ROOT = Path("debug_vlm_release")
OUTPUT_ROOT = Path("release_exp_pairs")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv")
EXTRA_X_PAD_LEFT_RATIO = 2.0  # no padding expansion on the left
EXTRA_X_PAD_RIGHT_RATIO = 0.3  # add padding to the right
EXTRA_DOWN_PAD_RATIO = 0.5  # add padding below
FACE_LEFT_TRIM_RATIO = 0.0
LEFT_EXTRA_CUT_RATIO = 0.0
# TOP_DROP_RATIO = 1.08  # drop this fraction of the detected box from the top
TOP_DROP_RATIO = 0.5
BOTTOM_DROP_RATIO = 0.18  # drop this fraction of the detected box from the bottom
CROP_KEEP_W_RATIO = 0.35  # keep 35% of width, anchored at right edge
CROP_KEEP_H_RATIO = 1.0  # keep full height, anchored at bottom edge
FACE_BORDER_PX = 15  # white border around crops for visual separation
LABEL_BANNER_PX = 64
FACE_MIN_H_MULT = 1.0
UPSCALE_FACTOR = 3
MIN_UPSCALED_SHORT_SIDE = 1080
JPEG_QUALITY = 100
UPSCALING_INTERPOLATION = cv2.INTER_LANCZOS4
SATURATION_SCALE = 0.5
SR_MODEL_DIR = Path("expconfig/models/sr")
SR_MODEL_PATH = SR_MODEL_DIR / "EDSR_x3.pb"
MIN_SR_FILE_SIZE = 1000  # bytes; guard against missing/empty weights
SR_MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb"
POSE_MIN_VIS = 0.4
UNIFACE_DET_SIZE = (640, 640)
UNIFACE_CONF = 0.6
USE_GEMINI = True
DEFAULT_MODEL = "gemini-2.5-pro"


def crop_frame(frame, crop_box: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = crop_box
    return frame[y0:y1, x0:x1].copy()


def apply_vlm_preprocessing(image):
    return image


def load_prompt_text(path: Path) -> str:
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"Prompt file not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose-based release detection with VLM voting.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("expvids"),
        help="Folder containing source videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Where to save side-by-side release frame pairs.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEBUG_ROOT,
        help="Folder to stash all crops before each Gemini request.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model id to use for release classification.",
    )
    return parser.parse_args()


def iter_videos(video_dir: Path):
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    videos = [p for p in sorted(video_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        print(f"No videos found under {video_dir}.")
    return videos


def _fallback_face_crop_box(frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Fallback crop: centered upper portion with generous padding."""
    h, w = frame_shape[:2]
    box_w = int(w * 0.6)
    box_h = int(h * 0.65)
    x0 = max(0, w // 2 - box_w // 2)
    y0 = max(0, int(h * 0.05))
    x1 = min(w, x0 + box_w)
    y1 = min(h, y0 + box_h)
    return x0, y0, x1, y1


def ensure_min_crop_size(
    frame_shape: Tuple[int, int, int],
    crop_box: Tuple[int, int, int, int],
    min_w_ratio: float = 0.4,
    min_h_ratio: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Enforce a minimum crop size based on frame dimensions."""
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = crop_box
    min_w = int(w * min_w_ratio)
    min_h = int(h * min_h_ratio)
    cur_w = max(1, x1 - x0)
    cur_h = max(1, y1 - y0)

    if cur_w < min_w:
        diff = min_w - cur_w
        x0 = max(0, x0 - diff // 2)
        x1 = min(w, x1 + diff - diff // 2)
    if cur_h < min_h:
        diff = min_h - cur_h
        y0 = max(0, y0 - diff // 2)
        y1 = min(h, y1 + diff - diff // 2)
    return x0, y0, x1, y1


def ensure_min_crop_from_face(
    frame_shape: Tuple[int, int, int],
    crop_box: Tuple[int, int, int, int],
    face_w: float,
    face_h: float,
    min_w_mult: float = 1.2,
    min_h_mult: float = FACE_MIN_H_MULT,
) -> Tuple[int, int, int, int]:
    """Enforce a minimum crop size based on detected face (or torso) size."""
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = crop_box
    min_w = int(face_w * min_w_mult)
    min_h = int(face_h * min_h_mult)
    cur_w = max(1, x1 - x0)
    cur_h = max(1, y1 - y0)

    if cur_w < min_w:
        diff = min_w - cur_w
        x0 = max(0, x0 - diff // 2)
        x1 = min(w, x1 + diff - diff // 2)
    if cur_h < min_h:
        diff = min_h - cur_h
        y1 = min(h, y1 + diff)
        if y1 - y0 < min_h:
            y0 = max(0, y1 - min_h)
    return x0, y0, x1, y1


def ensure_min_crop_pixels(
    frame_shape: Tuple[int, int, int],
    crop_box: Tuple[int, int, int, int],
    min_w_px: int,
    min_h_px: int,
) -> Tuple[int, int, int, int]:
    """Ensure the crop is at least min_w_px/min_h_px; clamp to frame."""
    if min_w_px <= 0 and min_h_px <= 0:
        return crop_box
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = crop_box
    cur_w = max(1, x1 - x0)
    cur_h = max(1, y1 - y0)
    if cur_w < min_w_px:
        diff = min_w_px - cur_w
        x0 = max(0, x0 - diff // 2)
        x1 = min(w, x1 + diff - diff // 2)
    if cur_h < min_h_px:
        diff = min_h_px - cur_h
        y1 = min(h, y1 + diff)
        if y1 - y0 < min_h_px:
            y0 = max(0, y1 - min_h_px)
    return x0, y0, x1, y1


def padded_box_from_bbox(
    frame_shape: Tuple[int, int, int],
    bbox: Tuple[int, int, int, int],
    x_pad_left_ratio: float = EXTRA_X_PAD_LEFT_RATIO,
    x_pad_right_ratio: float = EXTRA_X_PAD_RIGHT_RATIO,
    down_pad_ratio: float = EXTRA_DOWN_PAD_RATIO,
    left_trim_ratio: float = FACE_LEFT_TRIM_RATIO,
    left_extra_cut_ratio: float = LEFT_EXTRA_CUT_RATIO,
    top_drop_ratio: float = TOP_DROP_RATIO,
    bottom_drop_ratio: float = BOTTOM_DROP_RATIO,
    keep_w_ratio: float = CROP_KEEP_W_RATIO,
    keep_h_ratio: float = CROP_KEEP_H_RATIO,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, y, bw, bh = bbox
    # Start from raw detection and bias toward the right/downward with padding.
    pad_left = int(round(bw * x_pad_left_ratio))
    pad_right = int(round(bw * x_pad_right_ratio))
    extra_down = int(round(bh * down_pad_ratio))
    trim_left = int(round(bw * left_trim_ratio))
    extra_cut_left = int(round(bw * left_extra_cut_ratio))

    x0 = max(0, x - pad_left + trim_left + extra_cut_left)
    x1 = min(w, x + bw + pad_right)
    y0 = max(0, y + int(bh * top_drop_ratio))
    y1 = min(h, y + bh + extra_down)
    if bottom_drop_ratio > 0.0:
        drop = int(round((bh + extra_down) * bottom_drop_ratio))
        if drop > 0:
            y1 = max(y0 + 1, y1 - drop)

    cur_w = max(1, x1 - x0)
    cur_h = max(1, y1 - y0)

    # Anchor width to the right: keep only keep_w_ratio of width.
    if 0.0 < keep_w_ratio < 1.0:
        new_w = max(1, int(cur_w * keep_w_ratio))
        x0 = max(0, x1 - new_w)

    # Anchor height to the bottom: keep only keep_h_ratio of height.
    if 0.0 < keep_h_ratio < 1.0:
        new_h = max(1, int(cur_h * keep_h_ratio))
        y0 = max(0, y1 - new_h)

    return x0, y0, x1, y1


def apply_extra_down_pixels(
    frame_shape: Tuple[int, int, int],
    crop_box: Tuple[int, int, int, int],
    extra_down_px: int,
) -> Tuple[int, int, int, int]:
    if extra_down_px <= 0:
        return crop_box
    h, _ = frame_shape[:2]
    x0, y0, x1, y1 = crop_box
    y1 = min(h, y1 + extra_down_px)
    return x0, y0, x1, y1


def face_box_from_uniface(
    frame,
    input_size: Tuple[int, int] = UNIFACE_DET_SIZE,
    conf_thresh: float = UNIFACE_CONF,
) -> Optional[Tuple[int, int, int, int]]:
    if frame is None:
        return None
    try:
        import uniface  # type: ignore
    except Exception as exc:
        print(f"WARN: uniface not available ({exc}); skipping uniface detection.")
        return None
    try:
        faces = uniface.detect_faces(frame, method="retinaface", input_size=input_size, conf_thresh=conf_thresh)
    except Exception as exc:
        print(f"WARN: uniface detection failed: {exc}")
        return None
    if not faces:
        return None
    frame_h, frame_w = frame.shape[:2]
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0

    def face_center_distance(face) -> float:
        bbox = face.get("bbox")
        if bbox is None or len(bbox) < 4:
            return float("inf")
        x0, y0, x1, y1 = bbox[:4]
        cx = (float(x0) + float(x1)) / 2.0
        cy = (float(y0) + float(y1)) / 2.0
        dx = cx - frame_cx
        dy = cy - frame_cy
        return dx * dx + dy * dy

    faces = sorted(
        faces,
        key=face_center_distance,
    )
    bbox = faces[0].get("bbox")
    if bbox is None or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in bbox[:4]]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    return padded_box_from_bbox(frame.shape, (x0, y0, bw, bh))


def landmarks_to_keypoints(mp, results, frame_shape: Tuple[int, int, int], min_vis: float = POSE_MIN_VIS):
    if results is None or getattr(results, "pose_landmarks", None) is None:
        return {}
    height, width = frame_shape[:2]
    keypoints: Dict[str, Tuple[float, float]] = {}
    for landmark in mp.solutions.pose.PoseLandmark:
        lm = results.pose_landmarks.landmark[landmark]
        if lm.visibility < min_vis:
            continue
        x = max(0.0, min(lm.x * width, width - 1))
        y = max(0.0, min(lm.y * height, height - 1))
        keypoints[landmark.name.lower()] = (float(x), float(y))
    return keypoints


def face_box_from_pose(keypoints: Dict[str, Tuple[float, float]], frame_shape: Tuple[int, int, int]):
    face_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    face_pts = [keypoints[n] for n in face_names if n in keypoints]
    if face_pts:
        xs = [p[0] for p in face_pts]
        ys = [p[1] for p in face_pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        return padded_box_from_bbox(frame_shape, (int(x0), int(y0), int(bw), int(bh)))

    torso_names = ["left_shoulder", "right_shoulder"]
    torso_pts = [keypoints[n] for n in torso_names if n in keypoints]
    if torso_pts:
        xs = [p[0] for p in torso_pts]
        ys = [p[1] for p in torso_pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0) * 0.8
        return padded_box_from_bbox(frame_shape, (int(x0), int(y0 - bh * 0.4), int(bw), int(bh)))

    return _fallback_face_crop_box(frame_shape)


def save_debug_crop(debug_dir: Path, frame_idx: int, image, seq: Optional[int] = None) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{seq:04d}_" if seq is not None else ""
    path = debug_dir / f"{prefix}frame_{frame_idx:05d}.png"
    cv2.imwrite(str(path), image)
    return path


def query_release_status(
    image,
    gemini_model: Optional[genai.GenerativeModel],
    prompt_texts: Tuple[str, ...],
    model: str,
) -> Tuple[PhaseLabel, Tuple[str, ...]]:
    """
    Ask the VLM with multiple prompts. Release only if both vote release; draw otherwise.
    """
    raw_answers: list[str] = []
    labels: list[PhaseLabel] = []
    if USE_GEMINI:
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("JPEG encoding failed for Gemini request.")
        img_bytes = buffer.tobytes()
        for prompt_text in prompt_texts:
            response = (
                gemini_model.generate_content(
                    [prompt_text, {"mime_type": "image/jpeg", "data": img_bytes}],
                    generation_config={"temperature": 0},
                )
                if gemini_model is not None
                else None
            )
            raw = response.text if response is not None else ""
            raw_answers.append(raw)
            try:
                labels.append(normalize_phase_label(raw))
            except ValueError:
                labels.append(PhaseLabel.DRAW)
            print(f"DEBUG Gemini raw response: {raw!r}")
    else:
        raw_answers = ["[Gemini call skipped for face detector debugging]"] * max(len(prompt_texts), 1)
        labels = [PhaseLabel.DRAW for _ in raw_answers]

    final_label = labels[0] if labels else PhaseLabel.DRAW
    return final_label, tuple(raw_answers)


def combine_side_by_side(img_a, img_b):
    if img_a is None:
        return img_b
    if img_b is None:
        return img_a
    target_h = max(img_a.shape[0], img_b.shape[0])

    def resize_keep_aspect(img):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        new_w = int(round(w * (target_h / float(h))))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    a_resized = resize_keep_aspect(img_a)
    b_resized = resize_keep_aspect(img_b)
    return np.concatenate([a_resized, b_resized], axis=1)


def combine_three_side_by_side(img_a, img_b, img_c):
    """Combine up to three images horizontally, skipping any that are None."""
    images = [img for img in (img_a, img_b, img_c) if img is not None]
    if not images:
        return None
    cur = images[0]
    for nxt in images[1:]:
        cur = combine_side_by_side(cur, nxt)
    return cur


def add_white_border(image, border_px: int = FACE_BORDER_PX):
    if image is None or border_px <= 0:
        return image
    return cv2.copyMakeBorder(
        image,
        border_px,
        border_px,
        border_px,
        border_px,
        borderType=cv2.BORDER_CONSTANT,
        value=(77, 238, 234),
    )


def add_label_banner(image, text: str, banner_px: int = LABEL_BANNER_PX):
    if image is None or banner_px <= 0:
        return image
    out = image.copy()
    h, w = out.shape[:2]
    banner_h = min(banner_px, max(20, h // 4))
    cv2.rectangle(out, (0, 0), (w, banner_h), (20, 20, 20), thickness=-1)
    font_scale = max(0.8, banner_h / 32.0)
    thickness = max(2, int(round(font_scale * 2.5)))
    cv2.putText(
        out,
        text,
        (10, int(banner_h * 0.75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def to_gray_bgr(image):
    """
    Convert to grayscale while keeping 3 channels (comment this out to revert to color).
    """
    if image is None:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def extra_sharpen(image, strength: float = 1.0, radius: float = 1.2):
    """
    Lightweight unsharp mask. Comment out its usage to revert the extra sharpening step.
    """
    if image is None:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
    return sharp


def enhance_edges(image):
    if image is None:
        return image
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Contrast boost on luminance with CLAHE + unsharp on L channel to keep color stable.
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    l_blur = cv2.GaussianBlur(l_eq, (0, 0), sigmaX=1.2)
    l_sharp = cv2.addWeighted(l_eq, 1.45, l_blur, -0.45, 0)
    l_sharp = np.clip(l_sharp, 0, 255).astype(np.uint8)

    a_adj = np.clip(128.0 + (a.astype(np.float32) - 128.0) * SATURATION_SCALE, 0.0, 255.0).astype(np.uint8)
    b_adj = np.clip(128.0 + (b.astype(np.float32) - 128.0) * SATURATION_SCALE, 0.0, 255.0).astype(np.uint8)

    lab_eq = cv2.merge((l_sharp, a_adj, b_adj))
    eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Gentle gamma to open shadows without blowing highlights.
    eq_float = np.clip(eq.astype(np.float32) / 255.0, 0.0, 1.0)
    eq_gamma = np.power(eq_float, 0.9)
    eq_gamma = (np.clip(eq_gamma * 255.0, 0.0, 255.0)).astype(np.uint8)

    # Edge-focused detail boost using a bilateral base to avoid haloing.
    eq_f = eq_gamma.astype(np.float32)
    base = cv2.bilateralFilter(eq_f, d=9, sigmaColor=24, sigmaSpace=9)
    detail = eq_f - base
    boosted = np.clip(eq_f + 1.25 * detail, 0.0, 255.0).astype(np.uint8)
    # Gentle global contrast lift to make crops pop.
    boosted = np.clip((boosted.astype(np.float32) - 128.0) * 1.12 + 128.0, 0.0, 255.0).astype(np.uint8)
    return boosted


def upscale_image(
    image,
    scale: float = UPSCALE_FACTOR,
    min_short_side: int = MIN_UPSCALED_SHORT_SIDE,
    interpolation: int = UPSCALING_INTERPOLATION,
):
    """
    Upscale with a target minimum short side and a high-quality kernel.
    """
    if image is None:
        return image
    h, w = image.shape[:2]
    target_scale = max(scale, float(min_short_side) / float(max(1, min(h, w))))
    if target_scale <= 1.0:
        return image.copy()
    new_w = max(1, int(round(w * target_scale)))
    new_h = max(1, int(round(h * target_scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def load_sr_model():
    if not SR_MODEL_PATH.exists() or SR_MODEL_PATH.stat().st_size < MIN_SR_FILE_SIZE:
        print(f"INFO: SR weights missing at {SR_MODEL_PATH}, downloading...")
        SR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request

            urllib.request.urlretrieve(SR_MODEL_URL, SR_MODEL_PATH)
            print(f"INFO: Downloaded SR model to {SR_MODEL_PATH}")
        except Exception as exc:
            print(f"WARN: Unable to download SR model: {exc}")
            return None
    try:
        from cv2 import dnn_superres
    except Exception as exc:
        print(
            f"WARN: dnn_superres not available ({exc}). "
            "Install opencv-contrib-python>=4.10.0 for ML super-resolution."
        )
        return None
    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(SR_MODEL_PATH))
        sr.setModel("edsr", 3)
        print(f"INFO: SR model loaded from {SR_MODEL_PATH}")
        return sr
    except Exception as exc:
        print(f"WARN: Failed to load SR model: {exc}")
        return None


def apply_superres(sr, image, min_short_side: int = MIN_UPSCALED_SHORT_SIDE):
    """
    Apply ML super-resolution if available; otherwise fall back to a high-quality upscale.
    Returns (enhanced_image, used_sr_flag).
    """
    if image is None:
        return image, False
    if sr is None:
        return upscale_image(image, min_short_side=min_short_side), False
    try:
        out = sr.upsample(image)
        if out.shape[:2] == image.shape[:2]:
            print("WARN: SR output shape matches input; superres may not be applying.")
        if min(out.shape[:2]) < min_short_side:
            out = upscale_image(out, scale=1.0, min_short_side=min_short_side)
        return out, True
    except Exception as exc:
        print(f"WARN: SR upsample failed, falling back to resize: {exc}")
        return upscale_image(image, min_short_side=min_short_side), False


def find_last_readable_index(cap, approx_count: int) -> int:
    end = max(approx_count - 1, 0)
    for idx in range(end, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, _ = cap.read()
        if ok:
            return idx
    return -1


def transcode_to_mp4(input_path: Path, output_path: Path) -> bool:
    """Transcode a video to mp4 using ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        print(f"WARN: Failed to transcode {input_path} -> {output_path}: {exc}")
        return False


def ensure_mp4_videos(video_dir: Path) -> list[Path]:
    """
    Ensure all videos have mp4 versions. Returns list of mp4 paths.
    If a source is already mp4, keep it. Otherwise transcode to <stem>.mp4
    if not already present.
    """
    mp4s: list[Path] = []
    for src in sorted(video_dir.iterdir()):
        if not src.is_file():
            continue
        suffix = src.suffix.lower()
        if suffix == ".mp4":
            mp4s.append(src)
            continue
        if suffix not in VIDEO_EXTENSIONS:
            continue

        target = src.with_suffix(".mp4")
        if target.exists():
            mp4s.append(target)
            continue

        print(f"Transcoding {src.name} -> {target.name} ...")
        ok = transcode_to_mp4(src, target)
        if ok:
            mp4s.append(target)
    return mp4s


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    gemini_model: Optional[genai.GenerativeModel],
    prompt_texts: Tuple[str, ...],
    sr_model,
) -> Optional[Path]:
    video_stem = video_path.stem
    output_path = args.output_dir / f"{video_stem}_release_pair.jpg"
    if output_path.exists():
        print(f"Skipping {video_stem}: output already exists at {output_path}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: Could not open {video_path}, skipping.")
        return None
    pose = None
    video_debug_dir = args.debug_dir / video_stem
    label_cache: Dict[int, PhaseLabel] = {}
    frame_cache: Dict[int, np.ndarray] = {}
    crop_cache: Dict[int, np.ndarray] = {}
    first_context: Optional[np.ndarray] = None
    last_context: Optional[np.ndarray] = None
    approx_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sr_available = sr_model is not None
    debug_seq = 0
    last_readable = find_last_readable_index(cap, approx_frame_count)
    if last_readable < 0:
        print(f"WARN: No readable frames in {video_stem}.")
        cap.release()
        return None
    frame_count = last_readable + 1

    def read_frame(frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            return frame
        for idx in range(frame_idx - 1, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok2, frame2 = cap.read()
            if ok2:
                print(f"WARN: Unable to read frame {frame_idx} for {video_stem}; using {idx} instead.")
                return frame2
        return None

    def cached_or_read_frame(frame_idx: int) -> Optional[np.ndarray]:
        cached = frame_cache.get(frame_idx)
        if cached is not None:
            return cached
        return read_frame(frame_idx)

    def choose_crop_box(frame, update_min: bool = True) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        crop_box = face_box_from_uniface(frame)
        detection_source = "uniface" if crop_box is not None else "none"
        return crop_box, detection_source

    def preprocess_frame(frame_idx: int, label: str, update_min: bool) -> Optional[np.ndarray]:
        frame = read_frame(frame_idx)
        if frame is None:
            print(f"WARN: Unable to read frame {frame_idx} ({label}) for {video_stem}.")
            return None
        crop_box, detection_source = choose_crop_box(frame, update_min=update_min)
        if crop_box is None:
            print(f"WARN: No face detected for {label} frame {frame_idx} in {video_stem}.")
            return None
        processed_pre = apply_vlm_preprocessing(crop_frame(frame, crop_box))
        processed_sr, used_sr = apply_superres(sr_model, processed_pre)
        processed = enhance_edges(processed_sr)
        processed = add_white_border(processed)
        # processed = to_gray_bgr(processed)  # uncomment to force grayscale intermediates
        processed = extra_sharpen(processed)  # comment this line to remove extra sharpening
        if used_sr:
            print(
                f"SR applied for {label} frame {frame_idx} "
                f"({processed_pre.shape[1]}x{processed_pre.shape[0]} -> {processed_sr.shape[1]}x{processed_sr.shape[0]})"
            )
        else:
            reason = "SR unavailable" if not sr_available else "SR unavailable for this frame, falling back"
            print(
                f"{reason} for {label} frame {frame_idx} "
                f"({processed_pre.shape[1]}x{processed_pre.shape[0]} -> {processed_sr.shape[1]}x{processed_sr.shape[0]})"
            )
        frame_cache.setdefault(frame_idx, frame.copy())
        crop_cache.setdefault(frame_idx, processed.copy())
        if label in ("first", "last"):
            video_debug_dir.mkdir(parents=True, exist_ok=True)
            named_path = video_debug_dir / f"{label}_0001.png"
            cv2.imwrite(str(named_path), processed)
            print(f"Saved {label} context frame to {named_path}")
        print(f"Prepared {label} frame {frame_idx} [{detection_source}]")
        return processed

    # Precompute context frames (best-effort) once crop helpers exist.
    first_context = preprocess_frame(0, "first", update_min=True)
    last_context = preprocess_frame(frame_count - 1, "last", update_min=True)

    def classify_frame(frame_idx: int) -> Optional[PhaseLabel]:
        nonlocal debug_seq, first_context, last_context
        if frame_idx in label_cache:
            return label_cache[frame_idx]

        frame = read_frame(frame_idx)
        if frame is None:
            print(f"WARN: Unable to read frame {frame_idx} for {video_stem}.")
            return None

        crop_box, detection_source = choose_crop_box(frame, update_min=True)
        if crop_box is None:
            print(f"WARN: No face detected for frame {frame_idx} in {video_stem}; skipping.")
            return None
        cropped = crop_frame(frame, crop_box)
        processed_pre = apply_vlm_preprocessing(cropped)
        processed_sr, used_sr = apply_superres(sr_model, processed_pre)
        processed = enhance_edges(processed_sr)
        processed = add_white_border(processed)
        # processed = to_gray_bgr(processed)  # uncomment to force grayscale intermediates
        processed = extra_sharpen(processed)  # comment this line to remove extra sharpening
        if used_sr:
            print(
                f"SR applied for frame {frame_idx} "
                f"({processed_pre.shape[1]}x{processed_pre.shape[0]} -> {processed_sr.shape[1]}x{processed_sr.shape[0]})"
            )
        else:
            reason = "SR unavailable" if not sr_available else "SR unavailable for this frame, falling back"
            print(
                f"{reason} for frame {frame_idx} "
                f"({processed_pre.shape[1]}x{processed_pre.shape[0]} -> {processed_sr.shape[1]}x{processed_sr.shape[0]})"
            )

        debug_seq += 1
        debug_path = save_debug_crop(video_debug_dir, frame_idx, processed, seq=debug_seq)
        print(f"Saved crop for frame {frame_idx} [{detection_source}] to {debug_path}")

        if first_context is None:
            refreshed = preprocess_frame(0, "first", update_min=False)
            if refreshed is not None:
                first_context = refreshed
        if last_context is None:
            refreshed = preprocess_frame(frame_count - 1, "last", update_min=False)
            if refreshed is not None:
                last_context = refreshed

        composite = combine_three_side_by_side(
            first_context if first_context is not None else processed,
            processed,
            last_context if last_context is not None else processed,
        )
        if composite is None:
            print(f"WARN: Unable to build composite for frame {frame_idx} in {video_stem}.")
            return None
        composite_path = save_debug_crop(video_debug_dir, frame_idx, composite, seq=debug_seq)
        print(f"Saved composite for frame {frame_idx} to {composite_path}")
        try:
            label, raw_answers = query_release_status(
                composite,
                gemini_model,
                prompt_texts,
                args.model,
            )
        except Exception as exc:
            print(f"WARN: Gemini query failed on frame {frame_idx} of {video_stem}: {exc}")
            return None

        answers_str = " | ".join(raw_answers)
        print(f"{video_stem} frame {frame_idx}: {answers_str} -> {label.value}")

        label_cache[frame_idx] = label
        frame_cache[frame_idx] = frame.copy()
        crop_cache[frame_idx] = processed.copy()
        return label

    if frame_count <= 0:
        print(f"WARN: Could not determine frame count for {video_stem}.")
        cap.release()
        return None

    # Binary search assuming first frame = draw, last frame = release.
    low_idx = 0
    high_idx = max(frame_count - 1, 0)
    mid_idx = (low_idx + high_idx) // 2
    mid_label = classify_frame(mid_idx)
    if mid_label is None:
        cap.release()
        return None

    if mid_label == PhaseLabel.RELEASE:
        high_idx = mid_idx
    else:
        low_idx = mid_idx

    while low_idx + 1 < high_idx:
        mid_idx = (low_idx + high_idx) // 2
        mid_label = classify_frame(mid_idx)
        if mid_label is None:
            break
        if mid_label == PhaseLabel.RELEASE:
            high_idx = mid_idx
        else:
            low_idx = mid_idx

    release_idx = high_idx
    before_idx = max(release_idx - 1, 0)

    classify_frame(before_idx)
    classify_frame(release_idx)

    before_frame = crop_cache.get(before_idx)
    if before_frame is None:
        frame = cached_or_read_frame(before_idx)
        if frame is not None:
            crop_box, _ = choose_crop_box(frame, update_min=False)
            if crop_box is None:
                print(f"WARN: No face detected for before-frame {before_idx} in {video_stem}; aborting.")
                cap.release()
                return None
            before_frame_pre = apply_vlm_preprocessing(crop_frame(frame, crop_box))
            before_frame_sr, _ = apply_superres(sr_model, before_frame_pre)
            before_frame = enhance_edges(before_frame_sr)
            before_frame = add_white_border(before_frame)
            # before_frame = to_gray_bgr(before_frame)  # uncomment to force grayscale intermediates
            before_frame = extra_sharpen(before_frame)  # comment this line to remove extra sharpening

    release_frame = crop_cache.get(release_idx)
    if release_frame is None:
        frame = cached_or_read_frame(release_idx)
        if frame is not None:
            crop_box, _ = choose_crop_box(frame, update_min=False)
            if crop_box is None:
                print(f"WARN: No face detected for release-frame {release_idx} in {video_stem}; aborting.")
                cap.release()
                return None
            release_frame_pre = apply_vlm_preprocessing(crop_frame(frame, crop_box))
            release_frame_sr, _ = apply_superres(sr_model, release_frame_pre)
            release_frame = enhance_edges(release_frame_sr)
            release_frame = add_white_border(release_frame)
            # release_frame = to_gray_bgr(release_frame)  # uncomment to force grayscale intermediates
            release_frame = extra_sharpen(release_frame)  # comment this line to remove extra sharpening

    if before_frame is None or release_frame is None:
        print(f"WARN: Missing frames for composite in {video_stem}.")
        cap.release()
        return None

    before_orig = cached_or_read_frame(before_idx)
    release_orig = cached_or_read_frame(release_idx)
    if before_orig is None or release_orig is None:
        print(f"WARN: Missing original frames for {video_stem}; cannot save release pair.")
        cap.release()
        if pose is not None:
            pose.close()
        return None

    before_orig_lab = add_label_banner(before_orig, f"Frame #{before_idx}")
    release_orig_lab = add_label_banner(release_orig, f"Frame #{release_idx}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    composite = combine_side_by_side(before_orig_lab, release_orig_lab)
    cv2.imwrite(str(output_path), composite, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    print(f"Release transition for {video_stem}: before={before_idx} after={release_idx} -> {output_path}")

    # Also write a combined before/after crop into the debug folder.
    release_pair_debug = combine_side_by_side(before_frame, release_frame)
    save_debug_crop(video_debug_dir, release_idx, release_pair_debug, seq=debug_seq + 1)

    cap.release()
    if pose is not None:
        pose.close()
    return output_path


def main():
    args = parse_args()
    if args.debug_dir.exists():
        shutil.rmtree(args.debug_dir)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_texts: list[str] = []
    if EXPERIMENT_PROMPT_PATH.exists():
        prompt_texts.append(load_prompt_text(EXPERIMENT_PROMPT_PATH))
    else:
        print(f"WARN: Prompt not found at {EXPERIMENT_PROMPT_PATH}; skipping.")
    if not prompt_texts:
        raise SystemExit("No prompts found. Ensure the prompt files exist in expconfig/prompts.")

    if not USE_GEMINI:
        print("INFO: Gemini calls are disabled (USE_GEMINI=False); labels will default to DRAW.")

    print(f"Prompt loaded: {prompt_texts[0]!r}")

    sr_model = load_sr_model()
    print(f"SR model available: {sr_model is not None}")
    if USE_GEMINI:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        client = genai.GenerativeModel(args.model)
    else:
        client = None
    mp4_videos = ensure_mp4_videos(args.video_dir)

    for video_path in mp4_videos:
        process_video(video_path, args, client, tuple(prompt_texts), sr_model)


if __name__ == "__main__":
    main()
