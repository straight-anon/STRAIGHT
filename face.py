"""
YOLO-based face cropper using the `uniface` YOLOv5-Face detector.

Given an input image, it finds the most confident face bounding box and saves a
crop that extends the lower edge by 50% of the face height (clipped to the
image). Usage:

    python face.py --image pp1.png --output face_crop.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2


def _import_uniface():
    """
    Import uniface, falling back to the local .venv if needed.
    """
    try:
        import uniface  # type: ignore
        return uniface
    except ImportError:
        venv_site = (
            Path(__file__).parent
            / ".venv"
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if venv_site.exists():
            sys.path.append(str(venv_site))
            import uniface  # type: ignore

            return uniface
        raise


def detect_best_face(image, conf: float) -> Tuple[int, int, int, int]:
    uniface = _import_uniface()
    faces = uniface.detect_faces(image, method="yolov5face", conf_thresh=conf)
    if not faces:
        raise RuntimeError("No face detected.")
    best = max(faces, key=lambda f: f.get("confidence", 0.0))
    x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
    return x1, y1, x2, y2


def crop_face_with_extra_space(
    image_path: Path, output_path: Path, extra_ratio: float = 0.5, conf: float = 0.5
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image at {image_path}")

    h, w = image.shape[:2]
    x1, y1, x2, y2 = detect_best_face(image, conf=conf)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face_height = max(1, y2 - y1)
    y2_ext = min(h, y2 + int(face_height * extra_ratio))

    crop = image[y1:y2_ext, x1:x2]
    if crop.size == 0:
        raise RuntimeError("Empty crop produced; check detection coordinates.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop):
        raise RuntimeError(f"Failed to write output to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO face cropper with extra downward margin.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=Path, default=Path("face_crop.png"), help="Path to save the cropped face.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument(
        "--extra-ratio",
        type=float,
        default=0.5,
        help="Additional fraction of face height to extend downward in the crop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = crop_face_with_extra_space(
        args.image,
        args.output,
        extra_ratio=args.extra_ratio,
        conf=args.conf,
    )
    print(f"Saved face crop to {result_path}")