#!/usr/bin/env python3
"""Transcode a single video to 60 fps for the STRAIGHT pipeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcode a video to 60 fps MP4.")
    parser.add_argument("--input", required=True, type=Path, help="Source video path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination path (default: training_videos/<input_stem>.mp4).",
    )
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def transcode(source: Path, dest: Path, fps: int, overwrite: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"⏭️  Skipping transcode; destination exists: {dest}")
        return
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    print(f"🎬 Transcoding {source} → {dest} at {fps} fps ...")
    subprocess.run(cmd, check=True)
    print(f"✅ Done: {dest}")


def main() -> None:
    args = parse_args()
    src = args.input.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")
    dst = args.output.expanduser() if args.output else Path("training_videos") / f"{src.stem}.mp4"
    transcode(src, dst, args.fps, overwrite=args.force)


if __name__ == "__main__":
    main()
