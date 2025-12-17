#!/usr/bin/env python3
"""Upload a local training video, run remote MMPose inference, and download results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import paramiko

HOST = "47.186.63.142"
PORT = 52946
SSH_USER = "root"
KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")

REMOTE_USER_HOME = Path("/home/user")
REMOTE_WORKSPACE = Path("/workspace")
REMOTE_MMPPOSE = REMOTE_WORKSPACE / "mmpose"
REMOTE_INPUT = REMOTE_WORKSPACE / "input_videos"
REMOTE_OUTPUT = REMOTE_WORKSPACE / "output_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote inference for a training video.")
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to the local .mp4 video or the filename relative to --training-dir.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("training_videos"),
        help="Directory that stores training videos (used when --video is relative).",
    )
    parser.add_argument(
        "--output-video-dir",
        type=Path,
        default=Path("saved_videos"),
        help="Directory to store rendered inference videos.",
    )
    parser.add_argument(
        "--output-json-dir",
        type=Path,
        default=Path("inference_data"),
        help="Directory to store downloaded pose JSON files.",
    )
    parser.add_argument("--host", default=HOST, help="Remote SSH host.")
    parser.add_argument("--port", type=int, default=PORT, help="Remote SSH port.")
    parser.add_argument("--ssh-user", default=SSH_USER, help="Remote SSH user.")
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=Path(KEY_PATH),
        help="Path to the SSH private key used for authentication.",
    )
    return parser.parse_args()


def resolve_video_path(video_arg: Path, training_dir: Path) -> Path:
    if video_arg.exists():
        return video_arg
    candidate = training_dir / video_arg
    if candidate.exists():
        return candidate
    if video_arg.suffix != ".mp4":
        mp4_candidate = video_arg.with_suffix(".mp4")
        if mp4_candidate.exists():
            return mp4_candidate
        candidate = training_dir / mp4_candidate.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Local video not found: {video_arg}")


def ensure_remote_dirs(sftp, *directories: Path) -> None:
    for directory in directories:
        try:
            sftp.mkdir(str(directory))
        except OSError:
            continue


def safe_download(sftp, remote: Path, local: Path) -> None:
    try:
        sftp.stat(str(remote))
    except FileNotFoundError:
        print(f"❌ Missing: {remote}")
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(str(remote), str(local))
    print(f"Downloaded: {local}")


def main() -> None:
    args = parse_args()
    local_video_path = resolve_video_path(args.video, args.training_dir)
    video_name = local_video_path.name
    remote_input_path = REMOTE_INPUT / video_name

    args.output_video_dir.mkdir(parents=True, exist_ok=True)
    args.output_json_dir.mkdir(parents=True, exist_ok=True)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        args.host,
        port=args.port,
        username=args.ssh_user,
        key_filename=str(args.ssh_key.expanduser()),
    )
    sftp = ssh.open_sftp()
    ensure_remote_dirs(sftp, REMOTE_INPUT, REMOTE_OUTPUT)

    print(f"Uploading {video_name}...")
    sftp.put(str(local_video_path), str(remote_input_path))
    print("Upload complete.")

    remote_cmd = f"""
sudo -iu user bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh &&
conda activate openmmlab &&
cd {REMOTE_MMPPOSE} &&
python3 demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input "{remote_input_path}" \
    --output-root "{REMOTE_OUTPUT}" \
    --save-predictions \
    --device cuda:0 \
    --thickness 2
'
"""

    print("Running remote inference...\n")
    stdin, stdout, stderr = ssh.exec_command(remote_cmd)
    for line in stdout:
        print(line, end="")
    for line in stderr:
        print("[ERR]", line, end="")

    print("\nChecking output files...")
    remote_out_vid = REMOTE_OUTPUT / video_name
    remote_out_json = REMOTE_OUTPUT / f"results_{video_name.replace('.mp4', '')}.json"
    local_vid = args.output_video_dir / remote_out_vid.name
    local_json = args.output_json_dir / remote_out_json.name
    safe_download(sftp, remote_out_vid, local_vid)
    safe_download(sftp, remote_out_json, local_json)

    sftp.close()
    ssh.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
