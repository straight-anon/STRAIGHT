#!/usr/bin/env python3
"""FastAPI bridge exposing STRAIGHT CLI + config to the React frontend."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

import video_analysis_pipeline as pipeline
from video_analysis_pipeline import (
    compute_release_summary,
    detect_video_fps,
    copy_artifact,
    read_dfl_report,
    read_post_release_report,
    read_ghost_report,
    read_spine_report,
    crop_and_banner_video,
    adjust_pose_for_crop,
    export_run_assets,
    update_runs_manifest,
    bundle_run_outputs,
    run_draw_force_analysis,
    run_kalman_smoothing,
    run_post_release_analysis,
    run_skeleton_overlay,
    run_remote_inference,
    run_spine_analysis,
    run_vlm_phase_estimation,
    write_dfl_reports,
    write_post_release_reports,
    write_report,
    transcode_to_training_video,
    ensure_ghost_overlay,
)

import draw_force_line as dfl_mod
import ghost_overlay as ghost_mod
import post_release_analysis as post_mod
import spine_straight as spine_mod
import visualize_skeleton as viz_skel_mod
from video_analysis_pipeline import CROPPED_ASSETS_DIR
from rag_context import get_rag_context

CONFIG_DIR = Path("config")
PROFILE_PATH = CONFIG_DIR / "personal_info.txt"
NOTES_DIR = CONFIG_DIR / "notes"
PROMPTS_DIR = CONFIG_DIR / "prompts"
CHAT_PROMPT_PATH = PROMPTS_DIR / "chat_prompt.txt"
EXCLUSIONS_PATH = CONFIG_DIR / "excluded_runs.json"
GHOST_REFERENCE_PATH = CONFIG_DIR / "ghost_reference.json"
TRAINING_DIR = pipeline.TRAINING_DIR
STREAM_OUTPUT_DIR = pipeline.STREAM_OUTPUT_DIR
SAVED_VIDEOS_DIR = pipeline.SAVED_VIDEOS_DIR
DEFAULT_REFERENCE_ID = "nfst008"
ASSETS_BASE_DIR = pipeline.WEB_ASSETS_RUNS_DIR.parent

MEDIA_DIRECTORIES = {
    "training": TRAINING_DIR,
    "cropped": CROPPED_ASSETS_DIR,
    "saved": SAVED_VIDEOS_DIR,
    "ghost": ghost_mod.OUTPUT_DIR,
    "skeleton": viz_skel_mod.OUTPUT_DIR,
    "spine": spine_mod.OUTPUT_DIR,
    "drawforce": dfl_mod.OUTPUT_DIR,
    "postrelease": post_mod.OUTPUT_DIR,
    "stream": STREAM_OUTPUT_DIR,
}


class Profile(BaseModel):
    name: str
    drawWeight: str
    notes: Optional[str] = ""


class PipelineStep(BaseModel):
    id: str
    label: str
    status: str


class GhostReferencePayload(BaseModel):
    referenceId: str


class ChatMessage(BaseModel):
    role: str
    content: str


class CoachChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    runId: Optional[str] = None


@dataclass
class PipelineResult:
    original_video: Path
    skeleton_video: Path
    ghost_video: Path
    spine_image: Path
    draw_force_image: Path
    post_release_image: Path
    ghost_markdown: Optional[str]
    spine_markdown: Optional[str]
    draw_force_markdown: Optional[str]
    post_release_markdown: str
    report_path: Path


@dataclass
class PipelineJob:
    job_id: str
    video_id: str
    uploaded_path: Path
    steps: List[PipelineStep]
    use_st_prompt: bool
    done: bool = False
    error: Optional[str] = None
    result: Optional[PipelineResult] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, PipelineJob] = {}
        self._lock = threading.Lock()

    def create_job(self, video_id: str, uploaded_path: Path, use_st_prompt: bool) -> PipelineJob:
        job_id = uuid.uuid4().hex
        step_defs = [
            PipelineStep(id="upload", label="Uploading video", status="done"),
            PipelineStep(id="transcode", label="Transcoding to 60 fps", status="pending"),
            PipelineStep(id="remote-inference", label="Remote inference", status="active"),
            PipelineStep(id="phase-estimation", label="Phase estimation", status="pending"),
            PipelineStep(id="kalman", label="Kalman smoothing", status="pending"),
            PipelineStep(id="crop", label="Cropping & bannering", status="pending"),
            PipelineStep(id="skeleton", label="Skeleton overlay", status="pending"),
            PipelineStep(id="ghost", label="Ghost overlay analysis", status="pending"),
            PipelineStep(id="draw-force", label="Draw-force analysis", status="pending"),
            PipelineStep(id="post-release", label="Post-release analysis", status="pending"),
            PipelineStep(id="finalize", label="Finalizing report", status="pending"),
        ]
        job = PipelineJob(job_id=job_id, video_id=video_id, uploaded_path=uploaded_path, steps=step_defs, use_st_prompt=use_st_prompt)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        with self._lock:
            return self._jobs.get(job_id)


job_manager = JobManager()
app = FastAPI(title="STRAIGHT API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
)


def read_profile() -> Optional[Profile]:
    if not PROFILE_PATH.exists():
        return None
    lines = PROFILE_PATH.read_text(encoding="utf-8").splitlines()
    name: Optional[str] = None
    draw_weight: Optional[str] = None
    note_lines: List[str] = []
    capturing_notes = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if ":" in line and not capturing_notes:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "student name":
                name = value
                continue
            if key == "student draw weight":
                draw_weight = value
                continue
            if key == "additional notes":
                capturing_notes = True
                if value:
                    note_lines.append(value)
                continue
        if capturing_notes:
            note_lines.append(line.strip())

    if name and draw_weight:
        notes = "\n".join(note_lines).strip()
        return Profile(name=name, drawWeight=draw_weight, notes=notes)
    return None


def write_profile(profile: Profile) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    notes = (profile.notes or "").strip()
    content = (
        f"Student Name: {profile.name}\n"
        f"Student Draw Weight: {profile.drawWeight}\n"
        f"Additional Notes: {notes}\n"
    )
    PROFILE_PATH.write_text(content, encoding="utf-8")


def read_ghost_reference_id() -> str:
    if GHOST_REFERENCE_PATH.exists():
        try:
            text = GHOST_REFERENCE_PATH.read_text(encoding="utf-8").strip()
            if text:
                data = json.loads(text)
                ref = data.get("referenceId") or data.get("id")
                if isinstance(ref, str) and ref.strip():
                    return ref.strip()
            else:
                logger.warning("Ghost reference file is empty; resetting to default.")
        except Exception as exc:
            logger.warning("Failed to read ghost reference file: %s", exc)
    # Default and rewrite if missing/invalid.
    try:
        write_ghost_reference_id(DEFAULT_REFERENCE_ID)
    except Exception:
        pass
    return DEFAULT_REFERENCE_ID


def resolve_reference_pose(reference_id: str) -> Optional[Path]:
    """Locate a smoothed pose JSON for the chosen ghost reference, searching common variants."""
    candidates = [
        pipeline.SMOOTHED_DIR / f"results_{reference_id}_cropped.json",
        pipeline.SMOOTHED_DIR / f"results_{reference_id}.json",
        pipeline.SMOOTHED_DIR / "backup_data" / f"results_{reference_id}_cropped.json",
        pipeline.SMOOTHED_DIR / "backup_data" / f"results_{reference_id}.json",
        Path("runs") / reference_id / "poses" / "smoothed_cropped.json",
        Path("runs") / reference_id / "poses" / "smoothed.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_reference_video(reference_id: str) -> Optional[Path]:
    """Locate a reference video for the ghost overlay."""
    candidates = [
        pipeline.TRAINING_DIR / f"{reference_id}.mp4",
        pipeline.SAVED_VIDEOS_DIR / f"{reference_id}.mp4",
        Path("runs") / reference_id / "videos" / "cropped_bannered.mp4",
        Path("runs") / reference_id / "videos" / "cropped.mp4",
        Path("runs") / reference_id / "videos" / "source.mp4",
        Path("web/public/assets/originals") / f"{reference_id}_original.mp4",
    ]
    return next((p for p in candidates if p.exists()), None)


def write_ghost_reference_id(reference_id: str) -> str:
    ref = reference_id.strip()
    if not ref:
        raise ValueError("reference_id is empty")
    GHOST_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GHOST_REFERENCE_PATH.write_text(json.dumps({"referenceId": ref}), encoding="utf-8")
    return ref


def safe_read_text(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to read %s: %s", path, exc)
    return ""


def strip_markdown_images(markdown: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*]\([^)]+\)", "", markdown)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def remove_data_blocks(text: str) -> str:
    """Strip any DATA_START/DATA_END blocks from markdown text."""
    lines = text.splitlines()
    result: List[str] = []
    skipping = False
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().upper() == "DATA_START":
            skipping = True
            continue
        if line.strip().upper() == "DATA_END":
            skipping = False
            continue
        if not skipping:
            result.append(line)
    return "\n".join(result)


def load_notes_text() -> str:
    if not NOTES_DIR.exists():
        return ""
    parts: List[str] = []
    for note_path in sorted(NOTES_DIR.glob("*.txt")):
        text = safe_read_text(note_path)
        if text:
            parts.append(f"{note_path.stem}:\n{text}")
    return "\n\n".join(parts).strip()


def resolve_asset_path(asset_path: Optional[str]) -> Optional[Path]:
    if not asset_path:
        return None
    candidate = (ASSETS_BASE_DIR / asset_path).resolve()
    base = ASSETS_BASE_DIR.resolve()
    if not str(candidate).startswith(str(base)):
        return None
    return candidate


def _manifest_timestamp(run: Dict[str, object]) -> float:
    assets = run.get("assets") or {}
    candidates = [
        assets.get("report"),
        assets.get("original"),
        assets.get("ghost"),
        run.get("ghostMarkdown"),
        run.get("spineMarkdown"),
        run.get("drawForceMarkdown"),
        run.get("drawLengthMarkdown"),
        run.get("postReleaseMarkdown"),
    ]
    timestamps: List[float] = []
    for rel_path in candidates:
        path = resolve_asset_path(rel_path) if isinstance(rel_path, str) else None
        if path and path.exists():
            try:
                timestamps.append(path.stat().st_mtime)
            except OSError:
                continue
    return max(timestamps) if timestamps else -1.0


def extract_data_lines(text: str) -> List[str]:
    lines = text.splitlines()
    collected: List[str] = []
    capturing = False
    for raw in lines:
        line = raw.strip()
        if line.upper() == "DATA_START":
            capturing = True
            continue
        if line.upper() == "DATA_END":
            capturing = False
            continue
        if capturing and line:
            collected.append(line)
    return collected


def load_latest_run_reports(run_id: Optional[str] = None) -> tuple[str, Optional[str], str]:
    manifest_path = pipeline.WEB_ASSETS_RUNS_DIR / "index.json"
    if not manifest_path.exists():
        return "", None, ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to read runs manifest: %s", exc)
        return "", None, ""
    runs = manifest.get("runs") or []
    target_run: Optional[Dict[str, object]] = None
    if run_id:
        target_run = next(
            (
                r
                for r in runs
                if str(r.get("id") or "").strip() == run_id.strip()
                or str(r.get("label") or "").strip() == run_id.strip()
            ),
            None,
        )
        if target_run is None:
            logger.warning("Requested run %s not found in manifest.", run_id)
            return "", run_id, ""
    if target_run is None:
        latest_ts = -1.0
        for run in runs:
            ts = _manifest_timestamp(run)
            if ts > latest_ts:
                latest_ts = ts
                target_run = run
    if not target_run:
        return "", None, ""

    run_id = str(target_run.get("id") or target_run.get("label") or "").strip() or None
    assets = target_run.get("assets") or {}
    report_paths = [
        assets.get("report"),
        target_run.get("ghostMarkdown"),
        target_run.get("spineMarkdown"),
        target_run.get("drawForceMarkdown"),
        target_run.get("drawLengthMarkdown"),
        target_run.get("releaseMarkdown") or target_run.get("postReleaseMarkdown"),
        target_run.get("followThroughMarkdown") or target_run.get("postReleaseBowMarkdown"),
        target_run.get("postReleaseDrawMarkdown"),
        target_run.get("postReleaseBowMarkdown"),
    ]

    sections: List[str] = []
    data_sections: List[str] = []
    for rel_path in report_paths:
        path = resolve_asset_path(rel_path) if isinstance(rel_path, str) else None
        if not path or not path.exists():
            continue
        text = safe_read_text(path)
        if text:
            cleaned = strip_markdown_images(remove_data_blocks(text))
            if cleaned:
                sections.append(f"{path.stem}:\n{cleaned}")
            data_lines = extract_data_lines(text)
            if data_lines:
                data_sections.append(f"{path.stem}:\n" + "\n".join(data_lines))
    combined = "\n\n".join(sections).strip()
    data_preview = "\n\n".join(data_sections).strip()
    return combined, run_id, data_preview


def build_chat_prompt(question: str, context: str, notes: str, personal_info: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    template = safe_read_text(CHAT_PROMPT_PATH)
    if not template:
        raise RuntimeError("Chat prompt template missing or empty.")
    return (
        template.replace("{personal info}", personal_info.strip() or "Not provided.")
        .replace("{context}", context.strip() or "No additional context.")
        .replace("{notes}", notes.strip() or "No notes recorded.")
        .replace("{question}", question.strip())
    )


def format_history(history: List[ChatMessage], limit: int = 6) -> str:
    if not history:
        return ""
    recent = history[-limit:]
    lines = []
    for msg in recent:
        role = msg.role.lower()
        label = "Coach" if role.startswith("assistant") or role == "coach" else "Student"
        text = msg.content.strip()
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines).strip()


def extract_response_text(response) -> str:
    if hasattr(response, "output_text"):
        text = "".join(response.output_text)
        if text:
            return text
    output = getattr(response, "output", None)
    if output:
        chunks = []
        for item in output:
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    chunks.append(content.text)
        if chunks:
            return " ".join(chunks)
    return ""


def normalize_run_id(run_id: str) -> str:
    base = run_id.strip()
    if base.endswith(".json"):
        base = base[: -len(".json")]
    if base.startswith("results_"):
        base = base.split("results_", 1)[1]
    if base.endswith("_cropped"):
        base = base[: -len("_cropped")]
    # normalize accidental truncation like 00.00.43 -> 00.00.43.000
    parts = base.split("-")
    fixed_parts = []
    for part in parts:
        if part.count(".") == 2 and len(part.split(".")[-1]) < 3:
            # pad last segment to 3 digits
            base, last = part.rsplit(".", 1)
            fixed_parts.append(f"{base}.{last.zfill(3)}")
        elif part.count(".") == 3 and len(part.split(".")[-1]) < 3:
            segs = part.split(".")
            segs[-1] = segs[-1].zfill(3)
            fixed_parts.append(".".join(segs))
        else:
            fixed_parts.append(part)
    return "-".join(fixed_parts)


def read_excluded_runs() -> List[str]:
    if not EXCLUSIONS_PATH.exists():
        return []
    try:
        data = json.loads(EXCLUSIONS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read exclusions: %s", exc)
    return []


def write_excluded_runs(ids: List[str]) -> List[str]:
    cleaned = sorted({normalize_run_id(str(item).strip()) for item in ids if str(item).strip()})
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXCLUSIONS_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return cleaned


def set_step(job: PipelineJob, step_id: str, status: str) -> None:
    for step in job.steps:
        if step.id == step_id:
            step.status = status
            break


def format_post_release_markdown(payload: Dict[str, object]) -> str:
    lines = ["### Post-release follow-through"]
    release_frame = payload.get("release_frame")
    analyzed = payload.get("analyzed_frames")
    draw_ratio = payload.get("draw_ratio")
    draw_warning = payload.get("draw_warning")
    bow_drop = payload.get("bow_angle_drop")
    bow_warning = payload.get("bow_warning")
    if release_frame is not None:
        lines.append(f"- Release frame analyzed: {release_frame}")
    if analyzed is not None:
        lines.append(f"- Frames evaluated: {analyzed}")
    if draw_ratio is not None:
        pct = float(draw_ratio) * 100.0
        lines.append(f"- Draw hand behind head {pct:.1f}% of frames ({'⚠️' if draw_warning else '✅'})")
    if bow_drop is not None:
        lines.append(f"- Bow-arm angle shift: {float(bow_drop):.1f}° ({'⚠️' if bow_warning else '✅'})")
    return "\n".join(lines)


def save_upload(upload: UploadFile) -> Path:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRAINING_DIR / upload.filename
    with dest.open("wb") as fp:
        for chunk in iter(lambda: upload.file.read(1024 * 1024), b""):
            if not chunk:
                break
            fp.write(chunk)
    return dest


def pipeline_worker(job: PipelineJob) -> None:
    try:
        logger.info(
            "Pipeline job %s started for %s (shot trainer prompt: %s)",
            job.job_id,
            job.video_id,
            job.use_st_prompt,
        )
        set_step(job, "transcode", "active")
        training_video = transcode_to_training_video(job.video_id, job.uploaded_path, force=False)
        set_step(job, "transcode", "done")
        set_step(job, "remote-inference", "active")
        inference_json_path = pipeline.INFERENCE_DIR / f"results_{job.video_id}.json"
        rendered_video_path = pipeline.SAVED_VIDEOS_DIR / f"{job.video_id}.mp4"
        if inference_json_path.exists() and rendered_video_path.exists():
            logger.info(
                "Remote inference artifacts already exist for %s; skipping remote inference.", job.video_id
            )
            inference_json, remote_video = inference_json_path, rendered_video_path
        else:
            inference_json, remote_video = run_remote_inference(job.video_id, force=False, skip=False)
        set_step(job, "remote-inference", "done")
        set_step(job, "phase-estimation", "active")
        label_path = run_vlm_phase_estimation(job.video_id, job.use_st_prompt, enable_debug=False, force=False, skip=False)
        set_step(job, "phase-estimation", "done")
        set_step(job, "kalman", "active")
        smoothed_path = run_kalman_smoothing(job.video_id, force=False, skip=False)
        set_step(job, "kalman", "done")
        set_step(job, "crop", "active")
        processed_video, crop_offset = crop_and_banner_video(job.video_id, training_video, smoothed_path, label_path, force=False)
        window_start = 0
        clip_frame_count = 0
        total_frames = 0
        meta_path = processed_video.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                clip_frame_count = int(meta.get("clip_frame_count") or 0)
                total_frames = int(meta.get("source_frame_count") or 0)
            except Exception:
                pass
        try:
            import cv2

            cap = cv2.VideoCapture(str(processed_video))
            if cap.isOpened():
                total_frames = total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            pass
        clip_frame_count = clip_frame_count or total_frames
        pose_for_video = adjust_pose_for_crop(smoothed_path, crop_offset, window_start, clip_frame_count)
        set_step(job, "crop", "done")
        set_step(job, "skeleton", "active")
        logger.info("Generating skeleton overlay for %s", job.video_id)
        skeleton_overlay = run_skeleton_overlay(pose_for_video, processed_video, force=False)
        logger.info("Skeleton overlay saved → %s", skeleton_overlay)
        set_step(job, "skeleton", "done")
        reference_id = read_ghost_reference_id()
        reference_pose = resolve_reference_pose(reference_id)
        reference_video_override = resolve_reference_video(reference_id)
        if not reference_pose:
            logger.warning(
                "Reference pose for %s missing; using current run pose as ghost reference.", reference_id
            )
            reference_pose = pose_for_video
            reference_video_override = processed_video
        elif not reference_video_override:
            logger.warning(
                "Reference video for %s missing; falling back to current processed video.", reference_id
            )
            reference_video_override = processed_video
        set_step(job, "ghost", "active")
        ghost_video = ensure_ghost_overlay(
            pose_for_video,
            reference_pose,
            force=False,
            actual_video=processed_video,
            reference_video=reference_video_override,
            crop_offset=None,
            asset_video_id=job.video_id,
            report_video_id=smoothed_path.stem,
        )
        set_step(job, "ghost", "done")
        set_step(job, "draw-force", "active")
        spine_path, spine_markdown = run_spine_analysis(pose_for_video, processed_video, force=False)
        dfl_summary = run_draw_force_analysis(pose_for_video, processed_video, force=False)
        dfl_angle_report_path, dfl_length_report_path = write_dfl_reports(
            pose_for_video.stem,
            dfl_summary.geometry,
            dfl_summary.angle_stats,
            dfl_summary.length_stats,
            dfl_summary.image_path,
            dfl_summary.angle_count,
            dfl_summary.length_count,
        )
        draw_force_markdown = read_dfl_report(pose_for_video)
        set_step(job, "draw-force", "done")
        set_step(job, "post-release", "active")
        post_summary = run_post_release_analysis(pose_for_video, processed_video, force=False)
        post_draw_report_path, post_bow_report_path, post_report_path = write_post_release_reports(
            pose_for_video.stem, post_summary, post_summary.image_path
        )
        post_summary.draw_markdown_path = post_draw_report_path
        post_summary.bow_markdown_path = post_bow_report_path
        if post_report_path:
            post_summary.markdown_path = post_report_path
        post_report = read_post_release_report(pose_for_video)
        ghost_report_path = pipeline.GHOST_ANALYSIS_DIR / f"{smoothed_path.stem}.md"
        spine_report_path = None
        if spine_markdown:
            try:
                spine_report_path = pipeline.WEB_ASSETS_RUNS_DIR / job.video_id / "spine.md"
                spine_report_path.parent.mkdir(parents=True, exist_ok=True)
                spine_report_path.write_text(spine_markdown.strip(), encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to write spine markdown to assets: %s", exc)
        set_step(job, "post-release", "done")
        set_step(job, "finalize", "active")
        fps = detect_video_fps(processed_video)
        release_summary = compute_release_summary(smoothed_path, fps)
        ghost_markdown = read_ghost_report(smoothed_path)
        post_release_markdown = post_report or format_post_release_markdown(post_summary.payload)
        final_video_copy = copy_artifact(ghost_video, STREAM_OUTPUT_DIR / f"{job.video_id}_ghost.mp4")
        spine_copy = copy_artifact(spine_path, STREAM_OUTPUT_DIR / spine_path.name)
        dfl_copy = copy_artifact(dfl_summary.image_path, STREAM_OUTPUT_DIR / dfl_summary.image_path.name)
        post_copy = copy_artifact(post_summary.image_path, STREAM_OUTPUT_DIR / post_summary.image_path.name)
        report_path = STREAM_OUTPUT_DIR / f"{job.video_id}_analysis_report.md"
        # Mirror CLI report copies into stream_output
        if ghost_report_path and ghost_report_path.exists():
            copy_artifact(ghost_report_path, STREAM_OUTPUT_DIR / ghost_report_path.name)
        if dfl_angle_report_path and dfl_angle_report_path.exists():
            copy_artifact(dfl_angle_report_path, STREAM_OUTPUT_DIR / dfl_angle_report_path.name)
        if dfl_length_report_path and dfl_length_report_path.exists():
            copy_artifact(dfl_length_report_path, STREAM_OUTPUT_DIR / dfl_length_report_path.name)
        if post_summary.markdown_path and post_summary.markdown_path.exists():
            copy_artifact(post_summary.markdown_path, STREAM_OUTPUT_DIR / post_summary.markdown_path.name)
        if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists():
            copy_artifact(post_summary.draw_markdown_path, STREAM_OUTPUT_DIR / post_summary.draw_markdown_path.name)
        if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists():
            copy_artifact(post_summary.bow_markdown_path, STREAM_OUTPUT_DIR / post_summary.bow_markdown_path.name)
        if job.use_st_prompt:
            logger.info("✅ Shot trainer prompt was USED for job %s (%s)", job.job_id, job.video_id)
        export_run_assets(
            job.video_id,
            processed_video,
            final_video_copy,
            skeleton_overlay,
            spine_copy,
            dfl_copy,
            dfl_summary.sequence_paths,
            dfl_summary.draw_length_image,
            dfl_summary.draw_length_sequence,
            post_summary.release_image,
            post_summary.release_sequence,
            post_summary.follow_image,
            post_summary.follow_sequence,
            post_summary.image_path,
            report_path,
            ghost_report_path if ghost_report_path.exists() else None,
            spine_report_path if spine_report_path and spine_report_path.exists() else None,
            dfl_angle_report_path if dfl_angle_report_path and dfl_angle_report_path.exists() else None,
            dfl_length_report_path if dfl_length_report_path and dfl_length_report_path.exists() else None,
            post_summary.markdown_path if post_summary.markdown_path and post_summary.markdown_path.exists() else None,
            post_summary.draw_markdown_path if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists() else None,
            post_summary.bow_markdown_path if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists() else None,
        )
        meta_path = processed_video.with_suffix(".meta.json") if processed_video.with_suffix(".meta.json").exists() else None
        update_runs_manifest(
            job.video_id,
            label_path,
            meta_path,
            ghost_report_path if ghost_report_path.exists() else None,
            spine_report_path if spine_report_path and spine_report_path.exists() else None,
            dfl_angle_report_path if dfl_angle_report_path and dfl_angle_report_path.exists() else None,
            dfl_length_report_path if dfl_length_report_path and dfl_length_report_path.exists() else None,
            post_summary.markdown_path if post_summary.markdown_path and post_summary.markdown_path.exists() else None,
            post_summary.draw_markdown_path if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists() else None,
            post_summary.bow_markdown_path if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists() else None,
            dfl_summary.sequence_paths,
            dfl_summary.draw_length_sequence,
            post_summary.release_sequence,
            post_summary.follow_sequence,
            post_summary.release_image,
            post_summary.follow_image,
        )
        label_cropped = Path("estimated_labels") / f"vlm_estimated_label_{pose_for_video.stem.replace('results_', '')}.json"
        bundle_run_outputs(
            job.video_id,
            training_video,
            processed_video,
            inference_json,
            smoothed_path,
            pose_for_video,
            label_path,
            label_cropped,
            final_video_copy,
            skeleton_overlay,
            spine_copy,
            dfl_copy,
            dfl_summary.sequence_paths,
            dfl_summary.draw_length_image,
            dfl_summary.draw_length_sequence,
            post_summary.release_image,
            post_summary.release_sequence,
            post_summary.follow_image,
            post_summary.follow_sequence,
            post_copy,
            report_path,
            ghost_report_path if ghost_report_path.exists() else None,
            post_summary.markdown_path if post_summary.markdown_path and post_summary.markdown_path.exists() else None,
            post_summary.draw_markdown_path if post_summary.draw_markdown_path and post_summary.draw_markdown_path.exists() else None,
            post_summary.bow_markdown_path if post_summary.bow_markdown_path and post_summary.bow_markdown_path.exists() else None,
            [p for p in (dfl_angle_report_path, dfl_length_report_path) if p and p.exists()],
            meta_path,
        )
        write_report(
            report_path,
            job.video_id,
            processed_video,
            inference_json,
            remote_video,
            label_path,
            pose_for_video,
            DEFAULT_REFERENCE_ID,
            final_video_copy,
            release_summary,
            ghost_markdown,
            spine_copy,
            spine_markdown,
            dfl_summary,
            dfl_copy,
            dfl_summary.sequence_paths,
            draw_force_markdown,
            post_summary,
            post_copy,
            post_report,
        )
        job.result = PipelineResult(
            original_video=processed_video,
            skeleton_video=skeleton_overlay,
            ghost_video=ghost_video,
            spine_image=spine_path,
            draw_force_image=dfl_summary.image_path,
            post_release_image=post_summary.image_path,
            ghost_markdown=ghost_markdown,
            spine_markdown=spine_markdown,
            draw_force_markdown=draw_force_markdown,
            post_release_markdown=post_release_markdown,
            report_path=report_path,
        )
        set_step(job, "finalize", "done")
        job.done = True
    except Exception as exc:  # pragma: no cover - worker logs
        job.error = str(exc)
        job.done = True
        logger.exception("Pipeline job %s failed: %s", job.job_id, exc)


@app.get("/api/personal-profile", response_model=Profile)
async def api_get_profile() -> Profile:
    profile = read_profile()
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not configured")
    return profile


@app.post("/api/personal-profile", response_model=Profile)
async def api_set_profile(profile: Profile) -> Profile:
    write_profile(profile)
    return profile


@app.post("/api/upload")
async def api_upload(video: UploadFile = File(...), useStPrompt: bool = Form(False)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    saved_path = save_upload(video)
    video_id = Path(video.filename).stem.lower()
    job = job_manager.create_job(video_id, saved_path, use_st_prompt=useStPrompt)
    thread = threading.Thread(target=pipeline_worker, args=(job,), daemon=True)
    thread.start()
    return {"uploadId": job.job_id}


@app.get("/api/pipeline-status/{job_id}")
async def api_pipeline_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {
        "jobId": job.job_id,
        "steps": [step.model_dump() for step in job.steps],
        "done": job.done,
        "error": job.error,
    }


def build_media_url(kind: str, path: Path) -> str:
    directory = MEDIA_DIRECTORIES[kind]
    path = path.resolve()
    if not path.exists():
        logger.error("Media file missing: %s (%s)", path, kind)
        raise FileNotFoundError(path)
    logger.info("Serving media [%s]: %s", kind, path)
    return f"/media/{kind}/{path.name}"


@app.get("/api/jobs/{job_id}/results")
async def api_job_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    # Prefer assets exported to /web/public/assets/runs (matches CLI main)
    run_dir = pipeline.WEB_ASSETS_RUNS_DIR / job.video_id
    manifest_path = pipeline.WEB_ASSETS_RUNS_DIR / "index.json"
    if run_dir.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next((r for r in manifest.get("runs", []) if r.get("id") == job.video_id), None)
        except Exception:
            entry = None
        if entry:
            assets_entry = entry.get("assets") or {}
            def asset_path(key: str):
                val = assets_entry.get(key)
                return f"/assets/{val}" if val and not str(val).startswith("/assets/") else val

            def read_md(name: str):
                path = run_dir / name
                return path.read_text(encoding="utf-8") if path.exists() else None

            assets = {
                "original": asset_path("original"),
                "skeleton": asset_path("skeleton"),
                "ghost": asset_path("ghost"),
                "spineImage": asset_path("spineImage"),
                "drawForceImage": asset_path("drawForceImage"),
                "drawForceFrames": assets_entry.get("drawForceFrames"),
                "drawLengthImage": asset_path("drawLengthImage"),
                "drawLengthFrames": assets_entry.get("drawLengthFrames"),
                "releaseImage": asset_path("releaseImage"),
                "releaseFrames": assets_entry.get("releaseFrames"),
                "followThroughImage": asset_path("followThroughImage"),
                "followThroughFrames": assets_entry.get("followThroughFrames"),
                "postReleaseImage": asset_path("postReleaseImage"),
                "report": asset_path("report"),
            }
            markdown = {
                "ghost": read_md("ghost_overlay.md"),
                "spine": read_md("spine.md"),
                "drawForce": "\n\n".join(filter(None, [read_md("draw_force_angle.md"), read_md("draw_length.md")])),
                "release": read_md("post_release_draw.md"),
                "followThrough": read_md("post_release_bow.md"),
                "postRelease": read_md("post_release.md"),
            }
            return {"jobId": job.job_id, "videoId": job.video_id, "assets": assets, "markdown": markdown}

    if not job.result:
        raise HTTPException(status_code=409, detail="Job still running")

    # Fallback to live job result media endpoints
    result = job.result
    assets = {
        "original": build_media_url("cropped", result.original_video),
        "skeleton": build_media_url("skeleton", result.skeleton_video),
        "ghost": build_media_url("ghost", result.ghost_video),
        "spineImage": build_media_url("spine", result.spine_image),
        "drawForceImage": build_media_url("drawforce", result.draw_force_image),
        "postReleaseImage": build_media_url("postrelease", result.post_release_image),
        "report": build_media_url("stream", result.report_path),
    }
    markdown = {
        "ghost": result.ghost_markdown,
        "spine": result.spine_markdown,
        "drawForce": result.draw_force_markdown,
        "postRelease": result.post_release_markdown,
    }
    return {"jobId": job.job_id, "videoId": job.video_id, "assets": assets, "markdown": markdown}


@app.post("/api/coach-chat")
async def api_coach_chat(payload: CoachChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)
    notes_text = load_notes_text()
    personal_info = safe_read_text(PROFILE_PATH)
    history_text = format_history(payload.history)
    question = message if not history_text else f"{history_text}\n\nLatest question: {message}"

    rag_context = ""
    try:
        rag_context, _ = get_rag_context(message, client=client, api_key=api_key, top_k=4)
        rag_context = remove_data_blocks(rag_context).strip()
    except Exception as exc:  # pragma: no cover - RAG is best-effort
        logger.warning("Failed to fetch RAG context: %s", exc)
        rag_context = ""

    latest_reports, latest_run_id, data_preview = load_latest_run_reports(payload.runId)
    context_parts = []
    if latest_run_id or latest_reports:
        label = f"Latest run ({latest_run_id})" if latest_run_id else "Latest run"
        reports_block = latest_reports or "No reports found."
        context_parts.append(f"{label} reports:\n{reports_block}")
    if rag_context:
        context_parts.append(f"Retrieved references:\n{rag_context}")
    context_text = remove_data_blocks("\n\n".join(context_parts)).strip()

    try:
        prompt = build_chat_prompt(question, context_text, notes_text, personal_info)
    except Exception as exc:
        logger.error("Unable to build coach prompt: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to build coach prompt.")

    logger.info("=== Coach chat request ===\nUser prompt:\n%s\n=== End coach chat request ===", prompt)

    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        )
    except Exception as exc:
        logger.error("Coach chat request failed: %s", exc)
        raise HTTPException(status_code=500, detail="OpenAI request failed.")

    reply_text = extract_response_text(response).strip()
    if not reply_text:
        raise HTTPException(status_code=502, detail="Empty response from model.")
    preview_text = context_text if context_text else ""
    context_preview = preview_text if len(preview_text) <= 1200 else f"{preview_text[:1200]}…"
    return {"reply": reply_text, "latestRunId": latest_run_id, "contextPreview": context_preview}


@app.get("/api/run-exclusions")
async def api_get_run_exclusions():
    return {"excluded": read_excluded_runs()}


@app.post("/api/run-exclusions")
async def api_set_run_exclusions(payload: Dict[str, object]):
    raw_ids = payload.get("excluded")
    if not isinstance(raw_ids, list):
        raise HTTPException(status_code=400, detail="Expected 'excluded' to be a list.")
    cleaned = [str(item).strip() for item in raw_ids if str(item).strip()]
    saved = write_excluded_runs(cleaned)
    return {"excluded": saved}


@app.get("/api/ghost-reference")
async def api_get_ghost_reference():
    return {"referenceId": read_ghost_reference_id()}


@app.post("/api/ghost-reference")
async def api_set_ghost_reference(payload: GhostReferencePayload):
    try:
        ref = write_ghost_reference_id(payload.referenceId)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"referenceId": ref}


@app.post("/api/runs/{run_id}/reports/regenerate")
@app.post("/api/runs/{run_id}/missing-reports")
async def api_regenerate_reports(run_id: str):
    run_id = run_id.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="Missing run id")
    run_dir = Path("runs") / run_id
    pose_candidates = [
        pipeline.SMOOTHED_DIR / f"results_{run_id}_cropped.json",
        pipeline.SMOOTHED_DIR / f"results_{run_id}.json",
        pipeline.SMOOTHED_DIR / "backup_data" / f"results_{run_id}.json",
        run_dir / "poses" / "smoothed_cropped.json",
        run_dir / "poses" / "smoothed.json",
    ]
    pose_path = next((p for p in pose_candidates if p.exists()), None)
    if pose_path is None:
        raise HTTPException(status_code=404, detail=f"Pose JSON not found for run {run_id}")

    video_candidates = [
        run_dir / "videos" / "cropped_bannered.mp4",
        run_dir / "videos" / "cropped.mp4",
        run_dir / "videos" / "source.mp4",
        pipeline.TRAINING_DIR / f"{run_id}.mp4",
    ]
    video_path = next((p for p in video_candidates if p.exists()), None)
    if video_path is None:
        raise HTTPException(status_code=404, detail=f"Video not found for run {run_id}")

    fps = detect_video_fps(video_path)
    draw_frame, release_frame = ghost_mod.get_draw_release_frames(pose_path)
    draw_to_release_frames = (
        release_frame - draw_frame if draw_frame is not None and release_frame is not None else None
    )
    avg_release, std_release, release_count = ghost_mod.release_stats(pose_path)
    avg_window, std_window, window_count = ghost_mod.draw_release_window_stats(pose_path)

    ghost_mod.run_timing_analysis(
        video_id=run_id,
        video_path=video_path,
        release_frame=release_frame,
        draw_to_release_frames=draw_to_release_frames,
        fps=fps,
        avg_release_frame=avg_release,
        std_release_frame=std_release,
        dataset_count=release_count,
        avg_window_frames=avg_window,
        std_window_frames=std_window,
        window_count=window_count,
        asset_video_id=run_id,
    )

    spine_path, spine_markdown = run_spine_analysis(pose_path, video_path, force=True)
    dfl_summary = run_draw_force_analysis(pose_path, video_path, force=True)
    angle_report, length_report = write_dfl_reports(
        pose_path.stem,
        dfl_summary.geometry,
        dfl_summary.angle_stats,
        dfl_summary.length_stats,
        dfl_summary.image_path,
        dfl_summary.angle_count,
        dfl_summary.length_count,
    )
    post_summary = run_post_release_analysis(pose_path, video_path, force=True)
    post_draw_report_path, post_bow_report_path, post_report_path = write_post_release_reports(
        pose_path.stem, post_summary, post_summary.image_path
    )
    post_summary.draw_markdown_path = post_draw_report_path
    post_summary.bow_markdown_path = post_bow_report_path
    post_summary.markdown_path = post_report_path or post_summary.markdown_path

    asset_dir = pipeline.WEB_ASSETS_RUNS_DIR / run_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    warnings: List[str] = []

    try:
        copy_artifact(dfl_summary.image_path, asset_dir / "draw_force.png")
    except Exception as exc:
        warnings.append(f"draw_force.png: {exc}")
    if dfl_summary.draw_length_image:
        try:
            copy_artifact(dfl_summary.draw_length_image, asset_dir / "draw_length.png")
        except Exception as exc:
            warnings.append(f"draw_length.png: {exc}")

    if angle_report and angle_report.exists():
        try:
            copy_artifact(angle_report, asset_dir / "draw_force_angle.md")
            generated.append("draw_force_angle")
        except Exception as exc:
            warnings.append(f"draw_force_angle.md: {exc}")
    if length_report and length_report.exists():
        try:
            copy_artifact(length_report, asset_dir / "draw_length.md")
            generated.append("draw_length")
        except Exception as exc:
            warnings.append(f"draw_length.md: {exc}")

    ghost_report_path = ghost_mod.ANALYSIS_DIR / f"{run_id}.md"
    if ghost_report_path.exists():
        try:
            copy_artifact(ghost_report_path, asset_dir / "ghost_overlay.md")
            generated.append("ghost_timing")
        except Exception as exc:
            warnings.append(f"ghost_overlay.md: {exc}")

    spine_report_path = None
    if spine_markdown:
        try:
            spine_report_path = pipeline.WEB_ASSETS_RUNS_DIR / run_id / "spine.md"
            spine_report_path.parent.mkdir(parents=True, exist_ok=True)
            spine_report_path.write_text(spine_markdown.strip(), encoding="utf-8")
            generated.append("spine")
        except Exception as exc:
            warnings.append(f"spine.md: {exc}")

    if post_summary.image_path and post_summary.image_path.exists():
        try:
            copy_artifact(post_summary.image_path, asset_dir / "post_release.png")
            generated.append("post_release_image")
        except Exception as exc:
            warnings.append(f"post_release.png: {exc}")
    if post_summary.release_image and post_summary.release_image.exists():
        try:
            copy_artifact(post_summary.release_image, asset_dir / "release.png")
        except Exception as exc:
            warnings.append(f"release.png: {exc}")
    if post_summary.follow_image and post_summary.follow_image.exists():
        try:
            copy_artifact(post_summary.follow_image, asset_dir / "follow_through.png")
        except Exception as exc:
            warnings.append(f"follow_through.png: {exc}")

    if post_report_path and post_report_path.exists():
        try:
            copy_artifact(post_report_path, asset_dir / "post_release.md")
            generated.append("post_release_report")
        except Exception as exc:
            warnings.append(f"post_release.md: {exc}")
    if post_draw_report_path and post_draw_report_path.exists():
        try:
            copy_artifact(post_draw_report_path, asset_dir / "post_release_draw.md")
        except Exception as exc:
            warnings.append(f"post_release_draw.md: {exc}")
    if post_bow_report_path and post_bow_report_path.exists():
        try:
            copy_artifact(post_bow_report_path, asset_dir / "post_release_bow.md")
        except Exception as exc:
            warnings.append(f"post_release_bow.md: {exc}")

    skeleton_video = next(
        (
            p
            for p in [
                run_dir / "videos" / "skeleton.mp4",
                pipeline.SAVED_VIDEOS_DIR / f"{run_id}.mp4",
                video_path,
            ]
            if p.exists()
        ),
        video_path,
    )
    ghost_video = next(
        (
            p
            for p in [
                run_dir / "videos" / "ghost.mp4",
                pipeline.GHOST_ANALYSIS_DIR / f"{run_id}.mp4",
                video_path,
            ]
            if p.exists()
        ),
        video_path,
    )

    try:
        export_run_assets(
            run_id,
            video_path,
            ghost_video,
            skeleton_video,
            spine_path,
            dfl_summary.image_path,
            dfl_summary.sequence_paths,
            dfl_summary.draw_length_image,
            dfl_summary.draw_length_sequence,
            post_summary.release_image,
            post_summary.release_sequence,
            post_summary.follow_image,
            post_summary.follow_sequence,
            post_summary.image_path,
            post_report_path or (asset_dir / "report.md"),
            ghost_report_path if ghost_report_path.exists() else None,
            spine_report_path if spine_report_path and spine_report_path.exists() else None,
            angle_report if angle_report and angle_report.exists() else None,
            length_report if length_report and length_report.exists() else None,
            post_report_path if post_report_path and post_report_path.exists() else None,
            post_draw_report_path if post_draw_report_path and post_draw_report_path.exists() else None,
            post_bow_report_path if post_bow_report_path and post_bow_report_path.exists() else None,
        )
    except Exception as exc:
        warnings.append(f"export_run_assets: {exc}")

    label_path = next(
        (
            p
            for p in [
                Path("estimated_labels") / f"vlm_estimated_label_{run_id}.json",
                run_dir / "labels" / f"vlm_estimated_label_{run_id}.json",
            ]
            if p.exists()
        ),
        None,
    )
    meta_path = next(
        (
            p
            for p in [
                run_dir / "meta" / f"{run_id}_original.meta.json",
                pipeline.WEB_ASSETS_RUNS_DIR / run_id / f"{run_id}_original.meta.json",
            ]
            if p.exists()
        ),
        None,
    )

    update_runs_manifest(
        run_id,
        label_path or pose_path,
        meta_path,
        ghost_report_path if ghost_report_path.exists() else None,
        spine_report_path if spine_report_path and spine_report_path.exists() else None,
        angle_report if angle_report and angle_report.exists() else None,
        length_report if length_report and length_report.exists() else None,
        post_report_path if post_report_path and post_report_path.exists() else None,
        post_draw_report_path if post_draw_report_path and post_draw_report_path.exists() else None,
        post_bow_report_path if post_bow_report_path and post_bow_report_path.exists() else None,
        dfl_summary.sequence_paths,
        dfl_summary.draw_length_sequence,
        post_summary.release_sequence,
        post_summary.follow_sequence,
        post_summary.release_image,
        post_summary.follow_image,
    )

    return {"runId": run_id, "generated": generated, "warnings": warnings}


@app.get("/media/{kind}/{filename}")
async def serve_media(kind: str, filename: str):
    directory = MEDIA_DIRECTORIES.get(kind)
    if not directory:
        raise HTTPException(status_code=404, detail="Unknown media category")
    path = (directory / filename).resolve()
    if not path.exists() or not str(path).startswith(str(directory.resolve())):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


if WEB_DIST := Path("web/dist"):
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="ui")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("straight-api")
