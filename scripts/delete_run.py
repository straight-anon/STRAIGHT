#!/usr/bin/env python3
"""
Purge all artifacts for a given run id (videos, labels, analyses, assets, manifest).

Usage:
  python scripts/delete_run.py <run_id> [--dry-run]

Run from the repo root. Use --dry-run to see what would be removed without deleting.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Set

ROOT = Path(__file__).resolve().parent.parent
RUNS_MANIFEST = ROOT / "web/public/assets/runs/index.json"


def list_runs_from_manifest() -> list[str]:
    if not RUNS_MANIFEST.exists():
        return []
    try:
        data = json.loads(RUNS_MANIFEST.read_text(encoding="utf-8"))
        return [str(run.get("id")) for run in data.get("runs", []) if run.get("id")]
    except Exception:
        return []


def prompt_for_run(choices: list[str]) -> str:
    if not choices:
        return ""
    print("Available runs:")
    for idx, run_id in enumerate(choices, start=1):
        print(f"  [{idx}] {run_id}")
    choice = input("Enter the number to delete (or run id): ").strip()
    if not choice:
        return ""
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    return choice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete all artifacts for a run id.")
    parser.add_argument("run_id", nargs="?", help="Run identifier to delete (e.g., 00.00.40.000-00.00.49.000)")
    parser.add_argument("--dry-run", action="store_true", help="List files without deleting.")
    return parser.parse_args()


def collect_paths(run_id: str) -> Set[Path]:
    """Expand a set of glob patterns that should be removed for the run."""
    patterns = [
        f"runs/{run_id}",
        f"runs/{run_id}/**/*",
        f"training_videos/{run_id}.mp4",
        f"saved_videos/{run_id}.mp4",
        f"inference_data/results_{run_id}.json",
        f"smoothed_inference_data/results_{run_id}.json",
        f"smoothed_inference_data/results_{run_id}_cropped.json",
        f"smoothed_inference_data/backup_data/results_{run_id}.json",
        f"estimated_labels/vlm_estimated_label_{run_id}.json",
        f"estimated_labels/vlm_estimated_label_{run_id}_cropped.json",
        f"ghost_overlay_analysis/results_{run_id}*.md",
        f"spine_straight_analysis/results_{run_id}*.md",
        f"draw_force_angle_analysis/results_{run_id}*.md",
        f"draw_length_analysis/results_{run_id}*.md",
        f"post_release_analysis_reports/results_{run_id}*.md",
        f"post_release_analysis_reports/results_{run_id}*.bow.md",
        f"post_release_analysis_reports/results_{run_id}*.draw.md",
        f"post_release_visualizations/results_{run_id}*",
        f"draw_force_visualizations/results_{run_id}*",
        f"skeleton_visualizations/results_{run_id}*",
        f"ghost_visualizations/{run_id}*",
        f"static_left_wrist_graphs/{run_id}*",
        f"stream_output/{run_id}*",
        f"debug_vlm_calls/{run_id}",
        f"debug_vlm_calls/{run_id}/**/*",
        f"web/public/assets/runs/{run_id}",
        f"web/public/assets/runs/{run_id}/**/*",
        f"web/public/assets/originals/{run_id}_original.*",
        f"web/public/assets/originals/{run_id}.mp4",
        f"web/public/assets/originals/{run_id}.json",
        f"web/public/assets/originals/{run_id}.png",
        f"web/public/assets/{run_id}*",
    ]
    found: Set[Path] = set()
    for pattern in patterns:
        found.update(ROOT.glob(pattern))
    # Filter out non-existent and dedupe.
    return {p for p in found if p.exists()}


def delete_paths(paths: Iterable[Path], dry_run: bool) -> None:
    # Delete files before directories (deepest first).
    for path in sorted(paths, key=lambda p: (p.is_dir(), len(p.as_posix().split("/"))), reverse=True):
        rel = path.relative_to(ROOT)
        if dry_run:
            print(f"[dry-run] remove {rel}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory {rel}")
            else:
                path.unlink()
                print(f"Removed file {rel}")
        except Exception as exc:
            print(f"⚠️  Failed to remove {rel}: {exc}")


def update_runs_manifest(run_id: str, dry_run: bool) -> None:
    if not RUNS_MANIFEST.exists():
        return
    try:
        data = json.loads(RUNS_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠️  Unable to read {RUNS_MANIFEST}: {exc}")
        return
    runs = data.get("runs", [])
    filtered = [r for r in runs if str(r.get("id")) != run_id]
    if len(filtered) == len(runs):
        return
    if dry_run:
        print(f"[dry-run] would update runs manifest to remove {run_id}")
        return
    data["runs"] = filtered
    RUNS_MANIFEST.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated manifest {RUNS_MANIFEST.relative_to(ROOT)} (removed {run_id})")


def main() -> None:
    args = parse_args()
    manifest_runs = list_runs_from_manifest()

    selected_run = args.run_id.strip() if args.run_id else ""
    if not selected_run:
        selected_run = prompt_for_run(manifest_runs)
    if not selected_run:
        raise SystemExit("Run id is required.")

    paths = collect_paths(selected_run)
    if not paths:
        print(f"No artifacts found for run id '{selected_run}'.")
    else:
        delete_paths(paths, args.dry_run)
    update_runs_manifest(selected_run, args.dry_run)
    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to delete.")


if __name__ == "__main__":
    main()
