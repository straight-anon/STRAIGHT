# STRAIGHT Quick Ops

## Pipeline
- `python3 video_analysis_pipeline.py --input-video training_videos/<clip>.mp4 --video-id <slug> [--use-st-prompt]`  
  Runs remote inference → VLM labels → smoothing → ghost overlay → spine/draw-force/post-release coaching and writes results to `stream_output/`.

## Remote Inference
- `python3 run_remote_inference.py --video training_videos/<clip>.mp4`  
  Uploads the clip to the remote MMPose box, renders the overlay, and downloads `saved_videos/<clip>.mp4` + `inference_data/results_<clip>.json`.

## VLM Phase Estimation
- `python3 vlm_phase_estimation.py --video-id <slug> [--use-st-prompt] [--enable-debug]`  
  Reads `training_videos/<slug>.mp4` + `inference_data/results_<slug>.json`, queries the VLM with prompts from `config/prompts`, and writes `estimated_labels/vlm_estimated_label_<slug>.json`. Debug artifacts go to `debug_vlm_calls/`.

## Kalman Smoothing
- `python3 kalman_smoothing.py --pose inference_data/results_<slug>.json [--output smoothed_inference_data/results_<slug>.json]`  
  Applies the constant-velocity smoother, trims to the shot window, and saves under `smoothed_inference_data/`.

## Ghost Overlay
- `python3 ghost_overlay.py --actual-pose smoothed_inference_data/results_<slug>.json --reference-pose smoothed_inference_data/results_<ref>.json --actual-video training_videos/<slug>.mp4 --reference-video training_videos/<ref>.mp4`  
  Produces `ghost_visualizations/<slug>_ghost.mp4` and a timing report in `ghost_overlay_analysis/`. Prompts live in `config/prompts/ghost_timing.md`; notes in `config/notes/ghost_overlay_notes.txt`.

## Spine Straightness
- `python3 spine_straight.py --pose smoothed_inference_data/results_<slug>.json --video training_videos/<slug>.mp4`  
  Saves `spine_visualizations/<slug>_spine.png` plus Markdown coaching under `spine_straight_analysis/`. Uses prompt `config/prompts/spine.md` and appends notes to `config/notes/spine_straight_notes.txt`.

## Draw-Force Line
- `python3 draw_force_line.py --pose smoothed_inference_data/results_<slug>.json --video training_videos/<slug>.mp4`  
  Writes the annotated PNG to `draw_force_visualizations/` and two GPT reports: `draw_force_angle_analysis/<slug>.md` (angle) and `draw_length_analysis/<slug>.md` (length). Prompts live at `config/prompts/draw_force_angle.md` and `config/prompts/draw_force_length.md`. Notes accumulate in `config/notes/draw_force_notes.txt`.

## Post-Release Analysis
- `python3 post_release_analysis.py --pose smoothed_inference_data/results_<slug>.json --video training_videos/<slug>.mp4`  
  Exports `post_release_visualizations/<slug>_post.png` plus optional JSON metrics (via `--json-report`). Helps quantify draw-hand follow-through and bow-arm stability.

## Config Overview
- Prompts: `config/prompts/*.md`  
- Notes / persistent memory: `config/notes/*.txt`  
- Student profile: `config/personal_info.txt`
- API server: `uvicorn api_server:app --reload` (serves `/api` + built React UI)

All scripts accept absolute or relative paths; pass `--help` for the full option list.***
