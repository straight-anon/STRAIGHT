## Recreating the Release Frame Detection Experiment

This document describes how to reproduce the release frame detection experiment reported in the paper.

### Model Configuration
- Vision–Language Model (VLM): `gemini-2.5-flash`
- Decoding: greedy (temperature = 0)
- Prompts: fixed yes/no prompts as defined in the paper appendix

### Dataset
The released dataset includes self-recorded bow shooting videos only. Competition footage and release-trainer videos are excluded due to copyright and participant consent restrictions.

#### Download
```python
from datasets import load_dataset
# will be released upon acceptance
````

### Directory Setup

Save all downloaded video files under:

```
release_frame_experiment/ourvid/
```

### Running Release Detection

Execute the following command to estimate release frames for all videos in the directory:

```bash
python3 poc_vlm_release_detection.py \
  --video-dir release_frame_experiment/ourvid \
  --output-dir estimated_labels
```

### Verification Protocol

For each processed video, the script displays a two-frame composite showing the predicted transition from draw to release using consecutive frames.

* Press `y` to accept the prediction if the transition visually corresponds to release.
* Predictions with minor motion blur in the release frame may be accepted if the transition is unambiguous.
* Press `n` to reject the prediction and flag the video for further review.

### Outputs

* Release estimates are saved as JSON files at:

  ```
  estimated_labels/<video-dir-name>/vlm_release_estimate_<video>.json
  ```
* Cropped regions used for VLM queries are saved under:

  ```
  debug_vlm_calls/<video_id>/
  ```
* User confirmations for per-frame review are logged in:

  ```
  debug_vlm_calls/vlm_user_feedback.jsonl
  ```
* Composite rejections and per-frame confirmation results are recorded in:

  ```
  debug_vlm_calls/composite_review_flags.jsonl
  ```

### Options and Flags

* `--use-release-trainer`: switches to the release-trainer prompt, where a “yes” response indicates draw and a “no” response indicates release.
* `--per-frame-confirm`: disables composite verification and forces per-frame confirmation for all queried frames.

### Dependencies

The following dependencies are required:

* `ffmpeg`
* `opencv-python`
* `google-generativeai`
* A valid `GOOGLE_API_KEY`
* Face detection model automatically downloaded by `face.py` at run

Ensure all dependencies are installed and environment variables are set before running the experiment.