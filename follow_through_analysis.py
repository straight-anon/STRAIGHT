import json
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys

# ----------------------
# CONFIG
# ----------------------
POSE_DIR = "inference_data"
LABEL_DIR = "estimated_labels"
OUTPUT_PLOT_DIR = "static_left_wrist_graphs"
OUTPUT_CSV_DIR = "left_wrist_post_release_csv"
CLASSIFICATION_DIR = "shot_follow_through_classification"
SMOOTH_WINDOW = 10

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)

SUMMARY_JSON_PATH = os.path.join(CLASSIFICATION_DIR, "shot_follow_through_summary.json")

# -----------------------------------------------------------
# LOAD LEFT WRIST COORDINATES
# -----------------------------------------------------------
def load_left_wrist(video_id):
    pose_json = os.path.join(POSE_DIR, f"results_{video_id}.json")
    if not os.path.exists(pose_json):
        raise FileNotFoundError(f"MMPose JSON not found: {pose_json}")

    with open(pose_json, "r") as f:
        data = json.load(f)

    fps = data.get("fps", 60.0)
    frames = data.get("instance_info", [])

    xs, ys = [], []
    for frame in frames:
        instances = frame.get("instances", [])
        if not instances:
            xs.append(0.0); ys.append(0.0); continue

        keypoints = instances[0]["keypoints"]
        xs.append(keypoints[9][0])   # LEFT wrist x
        ys.append(keypoints[9][1])   # LEFT wrist y

    return np.array(xs, float), np.array(ys, float), fps

# -----------------------------------------------------------
# LOAD PHASES
# -----------------------------------------------------------
def load_phases(path, fps):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label JSON not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    phases = []
    for p in data.get("phases", []):
        if "start_frame" not in p or "end_frame" not in p:
            continue
        name = p["name"]
        start_sec = p["start_frame"] / fps
        end_sec = p["end_frame"] / fps
        phases.append((name, start_sec, end_sec))
    return phases

# -----------------------------------------------------------
# SMOOTHING
# -----------------------------------------------------------
def moving_average(signal, window):
    if window < 2:
        return signal
    return np.convolve(signal, np.ones(window)/window, mode="same")

# -----------------------------------------------------------
# KINEMATICS
# -----------------------------------------------------------
def compute_derivatives(xs, ys, fps):
    dt = 1.0 / fps

    vx = np.gradient(xs, dt)
    vy = np.gradient(ys, dt)
    v = np.sqrt(vx**2 + vy**2)

    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    a = np.sqrt(ax**2 + ay**2)

    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    j = np.sqrt(jx**2 + jy**2)

    v = moving_average(v, SMOOTH_WINDOW)
    a = moving_average(a, SMOOTH_WINDOW)
    j = moving_average(j, SMOOTH_WINDOW)

    return v, a, j

# -----------------------------------------------------------
# PLOT POST-RELEASE
# -----------------------------------------------------------
def plot_post_release(t, v, a, j, release_sec, video_id, global_max):
    v_max, a_max, j_max = global_max

    fig, axes = plt.subplots(3,1, figsize=(14,10), sharex=True)

    axes[0].plot(t, v, color="blue")
    axes[1].plot(t, a, color="green")
    axes[2].plot(t, j, color="purple")

    for ax, ymax, label in zip(axes, [v_max, a_max, j_max], ["Velocity", "Acceleration", "Jerk"]):
        ax.set_ylim(0, ymax)
        ax.axvline(release_sec, color="red", linestyle="--", label="release")
        ax.set_ylabel(label)
        ax.grid(True)

    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_PLOT_DIR, f"{video_id}_post_release.png")
    plt.savefig(outpath)
    plt.close()
    print(f"📈 Saved graph: {outpath}")

# -----------------------------------------------------------
# SAVE CSV
# -----------------------------------------------------------
def save_post_release_csv(video_id, t, xs, ys, v, a, j):
    csv_path = os.path.join(OUTPUT_CSV_DIR, f"{video_id}_post_release.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_sec", "x", "y", "velocity", "acceleration", "jerk"])
        for i in range(len(t)):
            writer.writerow([i, t[i], xs[i], ys[i], v[i], a[i], j[i]])
    print(f"✅ CSV saved (post-release): {csv_path}")

# -----------------------------------------------------------
# CLASSIFY FOLLOW-THROUGH
# -----------------------------------------------------------
def classify(video_id, peak_velocity, summary_dict):
    if peak_velocity < 250:
        label = "GOOD"
    elif 250 <= peak_velocity <= 600:
        label = "MEDIUM"
    else:
        label = "BAD"
    summary_dict[video_id] = {
        "peak_post_release_velocity": round(peak_velocity, 2),
        "follow_through_classification": label
    }
    print(f"🏹 Classified {video_id}: {label}")

# -----------------------------------------------------------
# PROCESS SINGLE VIDEO
# -----------------------------------------------------------
def process_video(video_id, global_max, summary_dict):
    label_json = os.path.join(LABEL_DIR, f"vlm_estimated_label_{video_id}.json")
    xs, ys, fps = load_left_wrist(video_id)
    phases = load_phases(label_json, fps)

    release_phase = [p for p in phases if p[0] == "release"]
    if not release_phase:
        print("❌ No release phase — skipping.")
        return

    release_sec = release_phase[0][1]

    v, a, j = compute_derivatives(xs, ys, fps)
    t = np.arange(len(xs)) / fps

    post_mask = t >= release_sec
    xs = xs[post_mask]; ys = ys[post_mask]
    v = v[post_mask]; a = a[post_mask]; j = j[post_mask]
    t = t[post_mask]

    if len(t) < 5:
        print("❌ Too few post-release frames.")
        return

    save_post_release_csv(video_id, t, xs, ys, v, a, j)
    # plot_post_release(t, v, a, j, release_sec, video_id, global_max) #OPTION TO GRAPH FOR TUNING THRESHOLDS
    peak_v = np.max(v)
    classify(video_id, peak_v, summary_dict)

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == "__main__":

    summary_dict = {}

    # Determine if manual or batch mode
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        video_list = [video_id]
        print(f"\n🔹 Manual mode → processing only {video_id}")
    else:
        video_list = [jf.replace("vlm_estimated_label_", "").replace(".json","")
                      for jf in os.listdir(LABEL_DIR) if jf.startswith("vlm_estimated_label_")]
        print(f"\n🔹 Batch mode → processing {len(video_list)} videos")

    # FIRST PASS → compute global max for scaling
    all_v, all_a, all_j = [], [], []
    for video_id in video_list:
        try:
            xs, ys, fps = load_left_wrist(video_id)
            v, a, j = compute_derivatives(xs, ys, fps)
            all_v.append(np.max(v))
            all_a.append(np.max(a))
            all_j.append(np.max(j))
        except:
            continue

    global_max = (np.max(all_v), np.max(all_a), np.max(all_j))
    print(f"📏 Global max ranges → Velocity={global_max[0]:.1f}, Acc={global_max[1]:.1f}, Jerk={global_max[2]:.1f}")

    # SECOND PASS → process each video
    for video_id in video_list:
        print(f"\n🎬 Processing {video_id}...")
        try:
            process_video(video_id, global_max, summary_dict)
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")

    # Save summary JSON
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_dict, f, indent=4)
    print(f"\n✅ Saved summary JSON: {SUMMARY_JSON_PATH}")