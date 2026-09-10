import os
from render_collect import WristRender
import time
import cv2
import json
import numpy as np
import mujoco as mj

# Dataset
DATASET_DIR = "dataset"

# Record image/action at 10 Hz
RECORD_HZ = 10.0
RECORD_INTERVAL = 1.0 / RECORD_HZ

MODEL_PATH = "arx_15/scene.xml"
WRIST_CAMERA_NAME = "wrist_cam"

# Current episode
# 0~19  green
# 20~39 blue
EPISODE_ID = 39
INSTRUCTION = "pick up the cube on blue platform"

renderer = WristRender(
    model_path=MODEL_PATH,
    width=224,
    height=224,
    title="ARX-15 - IK",
    record_interval=RECORD_INTERVAL
)
# Print controls
while True:
    print()
    print("=" * 70)
    print("                    MANUAL IK CONTROL")
    print("=" * 70)

    print()
    print("TRANSLATION")
    print()
    print("  up         Forward")
    print("  down       Backward")
    print("  left       Left")
    print("  right      Right")
    print("  W          Up")
    print("  S          Down")
    print("  space      Gribber Open/Close")
    print("  Q          Switch camera")

    break



episode_dir = f"{DATASET_DIR}/episode_{EPISODE_ID:04d}"
img_dir = f"{episode_dir}/images"
os.makedirs(img_dir, exist_ok=True)
frame_id = 0
last_record_time = time.perf_counter()



samples = []
def collect_sample(ctrl, img):
    global frame_id
    actions = ctrl.tolist()
    # --------------------------------------------------------
    # Store in RAM
    # --------------------------------------------------------

    sample = {
        "image": img,
        "instruction": INSTRUCTION,
        "action": actions,
    }

    samples.append(sample)

    print(f"[DATA] frame={frame_id:06d}, action={actions}")

    frame_id += 1

# =================== Dataset directory ==========================
try:
    renderer.run(callback=collect_sample)

finally:
    # --------------------------------------------------------
    # Save entire episode
    # --------------------------------------------------------
    json_path = os.path.join( episode_dir, "episode.json")
    json_samples = []
    for i, sample in enumerate(samples):

        # Save image
        img = sample["image"]

        img_path = os.path.join(img_dir, f"frame_{i:06d}.png")

        # MuJoCo image is RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(img_path, img_bgr)

        # Save metadata
        json_samples.append({
            "image": f"images/frame_{i:06d}.png",
            "instruction": sample["instruction"],
            "action": sample["action"],
        })

    # Save JSON
    json_path = os.path.join(episode_dir, "episode.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            json_samples,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("Episode finished")
    print(f"Samples: {len(samples)}")
    print(f"Saved: {json_path}")
    print("=" * 70)
renderer.close()
