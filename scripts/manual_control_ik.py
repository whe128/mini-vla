import os
from render import Render

# Dataset
DATASET_DIR = "dataset"



# Current episode
EPISODE_ID = 0

MODEL_PATH = "arx_15/scene.xml"



render = Render(
    model_path=MODEL_PATH,
    width=1200,
    height=900,
    title="ARX-15 - IK"
)

render.set_camera(
    azimuth=145.0,
    elevation=-33.0,
    distance=2.2,
    lookat=[
        0.0,
        0.3,
        0.0
    ]
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



render.run()

render.close()
