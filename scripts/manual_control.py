from mujoco.glfw import glfw

from render import Render

MODEL_PATH = "arx_15/scene.xml"

# ============================================================
# Controller
# ============================================================

STEP = 0.01

pressed_keys = set()


def update_control(data, key_set):

    if glfw.KEY_KP_1 in key_set:
        data.ctrl[0] += STEP

    if glfw.KEY_KP_2 in key_set:
        data.ctrl[0] -= STEP

    if glfw.KEY_KP_3 in key_set:
        data.ctrl[1] += STEP

    if glfw.KEY_KP_4 in key_set:
        data.ctrl[1] -= STEP

    if glfw.KEY_KP_5 in key_set:
        data.ctrl[2] += STEP

    if glfw.KEY_KP_6 in key_set:
        data.ctrl[2] -= STEP

    if glfw.KEY_KP_7 in key_set:
        data.ctrl[3] += STEP

    if glfw.KEY_KP_8 in key_set:
        data.ctrl[3] -= STEP

    if glfw.KEY_KP_9 in key_set:
        data.ctrl[4] += STEP

    if glfw.KEY_KP_0 in key_set:
        data.ctrl[4] -= STEP

    if glfw.KEY_KP_SUBTRACT in key_set:
        data.ctrl[5] += STEP

    if glfw.KEY_KP_ADD in key_set:
        data.ctrl[5] -= STEP


# ============================================================
# Renderer
# ============================================================

renderer = Render(
    model_path=MODEL_PATH,
    width=1200,
    height=900,
    title="ARX-15"
)

# ============================================================
# Camera initial position
# ============================================================

renderer.set_camera(
    azimuth=145.0,
    elevation=-33.0,
    distance=2.2,
    lookat=[
        0.0,
        0.3,
        0.0
    ]
)




# ============================================================
# Run
# ============================================================

renderer.run(update=update_control)


renderer.close()
