import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "../arx_15/scene.xml"

STEP = 0.01


# ============================================================
# Load model
# ============================================================

dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, MODEL_PATH)

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Load initial state
mj.mj_resetDataKeyframe( model, data, model.key("scene_home").id)


#============================================================
# Mouse state
#============================================================
button_left = False
button_middle = False
button_right = False

lastx = 0
lasty = 0

def mouse_button(window, button, action, mods):

    global button_left
    global button_middle
    global button_right

    button_left = (
        glfw.get_mouse_button(
            window,
            glfw.MOUSE_BUTTON_LEFT
        ) == glfw.PRESS
    )

    button_middle = (
        glfw.get_mouse_button(
            window,
            glfw.MOUSE_BUTTON_MIDDLE
        ) == glfw.PRESS
    )

    button_right = (
        glfw.get_mouse_button(
            window,
            glfw.MOUSE_BUTTON_RIGHT
        ) == glfw.PRESS
    )

def mouse_move(window, xpos, ypos):

    global lastx
    global lasty

    dx = xpos - lastx
    dy = ypos - lasty

    lastx = xpos
    lasty = ypos

    # no button pressed
    if not button_left and not button_middle and not button_right:
        return

    width, height = glfw.get_window_size(window)

    # Shift
    left_shift = (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT)
        == glfw.PRESS
    )

    right_shift = (
        glfw.get_key(window, glfw.KEY_RIGHT_SHIFT)
        == glfw.PRESS
    )

    mod_shift = left_shift or right_shift

    # right button
    if button_right:

        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V

    # left button
    elif button_left:

        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V

    # middle button
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(
        model,
        action,
        dx / height,
        dy / height,
        cam
    )


def scroll(window, xoffset, yoffset):

    action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(
        model,
        action,
        0.0,
        -0.05 * yoffset,
        cam
    )

# ============================================================
# Keyboard state
# ============================================================

pressed_keys = set()


def keyboard(window, key, scancode, action, mods):
    # --------------------------------------------------------
    # Key press
    # --------------------------------------------------------
    if action == glfw.PRESS:

        pressed_keys.add(key)

        # Q: switch camera
        if key == glfw.KEY_Q:
            switch_camera()

    # Key pressed
    if action == glfw.PRESS or action == glfw.REPEAT:
        pressed_keys.add(key)

    # Key released
    elif action == glfw.RELEASE:
        pressed_keys.discard(key)


# ============================================================
# Controller
# ============================================================

def update_control():

    # Joint 1
    if glfw.KEY_KP_1 in pressed_keys:
        data.ctrl[0] += STEP

    if glfw.KEY_KP_2 in pressed_keys:
        data.ctrl[0] -= STEP

    # Joint 2
    if glfw.KEY_KP_3 in pressed_keys:
        data.ctrl[1] += STEP

    if glfw.KEY_KP_4 in pressed_keys:
        data.ctrl[1] -= STEP

    # Joint 3
    if glfw.KEY_KP_5 in pressed_keys:
        data.ctrl[2] += STEP

    if glfw.KEY_KP_6 in pressed_keys:
        data.ctrl[2] -= STEP

    # Joint 4
    if glfw.KEY_KP_7 in pressed_keys:
        data.ctrl[3] += STEP

    if glfw.KEY_KP_8 in pressed_keys:
        data.ctrl[3] -= STEP

    # Joint 5
    if glfw.KEY_KP_9 in pressed_keys:
        data.ctrl[4] += STEP

    if glfw.KEY_KP_0 in pressed_keys:
        data.ctrl[4] -= STEP

    # Joint 6
    if glfw.KEY_KP_SUBTRACT in pressed_keys:
        data.ctrl[5] += STEP

    if glfw.KEY_KP_ADD in pressed_keys:
        data.ctrl[5] -= STEP


# ============================================================
# Camera
# ============================================================

cam = mj.MjvCamera()
opt = mj.MjvOption()

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

# ------------------------------------------------------------
# Find wrist camera
# ------------------------------------------------------------

WRIST_CAMERA_NAME = "wrist_cam"

wrist_camera_id = mj.mj_name2id(
    model,
    mj.mjtObj.mjOBJ_CAMERA,
    WRIST_CAMERA_NAME
)

if wrist_camera_id < 0:
    print(
        f"WARNING: camera '{WRIST_CAMERA_NAME}' "
        f"not found in XML."
    )

# False = free camera
# True  = wrist camera
use_wrist_camera = False


def switch_camera():

    global use_wrist_camera

    use_wrist_camera = not use_wrist_camera

    if use_wrist_camera:

        print("Camera -> WRIST CAMERA")

        if wrist_camera_id >= 0:

            cam.type = mj.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = wrist_camera_id

    else:

        print("Camera -> FREE CAMERA")

        cam.type = mj.mjtCamera.mjCAMERA_FREE


# ============================================================
# GLFW initialization
# ============================================================

if not glfw.init():
    raise RuntimeError("Could not initialize GLFW")


window = glfw.create_window(
    1200,
    900,
    "ARX-15",
    None,
    None
)

if not window:
    glfw.terminate()
    raise RuntimeError("Could not create GLFW window")


glfw.make_context_current(window)
glfw.swap_interval(1)


# mouse callback
glfw.set_mouse_button_callback(
    window,
    mouse_button
)

glfw.set_cursor_pos_callback(
    window,
    mouse_move
)

glfw.set_scroll_callback(
    window,
    scroll
)

# Keyboard callback
glfw.set_key_callback(
    window,
    keyboard
)


# ============================================================
# MuJoCo rendering structures
# ============================================================

scene = mj.MjvScene(
    model,
    maxgeom=10000
)

context = mj.MjrContext(
    model,
    mj.mjtFontScale.mjFONTSCALE_150.value
)


# ============================================================
# Camera configuration
# ============================================================

cam.azimuth = 145.0
cam.elevation = -33
cam.distance = 2.2

cam.lookat = np.array([0.0, 0.3, 0.0])


# ============================================================
# Main loop
# ============================================================

while not glfw.window_should_close(window):
    time.sleep(0.01)
    # --------------------------------------------------------
    # Keyboard control
    # --------------------------------------------------------

    update_control()


    # --------------------------------------------------------
    # Physics
    # --------------------------------------------------------

    mj.mj_step(model, data)


    # --------------------------------------------------------
    # Get framebuffer size
    # --------------------------------------------------------

    width, height = glfw.get_framebuffer_size(window)

    viewport = mj.MjrRect(
        0,
        0,
        width,
        height
    )


    # --------------------------------------------------------
    # Update scene
    # --------------------------------------------------------

    mj.mjv_updateScene(
        model,
        data,
        opt,
        None,
        cam,
        mj.mjtCatBit.mjCAT_ALL.value,
        scene
    )


    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    mj.mjr_render(
        viewport,
        scene,
        context
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    glfw.swap_buffers(window)

    # Process keyboard / mouse events
    glfw.poll_events()


# ============================================================
# Cleanup
# ============================================================

glfw.terminate()
