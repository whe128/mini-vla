import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import time

EE_SITE_NAME = "ee_site"
WRIST_CAMERA_NAME = "wrist_cam"

POSITION_STEP = 0.002
DAMPING = 0.05


class WristRender:

    def __init__(
        self,
        model_path,
        width=640,
        height=480,
        title="ARX-15 - Wrist Camera",
    ):

        # ====================================================
        # Load model
        # ====================================================

        model_path = os.path.abspath(model_path)
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.key_set = set()
        self.start_collect = False
        self.las_record_time = time.perf_counter()

        # ====================================================
        # End effector
        # ====================================================

        self.ee_site_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_SITE,
            EE_SITE_NAME
        )

        if self.ee_site_id < 0:
            raise RuntimeError(
                f"Cannot find site: {EE_SITE_NAME}"
            )

        # ====================================================
        # Initial state
        # ====================================================

        key_id = self.model.key("scene_home").id

        mj.mj_resetDataKeyframe(
            self.model,
            self.data,
            key_id
        )

        # ====================================================
        # GLFW
        # ====================================================

        if not glfw.init():
            raise RuntimeError(
                "Could not initialize GLFW"
            )

        self.window = glfw.create_window(width, height, title, None, None)

        if not self.window:
            glfw.terminate()
            raise RuntimeError(
                "Could not create GLFW window"
            )

        glfw.make_context_current(self.window)

        # Disable vsync.
        # This avoids limiting the rendering loop to monitor refresh.
        glfw.swap_interval(0)

        # ====================================================
        # Wrist camera
        # ====================================================

        self.wrist_camera_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_CAMERA,
            WRIST_CAMERA_NAME
        )

        if self.wrist_camera_id < 0:
            raise RuntimeError(
                f"Cannot find camera: {WRIST_CAMERA_NAME}"
            )

        self.wrist_cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.wrist_cam)
        self.wrist_cam.type = (mj.mjtCamera.mjCAMERA_FIXED)
        self.wrist_cam.fixedcamid = (self.wrist_camera_id)

        # ====================================================
        # Rendering
        # ====================================================

        self.opt = mj.MjvOption()

        mj.mjv_defaultOption(self.opt)

        self.scene = mj.MjvScene( self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # ====================================================
        # Keyboard
        # ====================================================

        glfw.set_key_callback(
            self.window,
            self._keyboard
        )

        print()
        print("=" * 60)
        print("             WRIST CAMERA CONTROL")
        print("=" * 60)
        print()
        print("  UP       Forward")
        print("  DOWN     Backward")
        print("  LEFT     Left")
        print("  RIGHT    Right")
        print("  W        Up")
        print("  S        Down")
        print("  SPACE    Gripper")
        print("  ESC      Exit")
        print()
        print("=" * 60)

    # ========================================================
    # IK
    # ========================================================

    def apply_ik(self):

        dx = 0.0
        dy = 0.0
        dz = 0.0

        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        if glfw.KEY_UP in self.key_set:
            dx += POSITION_STEP

        if glfw.KEY_DOWN in self.key_set:
            dx -= POSITION_STEP

        if glfw.KEY_LEFT in self.key_set:
            dy += POSITION_STEP

        if glfw.KEY_RIGHT in self.key_set:
            dy -= POSITION_STEP

        if glfw.KEY_W in self.key_set:
            dz += POSITION_STEP

        if glfw.KEY_S in self.key_set:
            dz -= POSITION_STEP

        # ----------------------------------------------------
        # Nothing to do
        # ----------------------------------------------------

        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return
        self.start_collect = True

        # ----------------------------------------------------
        # Jacobian
        # ----------------------------------------------------

        jacp = np.zeros(
            (3, self.model.nv)
        )

        jacr = np.zeros(
            (3, self.model.nv)
        )

        mj.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.ee_site_id
        )

        # ----------------------------------------------------
        # 6D Jacobian
        # ----------------------------------------------------

        J = np.vstack([
            jacp,
            jacr
        ])

        target = np.array([
            dx,
            dy,
            dz,
            0.0,
            0.0,
            0.0
        ])

        # ----------------------------------------------------
        # Damped Least Squares
        # ----------------------------------------------------

        lambda2 = DAMPING ** 2

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + lambda2 * np.eye(6),
                target
            )
        )

        # ----------------------------------------------------
        # Apply control
        # ----------------------------------------------------

        for i in range(
            min(6, self.model.nu)
        ):

            if i == 4:
                continue

            self.data.ctrl[i] += dq[i]

    # ========================================================
    # Gripper
    # ========================================================

    def switch_grabber(self):

        if self.data.ctrl[6] < 0.022:

            self.data.ctrl[6] = 0.044

        else:

            self.data.ctrl[6] = 0.0

    # ========================================================
    # Keyboard
    # ========================================================

    def _keyboard(
        self,
        window,
        key,
        scancode,
        action,
        mods
    ):

        # ESC
        if (
            key == glfw.KEY_ESCAPE
            and action == glfw.PRESS
        ):

            glfw.set_window_should_close(
                window,
                True
            )

            return

        # Space -> gripper
        if (
            key == glfw.KEY_SPACE
            and action == glfw.PRESS
        ):

            self.switch_grabber()

        # Key pressed
        if action == glfw.PRESS:

            self.key_set.add(key)

        # Key released
        elif action == glfw.RELEASE:

            self.key_set.discard(key)

    # ========================================================
    # Run
    # ========================================================

    def run(self, callback=None):

        while not glfw.window_should_close(self.window):
            current_time = time.perf_counter()
            # ------------------------------------------------
            # IK
            # ------------------------------------------------

            self.apply_ik()

            # ------------------------------------------------
            # Physics
            # ------------------------------------------------

            mj.mj_step(
                self.model,
                self.data
            )

            # ------------------------------------------------
            # Render wrist camera
            # ------------------------------------------------

            width, height = (
                glfw.get_framebuffer_size(
                    self.window
                )
            )

            viewport = mj.MjrRect(
                0,
                0,
                width,
                height
            )

            mj.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.wrist_cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene
            )

            mj.mjr_render(
                viewport,
                self.scene,
                self.context
            )

            # ------------------------------------------------
            # Data collection
            # ------------------------------------------------

            if callback is not None and self.start_collect and (current_time - self.las_record_time) >= 0.1:
                self.las_record_time = current_time
                # Read current rendered framebuffer
                img = np.zeros(
                    (224, 224, 3),
                    dtype=np.uint8
                )

                mj.mjr_readPixels(
                    img,
                    None,
                    viewport,
                    self.context
                )

                img = np.flipud(img)
                callback(self.data.ctrl, img)

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            glfw.swap_buffers(
                self.window
            )

            glfw.poll_events()


    # ========================================================
    # Close
    # ========================================================

    def close(self):

        glfw.terminate()
