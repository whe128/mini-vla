import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import os

EE_SITE_NAME = "ee_site"
POSITION_STEP = 0.002
DAMPING = 0.05

class Render:

    def __init__(
        self,
        model_path,
        width=1200,
        height=900,
        title="MuJoCo"
    ):
        # ====================================================
        # Load model
        # ====================================================

        model_path = os.path.abspath(model_path)

        self.model = mj.MjModel.from_xml_path(model_path)

        self.data = mj.MjData(self.model)
        self.key_set = set()

        # ====================================================
        # Find end-effector
        # ====================================================

        self.ee_site_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_SITE,
            EE_SITE_NAME
        )

        if self.ee_site_id < 0:
            raise RuntimeError(
                f"Cannot find end-effector site: "
                f"{EE_SITE_NAME}"
            )


        print(
            f"End-effector site: "
            f"{EE_SITE_NAME}, id={self.ee_site_id}"
        )

        # ====================================================
        # Initial state
        # ====================================================

        key_id = self.model.key("scene_home").id

        mj.mj_resetDataKeyframe(self.model, self.data, key_id)

        # ====================================================
        # GLFW
        # ====================================================

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        self.window = glfw.create_window(width, height, title, None, None)

        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # ====================================================
        # MuJoCo rendering
        # ====================================================

        self.scene = mj.MjvScene(self.model, maxgeom=10000)

        self.context = mj.MjrContext(
            self.model,
            mj.mjtFontScale.mjFONTSCALE_150.value
        )

        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        # ====================================================
        # Mouse state
        # ====================================================

        self.button_left = False
        self.button_middle = False
        self.button_right = False

        self.lastx = 0
        self.lasty = 0

        # ====================================================
        # Camera state
        # ====================================================

        self.use_wrist_camera = False

        self.wrist_camera_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_CAMERA,
            "wrist_cam"
        )

        # ====================================================
        # wrist camera scene
        # ====================================================
        self.wrist_cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.wrist_cam)
        self.wrist_cam.type = (mj.mjtCamera.mjCAMERA_FIXED)
        self.wrist_cam.fixedcamid = (self.wrist_camera_id)
        self.wrist_scene = mj.MjvScene(self.model, maxgeom=10000)
        self.wrist_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # ====================================================
        # Event callbacks
        # ====================================================

        glfw.set_key_callback(self.window, self._keyboard)

        glfw.set_mouse_button_callback(self.window, self._mouse_button)

        glfw.set_cursor_pos_callback(self.window, self._mouse_move)

        glfw.set_scroll_callback(self.window, self._scroll)

    def get_image(self):

        width = 224
        height = 224

        viewport = mj.MjrRect(
            0,
            0,
            width,
            height
        )

        # update wrist camera scene
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.wrist_cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.wrist_scene
        )

        # render
        mj.mjr_render(
            viewport,
            self.wrist_scene,
            self.wrist_context
        )

        # RGB image
        img = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        mj.mjr_readPixels(
            img,
            None,
            viewport,
            self.wrist_context
        )

        # flip vertically
        img = np.flipud(img)

        return img

    # ========================================================
    # Public API
    # ========================================================

    def set_camera(self, azimuth, elevation, distance, lookat):
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.distance = distance
        self.cam.lookat = np.array(lookat)

    # ============================================================
    # IK
    # ============================================================

    def apply_ik(self):

        # --------------------------------------------------------
        # Cartesian velocity
        # --------------------------------------------------------

        dx = 0.0
        dy = 0.0
        dz = 0.0

        roll = 0.0
        pitch = 0.0
        yaw = 0.0


        # --------------------------------------------------------
        # Translation
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Nothing to do
        # --------------------------------------------------------

        if (
            dx == 0.0
            and dy == 0.0
            and dz == 0.0
            and roll == 0.0
            and pitch == 0.0
            and yaw == 0.0
        ):
            return

        # --------------------------------------------------------
        # Jacobian
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # 6D Jacobian
        # --------------------------------------------------------

        J = np.vstack([
            jacp,
            jacr
        ])


        # --------------------------------------------------------
        # Desired Cartesian velocity
        # --------------------------------------------------------

        target = np.array([
            dx,
            dy,
            dz,
            roll,
            pitch,
            yaw
        ])


        # --------------------------------------------------------
        # Damped Least Squares
        #
        # dq = J.T (J J.T + λ²I)^-1 dx
        # --------------------------------------------------------

        lambda2 = DAMPING ** 2

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + lambda2 * np.eye(6),
                target
            )
        )


        # --------------------------------------------------------
        # Apply joint motion
        # --------------------------------------------------------
        for i in range(min(6, self.model.nu)):
            if i == 4:
                continue
            self.data.ctrl[i] += dq[i]

    def run(self, callback=None):

        while not glfw.window_should_close(self.window):
            # if callback is not None:
            #     callback(self.data.ctrl)

            self.apply_ik()

            # --------------------------------------------
            # Physics
            # --------------------------------------------

            mj.mj_step(self.model, self.data)


            # save information after IK
            if callback is not None:
                callback(self.data.ctrl)

            # --------------------------------------------
            # Render
            # --------------------------------------------

            width, height = ( glfw.get_framebuffer_size(self.window) )

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
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene
            )

            mj.mjr_render(
                viewport,
                self.scene,
                self.context
            )

            glfw.swap_buffers(self.window)

            glfw.poll_events()

    def close(self):
        glfw.terminate()

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

        # Q -> camera
        if (action == glfw.PRESS and key == glfw.KEY_Q):
            self.switch_camera()

        # space -> grabber
        elif ( action == glfw.PRESS and key == glfw.KEY_SPACE ):
            self.switch_grabber()


        if action == glfw.PRESS:
            self.key_set.add(key)

        elif action == glfw.RELEASE:
            self.key_set.discard(key)




    # ========================================================
    # Camera switching
    # ========================================================

    def switch_camera(self):

        self.use_wrist_camera = (
            not self.use_wrist_camera
        )

        if self.use_wrist_camera:

            print("Camera -> WRIST")

            if self.wrist_camera_id >= 0:

                self.cam.type = (
                    mj.mjtCamera.mjCAMERA_FIXED
                )

                self.cam.fixedcamid = (
                    self.wrist_camera_id
                )

        else:

            print("Camera -> FREE")

            self.cam.type = (
                mj.mjtCamera.mjCAMERA_FREE
            )

    # ========================================================
    # grabber switching
    # ========================================================
    def switch_grabber(self):

        if self.data.ctrl[6] < 0.022:
            self.data.ctrl[6] = 0.044
        else:
            self.data.ctrl[6] = 0.0

    # ========================================================
    # Mouse button
    # ========================================================

    def _mouse_button(
        self,
        window,
        button,
        action,
        mods
    ):

        self.button_left = (
            glfw.get_mouse_button(
                window,
                glfw.MOUSE_BUTTON_LEFT
            ) == glfw.PRESS
        )

        self.button_middle = (
            glfw.get_mouse_button(
                window,
                glfw.MOUSE_BUTTON_MIDDLE
            ) == glfw.PRESS
        )

        self.button_right = (
            glfw.get_mouse_button(
                window,
                glfw.MOUSE_BUTTON_RIGHT
            ) == glfw.PRESS
        )

        # Initialize cursor position
        self.lastx, self.lasty = (
            glfw.get_cursor_pos(window)
        )

    # ========================================================
    # Mouse move
    # ========================================================

    def _mouse_move(
        self,
        window,
        xpos,
        ypos
    ):

        dx = xpos - self.lastx
        dy = ypos - self.lasty

        self.lastx = xpos
        self.lasty = ypos

        # Wrist camera cannot be manually moved
        if self.use_wrist_camera:
            return

        if (
            not self.button_left
            and not self.button_middle
            and not self.button_right
        ):
            return

        width, height = (
            glfw.get_window_size(window)
        )

        if height <= 0:
            return

        left_shift = (
            glfw.get_key(
                window,
                glfw.KEY_LEFT_SHIFT
            ) == glfw.PRESS
        )

        right_shift = (
            glfw.get_key(
                window,
                glfw.KEY_RIGHT_SHIFT
            ) == glfw.PRESS
        )

        mod_shift = (
            left_shift or right_shift
        )

        # Right mouse
        if self.button_right:

            if mod_shift:
                action = (
                    mj.mjtMouse.mjMOUSE_MOVE_H
                )
            else:
                action = (
                    mj.mjtMouse.mjMOUSE_MOVE_V
                )

        # Left mouse
        elif self.button_left:

            if mod_shift:
                action = (
                    mj.mjtMouse.mjMOUSE_ROTATE_H
                )
            else:
                action = (
                    mj.mjtMouse.mjMOUSE_ROTATE_V
                )

        # Middle mouse
        else:

            action = (
                mj.mjtMouse.mjMOUSE_ZOOM
            )

        mj.mjv_moveCamera(
            self.model,
            action,
            dx / height,
            dy / height,
            self.cam
        )

    # ========================================================
    # Scroll
    # ========================================================

    def _scroll(
        self,
        window,
        xoffset,
        yoffset
    ):

        if self.use_wrist_camera:
            return

        action = (
            mj.mjtMouse.mjMOUSE_ZOOM
        )

        mj.mjv_moveCamera(
            self.model,
            action,
            0.0,
            -0.05 * yoffset,
            self.cam
        )
