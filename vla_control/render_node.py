import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import os
from multiprocessing import shared_memory
import socket
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

VLA_HOST = "127.0.0.1"
VLA_PORT = 5000

ACTION_DIM = 7
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS
ACTION_SIZE = ACTION_DIM * np.dtype(np.float32).itemsize

EE_SITE_NAME = "ee_site"
POSITION_STEP = 0.002
DAMPING = 0.05

class ControlRender:

    def __init__(
        self,
        model_path,
        width=1200,
        height=900,
        title="MuJoCo",
        image_shm_name="minivla_image",
        action_shm_name="minivla_action",
        vla_host = VLA_HOST,
        vla_port = VLA_PORT,
    ):
        #====================================================
        # shared memory
        #====================================================
        self.image_shm = shared_memory.SharedMemory(
            name=image_shm_name,
            create=True,
            size=IMAGE_SIZE
        )

        self.action_shm = shared_memory.SharedMemory(
            name=action_shm_name,
            create=True,
            size=ACTION_SIZE
        )
        self.image_buffer = np.ndarray(
            (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            dtype=np.uint8,
            buffer=self.image_shm.buf
        )
        self.action_buffer = np.ndarray(
            (ACTION_DIM,),
            dtype=np.float32,
            buffer=self.action_shm.buf
        )
        #====================================================
        # socket
        #====================================================
        self.vla_host = vla_host
        self.vla_port = vla_port
        self.recv_buffer = b""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.start_server()
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
        self.ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

        if self.ee_site_id < 0:
            raise RuntimeError( f"Cannot find end-effector site: {EE_SITE_NAME}")

        print(f"End-effector site: {EE_SITE_NAME}, id={self.ee_site_id}")

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


        # one rendering context shared for both scenes
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # main camera
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        # wrist camera
        self.wrist_camera_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_CAMERA,
            "wrist_cam"
        )
        self.wrist_scene = mj.MjvScene(self.model, maxgeom=10000)
        self.wrist_cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.wrist_cam)
        self.wrist_cam.type = (mj.mjtCamera.mjCAMERA_FIXED)

        self.wrist_cam.fixedcamid = (self.wrist_camera_id)

        # ====================================================
        # Camera state
        # ====================================================

        self.use_wrist_camera = False

        # ====================================================
        # Mouse state
        # ====================================================
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # ====================================================
        # Event callbacks
        # ====================================================


        glfw.set_mouse_button_callback(self.window, self._mouse_button)

        glfw.set_cursor_pos_callback(self.window, self._mouse_move)

        glfw.set_scroll_callback(self.window, self._scroll)

    # ========================================================
    # Public API
    # ========================================================
    def start_server(self):
        print(
            f"[Render] Starting Socket Server "
            f"{self.vla_host}:{self.vla_port}"
        )

        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.socket.bind((self.vla_host, self.vla_port))
        self.socket.listen(1)
        self.socket.settimeout(1)


        print(f"[Render] Waiting for VLA connection on {self.vla_host}:{self.vla_port} ...")

    def wait_for_vla(self):
        try:
            self.conn, addr = self.socket.accept()
            print(f"[Render] VLA connected from {addr}")
            self.conn.setblocking(False)
            return True
        except socket.timeout:
            return False

    def set_camera(self, azimuth, elevation, distance, lookat):
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.distance = distance
        self.cam.lookat = np.array(lookat)

    def send_img(self, viewport):
        # read the image from the viewport
        img = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            dtype=np.uint8
        )

        mj.mjr_readPixels(
            img,
            None,
            viewport,
            self.context
        )
        img = np.flipud(img)

        # copy to shared memory
        np.copyto(self.image_buffer, img)
        # notify
        try:
            self.conn.sendall(b"image_ready\n")
        except Exception as e:
            print(f"Error sending image_ready: {e}")

    def receive_message(self, wrist_viewport):
        while True:
            try:
                data = self.conn.recv(1024)
                if not data:
                    return
                # tcp is a stream
                self.recv_buffer += data
            except BlockingIOError:
                return
            except (ConnectionResetError, ConnectionAbortedError):
                print("Connection to VLA lost.")
                return

            # parse complete messages
            while b"\n" in self.recv_buffer:
                message, self.recv_buffer = self.recv_buffer.split(b"\n", 1)
                message = message.strip()
                if message == b"request_image":
                    self.send_img(wrist_viewport)
                elif message == b"action_ready":
                    # read action from shared memory
                    action = self.action_buffer.copy()
                    self.data.ctrl[:ACTION_DIM] = action
                    print(f"Received action: {action}")

    def run(self, callback=None):

        while not glfw.window_should_close(self.window):
            if self.conn is None:
                print(f"[Render] Waiting for VLA connection on {self.vla_host}:{self.vla_port} ...")

                self.wait_for_vla()

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

            main_viewport = mj.MjrRect( 0, 0, width, height)

            # update main scene
            mj.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene
            )

            mj.mjr_render( main_viewport, self.scene, self.context)

            # render wrist camera scene
            image_width = 224
            image_height = 224
            wrist_x = width - image_width - 10
            wrist_y = height - image_height -10

            wrist_viewport = mj.MjrRect( wrist_x,wrist_y,image_width, image_height)

            mj.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.wrist_cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.wrist_scene
            )

            # Render wrist scene
            mj.mjr_render(
                wrist_viewport,
                self.wrist_scene,
                self.context
            )

            if self.conn is not None:
                self.receive_message(wrist_viewport)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

        try:
            self.socket.close()
        except Exception:
            pass

        try:
            self.image_shm.close()
            self.image_shm.unlink()
        except Exception:
            pass

        try:
            self.action_shm.close()
            self.action_shm.unlink()
        except Exception:
            pass

        glfw.terminate()

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

    def _scroll( self, window, xoffset, yoffset):
        if self.use_wrist_camera:
            return

        action = ( mj.mjtMouse.mjMOUSE_ZOOM)

        mj.mjv_moveCamera(
            self.model,
            action,
            0.0,
            -0.05 * yoffset,
            self.cam
        )


if __name__ == "__main__":
    MODEL_PATH = "arx_15/scene.xml"

    render = ControlRender(
        model_path=MODEL_PATH,
        width=1200,
        height=900,
        title="ARX-15 - IK",

        image_shm_name="minivla_image",
        action_shm_name="minivla_action",

        vla_host="127.0.0.1",
        vla_port=5000
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
    try:
        render.run()
    finally:
        render.close()
