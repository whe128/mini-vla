import torch
from PIL import Image
from torchvision import transforms

from model.tokenizer import Tokenizer
from model.mini_vla import MiniVLA
from model.config import VLAConfig
import socket
from multiprocessing import shared_memory
import numpy as np
from PIL import Image
import time

VLA_HOST = "127.0.0.1"
VLA_PORT = 5000
IMAGE_SHM_NAME = "minivla_image"
ACTION_SHM_NAME = "minivla_action"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
ACTION_DIM = 7
FREQUENCY = 10

DT = 1 / FREQUENCY
IMAGE_SIZE = ( IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS )
ACTION_SIZE = ( ACTION_DIM * np.dtype(np.float32).itemsize )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "output/minivla.pt"


MODEL_PATH = "./arx_15/scene.xml"
WRIST_CAMERA_NAME = "wrist_cam"


class VLANode:
    def __init__(
            self,
            image_shm_name=IMAGE_SHM_NAME,
            action_shm_name=ACTION_SHM_NAME,
            vla_host=VLA_HOST,
            vla_port=VLA_PORT
        ):
        print(f"[VLA] Device: {DEVICE}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[VLA] Connecting to Render at {VLA_HOST}:{VLA_PORT} ...")
        self.socket.connect((VLA_HOST, VLA_PORT))
        print(f"[VLA] Connected to Render")

        # Create shared memory for image and action
        self.image_shm = shared_memory.SharedMemory(name=image_shm_name)
        self.action_shm = shared_memory.SharedMemory(name=action_shm_name)

        self.image_buffer = np.ndarray(
            ( IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS ),
            dtype=np.uint8,
            buffer=self.image_shm.buf
        )

        self.action_buffer = np.ndarray(
            (ACTION_DIM,),
            dtype=np.float32,
            buffer=self.action_shm.buf
        )

        print("[VLA] SharedMemory connected")


        # ====================================================
        # Load tokenizer
        # ====================================================
        print("[VLA] Loading tokenizer...")
        self.tokenizer = Tokenizer()

        # ====================================================
        # Load model
        # ====================================================
        print( f"[VLA] Loading checkpoint: " f"{CHECKPOINT}" )
        checkpoint = torch.load( CHECKPOINT, map_location=DEVICE )

        cfg = checkpoint["config"]
        self.model_config = VLAConfig(**cfg)
        self.model = MiniVLA(self.model_config).to(DEVICE)
        self.model.load_state_dict(checkpoint["model"])

        self.model.eval()

        print("[VLA] Model loaded")

        # ====================================================
        # socket receive buffer
        # ====================================================
        self.recv_buffer = b""

        # current instruction
        self.instruction = None

    def receive_message(self):
        while True:
            data = self.socket.recv(1024)
            if not data:
                raise ConnectionError("Render disconnected")
            self.recv_buffer += data
            while b"\n" in self.recv_buffer:
                message, self.recv_buffer = self.recv_buffer.split(b"\n", 1)
                message = message.strip()

                if message:
                    return message

    def read_image(self):
        # Read image from shared memory
        img = self.image_buffer.copy()
        return img

    @torch.no_grad()
    def predict_action(self, image, instruction):
        # Preprocess image
        image = Image.fromarray(image)
        image = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)

        # Tokenize instruction
        tokenized = self.tokenizer.tokenizer(
            instruction,
            max_length=self.model_config.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        text = tokenized["input_ids"].to(DEVICE)
        mask = tokenized["attention_mask"].to(DEVICE)

        action = self.model(image, text, mask)

        action = action[0]
        action = action .detach() .cpu() .numpy() .astype(np.float32)

        return action

    def send_action(self, action):
        # Write action to shared memory
        np.copyto( self.action_buffer, action )

        self.socket.sendall(b"action_ready\n")

    def run(self):
        print("[VLA] Running...")
        self.instruction = input( "Instruction: " )

        self.socket.sendall(b"request_image\n")

        last_time = time.perf_counter()
        while True:
            # Wait for image_ready signal from Render
            message = self.receive_message()
            if message == b"image_ready":
                # Read image from shared memory
                img = self.read_image()

                # Predict action
                action = self.predict_action(img, self.instruction)
                print("[VLA] Action:", action)

                # Send action to Render
                self.send_action(action)
                now = time.perf_counter()
                elapsed = now - last_time
                if elapsed < DT:
                    sleep_time = DT - elapsed
                    time.sleep(sleep_time)
                    print( f"[VLA] Waiting " f"{sleep_time * 1000:.1f} ms" )
                print(f"[VLA] Elapsed time: {elapsed * 1000:.1f} ms")
                # request next image
                self.socket.sendall(b"request_image\n")
                last_time = time.perf_counter()

    def close(self):
        try:
            self.socket.close()
        except Exception:
            pass

        try:
            self.image_shm.close()
        except Exception:
            pass

        try:
            self.action_shm.close()
        except Exception:
            pass

if __name__ == "__main__":
    vla_node = VLANode()
    try:
        vla_node.run()
    except KeyboardInterrupt:
        print("[VLA] KeyboardInterrupt received. Exiting...")
    finally:
        vla_node.close()
