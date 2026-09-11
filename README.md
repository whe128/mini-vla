# MiniVLA
<div align="center">
<img width="889" height="500" alt="搜狗截图20260911192000" src="https://github.com/user-attachments/assets/177c7e21-999e-4ae4-b125-b03695a89286" />
<div>

A lightweight Vision-Language-Action (VLA) system for robot manipulation using **PyTorch**, **MuJoCo**, and a custom MiniGPT-based Transformer.

The project connects a vision-language model with a simulated robot and predicts robot actions directly from:

- Camera images
- Natural-language instructions

The predicted action is a **7-dimensional absolute actuator control vector**, which is directly applied to the MuJoCo robot.

---

## 1. Overview

The overall pipeline is:

```text
                ┌──────────────────────┐
                │      Instruction     │
                │ "put cube on green   │
                │       platform"      │
                └──────────┬───────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│                    MiniVLA                      │
│                                                 │
│     Image ──► Vision Encoder ──► Image Tokens   │
│                          │                      │
│          Text ──► Tokenizer ──► Text Tokens     │
│                          │                      │
│                      Transformer                │
│                          │                      │
│                       Action Head               │
│                          │                      │
│                       7-D Action                │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
                    MuJoCo Robot
```

The VLA and MuJoCo renderer run as **separate processes**.

Communication uses:

```text
TCP Socket
    │
    ├── request_image
    ├── image_ready
    └── action_ready

Shared Memory
    │
    ├── Image Buffer
    └── Action Buffer
```

TCP is used for **notification and control**, while SharedMemory is used for **large data transfer**.

---

# 2. Architecture

The system consists of two processes.

## Render Process

The Render process owns the MuJoCo simulation and SharedMemory.

Responsibilities:

- Run MuJoCo physics
- Render the robot
- Render the wrist camera
- Create SharedMemory
- Start TCP server
- Receive image requests
- Write images to SharedMemory
- Receive actions
- Apply actions to `data.ctrl[:7]`

```text
MuJoCo
   │
   ├── Physics
   ├── Main Camera
   └── Wrist Camera
          │
          ▼
     224 × 224 × 3
          │
          ▼
     SharedMemory
```

---

## VLA Process

The VLA process connects to the Render process.

Responsibilities:

- Load the trained MiniVLA checkpoint
- Receive the natural-language instruction
- Request an image
- Read the image from SharedMemory
- Run inference
- Write the predicted action to SharedMemory
- Notify Render that the action is ready
- Request the next image

---

## Action

After receiving `image_ready`, VLA:

```text
Read Image
    ↓
MiniVLA Inference
    ↓
7-D Action
```

The action is written into SharedMemory.

VLA then sends:

```text
action_ready
```

Render reads the action and applies:

```python
data.ctrl[:7] = action
```

---

## Next request

After the action is processed, VLA checks the elapsed time since the previous `request_image`.

If:

```text
elapsed < DT
```

VLA waits for the remaining time.

Otherwise, it immediately requests the next image.

This prevents multiple image requests from accumulating while inference is still running.

---

# 3. Action Space

The current MiniVLA predicts:

```text
action_dim = 7
```

The action represents the absolute actuator control values:

```python
[
    ctrl[0],
    ctrl[1],
    ctrl[2],
    ctrl[3],
    ctrl[4],
    ctrl[5],
    ctrl[6],
]
```

The VLA does **not** use IK during inference.

The predicted action is directly applied to MuJoCo:

```python
self.data.ctrl[:7] = action
```

IK is only used for convenient teleoperation/data collection.

---

# 4. Observation

The current VLA uses the robot wrist camera.

Image size:

```text
224 × 224 × 3
```

Data type:

```text
uint8
```

Format:

```text
RGB
```

The image is stored in SharedMemory:

```python
image_buffer = np.ndarray(
    (224, 224, 3),
    dtype=np.uint8,
    buffer=image_shm.buf
)
```

---

# 5. Model

MiniVLA reuses the Transformer architecture from the MiniGPT project.

The model contains:

```text
Image
  ↓
Vision Encoder
  ↓
Image Tokens
       \
        \
         Transformer
        /
       /
Text Tokens
  ↓
Tokenizer
```

A learned `action_token` is appended to the sequence.

The final Transformer representation of the action token is passed to the action head:

```text
[Image Tokens]
      +
[Text Tokens]
      +
[Action Token]
      ↓
Transformer
      ↓
Final Action Token
      ↓
Action Head
      ↓
7-D Action
```

The Transformer uses full attention rather than causal language-model attention for the VLA fusion stage.

---

# 6. Dataset

The training data is organized by episode.

Example:

```text
dataset/
├── episode_0000/
│   ├── episode.json
│   └── images/
│       ├── frame_000000.png
│       ├── frame_000001.png
│       └── ...
│
├── episode_0001/
│   ├── episode.json
│   └── images/
│       ├── frame_000000.png
│       └── ...
│
└── ...
```

Each sample contains:

```json
{
  "image": "images/frame_000000.png",
  "instruction": "pick up the cube on green platform",
  "action": [0.0, 1.4, 0.6, 0.6, 0.0, 0.0, 0.0]
}
```

The dataset follows the standard **Behavior Cloning** formulation:

```text
observation + instruction → expert action
```

---

# 7. Data Collection

The robot can be controlled manually using MuJoCo and IK.

During data collection:

```text
User Control
     ↓
MuJoCo
     ↓
Robot State
     +
Wrist Camera Image
     ↓
Dataset
```

Images are recorded at approximately:

```text
10 Hz
```

The recorded action is the current actuator control:

```python
data.ctrl.tolist()
```

---

# 8. Training

The initial training strategy is:

### Stage 1

Freeze the vision encoder.

Train:

```text
Image Projection
+
Transformer Fusion
+
Action Head
```

This reduces the number of trainable parameters and is suitable for the limited dataset and GTX 1650 4GB GPU.

### Stage 2

After the first model works, optionally unfreeze part of the vision encoder for fine-tuning.

---

# 9. Inference

Start the Render process first:

```bash
python -m render_control
```

The Render process starts the TCP server:

```text
127.0.0.1:5000
```

Then start the VLA process in another terminal:

```bash
python -m vla_control.vla_node
```

The VLA connects to Render and asks for an instruction:

```text
[VLA] Running...
Instruction:
```

Example:

```text
put cube on green platform
```

The inference loop is:

```text
request_image
      ↓
image_ready
      ↓
read image
      ↓
inference
      ↓
action_ready
      ↓
wait if necessary
      ↓
request_image
      ↓
repeat
```

---

# 10. Example Output

A successful inference may produce:

```text
[VLA] Received image_ready signal from Render
[VLA] Starting inference...
[VLA] Inference done
[VLA] Action: [
    0.0430,
    1.5728,
    0.6282,
    0.6987,
   -0.0144,
   -0.0021,
    0.0142
]
[VLA] Elapsed time: 1595.4 ms
[VLA] Next request_image sent
```

The corresponding Render process receives:

```text
[Render] Received message: b'request_image'
[Render] Image sent to shared memory
[Render] Received message: b'action_ready'
[Render] Received action: [...]
```

---

# 11. Performance

The current hardware is:

```text
GPU: NVIDIA GTX 1650 4GB
```

The current MiniVLA inference latency can be significantly larger than:

```text
DT = 100 ms
```

For example:

```text
Inference time ≈ 150–200 ms
```

In this situation, the actual VLA control frequency is determined by inference latency rather than the configured 10 Hz target.

---

# 12. Project Structure

A typical project structure is:

```text
mini-vla/
│
├── arx_15/
│   └── scene.xml
│
├── dataset/
│
├── model/
│   ├── config.py
│   ├── tokenizer.py
│   ├── mini_vla.py
│   └── ...
│
├── output/
│   └── minivla.pt
│
├── render_control/
│   └── ...
│
├── vla_control/
│   └── vla_node.py
│
├── scripts/
│   ├── collect_demo.py
│   └── ...
│
└── README.md
```
