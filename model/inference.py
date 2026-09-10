import torch
from PIL import Image
from torchvision import transforms

from model.tokenizer import Tokenizer
from model.mini_vla import MiniVLA
from model.config import VLAConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "output/minivla.pt"
IMAGE_PATH = "dataset/episode_0000/images/frame_000000.png"

INSTRUCTION = "pick up the cube on blue platform"


def main():

    # =========================
    # Tokenizer
    # =========================

    tokenizer = Tokenizer()

    # =========================
    # Load checkpoint
    # =========================

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    cfg = checkpoint["config"]

    model_config = VLAConfig(**cfg)

    model = MiniVLA(model_config).to(DEVICE)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    # =========================
    # Image
    # =========================

    image = Image.open(IMAGE_PATH).convert("RGB")

    image = transforms.ToTensor()(image)

    # [3, 224, 224]
    image = image.unsqueeze(0).to(DEVICE)

    # [1, 3, 224, 224]

    # =========================
    # Text
    # =========================

    tokenized = tokenizer.tokenizer(
        INSTRUCTION,
        max_length=model_config.max_text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    text = tokenized["input_ids"].to(DEVICE)
    mask = tokenized["attention_mask"].to(DEVICE)

    # =========================
    # Inference
    # =========================

    with torch.no_grad():

        action = model(image, text, mask)

    print("Predicted action:")
    print(action[0].cpu().tolist())


if __name__ == "__main__":
    main()
