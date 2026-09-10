# models/dataset.py

import os
import json

import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

DATASET_DIR = "dataset"
NEED_AUGMENTATION_INSTRUCTION = True

INSTRUCTION_VARIANTS = {
    "pick up the cube on green platform": [
        "pick up the cube on the green platform",
        "pick up the cube from the green platform",
        "pick up the cube located on the green platform",
        "pick up the cube sitting on the green platform",

        "grab the cube on the green platform",
        "grab the cube from the green platform",
        "grab the cube located on the green platform",
        "grab the cube sitting on the green platform",

        "grasp the cube on the green platform",
        "grasp the cube from the green platform",
        "grasp the cube located on the green platform",

        "take the cube from the green platform",
        "take the cube on the green platform",
        "get the cube from the green platform",
        "get the cube on the green platform",

        "lift the cube from the green platform",
        "lift the cube on the green platform",

        "retrieve the cube from the green platform",
        "retrieve the cube on the green platform",

        # Short command-like forms
        "pick up cube on green platform",
        "grab cube on green platform",
        "grasp cube on green platform",
        "take cube from green platform",
        "lift cube from green platform",
    ],

    "pick up the cube on blue platform": [
        "pick up the cube on the blue platform",
        "pick up the cube from the blue platform",
        "pick up the cube located on the blue platform",
        "pick up the cube sitting on the blue platform",

        "grab the cube on the blue platform",
        "grab the cube from the blue platform",
        "grab the cube located on the blue platform",
        "grab the cube sitting on the blue platform",

        "grasp the cube on the blue platform",
        "grasp the cube from the blue platform",
        "grasp the cube located on the blue platform",

        "take the cube from the blue platform",
        "take the cube on the blue platform",
        "get the cube from the blue platform",
        "get the cube on the blue platform",

        "lift the cube from the blue platform",
        "lift the cube on the blue platform",

        "retrieve the cube from the blue platform",
        "retrieve the cube on the blue platform",

        # Short command-like forms
        "pick up cube on blue platform",
        "grab cube on blue platform",
        "grasp cube on blue platform",
        "take cube from blue platform",
        "lift cube from blue platform",
    ],
}

class MiniVLADataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.samples = []

        episodes = sorted(
            name for name in os.listdir(DATASET_DIR)
            if name.startswith("episode_")
        )

        for episode_name in episodes:
            episode_dir = os.path.join(DATASET_DIR, episode_name)
            json_path = os.path.join(episode_dir, "episode.json")

            if not os.path.isdir(episode_dir):
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            for item in data:
                # =========================================================
                # Instruction augmentation
                # =========================================================
                original_instruction = item["instruction"]
                if NEED_AUGMENTATION_INSTRUCTION and original_instruction in INSTRUCTION_VARIANTS:
                    instruction = random.choice(INSTRUCTION_VARIANTS[original_instruction])
                else:
                    # fallback
                    instruction = original_instruction

                self.samples.append({
                    "image": os.path.join(episode_dir, item["image"]),
                    "instruction": instruction,
                    "action": item["action"],
                })

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]


        # original image is already 224x224
        image = Image.open(item["image"]).convert("RGB")
        image = transforms.ToTensor()(image)


        tokenized = self.tokenizer.tokenizer(
            item["instruction"],
            max_length = self.config.max_text_len,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )

        text = tokenized["input_ids"].squeeze(0)
        mask = tokenized["attention_mask"].squeeze(0)

        action = torch.tensor(item["action"], dtype = torch.float32)

        return image, text, action, mask

