# models/train.py

import os
import random
import torch
import yaml
import torch.utils.data as data
import torch.nn as nn

from model.tokenizer import Tokenizer
from model.mini_vla import MiniVLA
from model.dataset import MiniVLADataset
from model.config import VLAConfig


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = Tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")

    # config
    model_config = VLAConfig(
        vocab_size = tokenizer.vocab_size(),
        image_size = cfg["image_size"],
        image_dim = cfg["image_dim"],
        patch_size = cfg["patch_size"],
        n_layer = cfg["n_layer"],
        n_head = cfg["n_head"],
        n_embed = cfg["n_embed"],
        dropout = cfg["dropout"],
        action_dim = cfg["action_dim"],
        max_text_len = cfg["max_text_len"]
    )

    # dataset
    train_dataset = MiniVLADataset(config = model_config, tokenizer = tokenizer)

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size = cfg["batch_size"],
        shuffle = True,
        drop_last = True)

    # model
    model = MiniVLA(model_config).to(device)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {parameters / 1e6:.2f}M")

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"]
    )

    # -----------------------
    # AMP
    # -----------------------
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp )


    # -----------------------
    # Load checkpoint
    # -----------------------
    checkpoint_path = "output/minivla.pt"

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        update_step = checkpoint["step"]

        print(f"Resume training from step {update_step}")
    else:
        update_step = 0
        print("No checkpoint found. Starting training from scratch.")

    # -----------------------
    # Training
    # -----------------------
    try:
        for epoch in range(cfg["num_epochs"]):
            for image, text, action, mask in train_dataloader:
                # stop training if we reach the max update step
                image = image.to(device)
                text = text.to(device)
                action = action.to(device)
                mask = mask.to(device)

                with torch.autocast(
                    device_type = device.type,
                    dtype = torch.float16,
                    enabled = torch.cuda.is_available()

                ):
                    output = model(image, text, mask)
                    loss = nn.MSELoss()(output, action)

                # backward pass
                optimizer.zero_grad(set_to_none = True)
                scaler.scale(loss).backward()

                # optimizer step
                scaler.step(optimizer)
                scaler.update()

                # update step
                update_step += 1

                print(
                    f"Epoch {epoch + 1}/{cfg['num_epochs']} "
                    f"| Step {update_step:5d} "
                    f"| Loss: {loss.item():.6f}"
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        # -----------------------
        # Save checkpoint
        # -----------------------
        # save the model
        os.makedirs("output", exist_ok = True)
        checkpoint_path = "output/minivla.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": update_step,
                "config": model_config.__dict__,
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved at step {update_step}: "
            f"{checkpoint_path}"
        )

        print( f"Training finished. " f"Checkpoint saved to {checkpoint_path}" )

if __name__ == "__main__": main()
