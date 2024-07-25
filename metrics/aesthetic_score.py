import torch
import clip
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="./output_images",
        help="The path to save generated images",
    )

    args = parser.parse_args()
    return args


args = parse_args()


class MLP(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# Set up model and CLIP
model_path = "pretrained/sac+logos+ava1-l14-linearMSE.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(768).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
model2, preprocess = clip.load("ViT-L/14", device=device)


def evaluate_images(folder_path):
    path = Path(folder_path)
    images = list(path.rglob("*.png"))
    scores = []

    for img_path in images:
        pil_image = Image.open(img_path)
        image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model2.encode_image(image)
            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            prediction = model(
                torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
            )
            scores.append(prediction.item())

        print(f"Aesthetic score for {img_path}: {prediction.item()}")

    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average aesthetic score for all images: {average_score}")
        return scores, average_score
    else:
        print("No images found.")
        return scores, None


scores, average_score = evaluate_images(args.image_path)
