import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from tqdm import tqdm
from utils.load_img_data import Dataset


def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print(
                "WARNING: You have a CUDA device, so you should probably set cuda=True"
            )
        dtype = torch.FloatTensor

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode="bilinear").type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batchv)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_inception_score(
    image_dir, cuda=True, batch_size=32, resize=True, splits=10
):
    imgs = Dataset(
        image_dir,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    return inception_score(
        imgs, cuda=cuda, batch_size=batch_size, resize=resize, splits=splits
    )


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for data loader"
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=10,
        help="Number of splits for Inception Score calculation",
    )
    parser.add_argument(
        "--resize", type=bool, default=True, help="Whether to resize images to 299x299"
    )
    parser.add_argument(
        "--cuda", type=bool, default=True, help="Whether to use GPU for calculations"
    )

    args = parser.parse_args()

    score, std = calculate_inception_score(
        args.path,
        cuda=args.cuda,
        batch_size=args.batch_size,
        resize=args.resize,
        splits=args.splits,
    )

    print(f"Inception Score: {score} ± {std}")
