import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import logging
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def compare_images(standard_output, current_output):
    img1 = standard_output
    img2 = current_output
    img1, img2 = np.asarray(img1), np.asarray(img2)
    ssim_value = ssim(img1, img2, channel_axis=2)
    mse = ((img1 - img2) ** 2).mean()
    mae = abs(img1 - img2).mean()
    logger.debug(f"SSIM: {ssim_value}, MSE: {mse}, MAE: {mae}")

    return {'ssim': ssim_value, 'mse': mse, 'mae': mae}

def average_metrics(folder1, folder2):
    metrics = {'ssim': [], 'mse': [], 'mae': []}
    # for subfolder in os.listdir(folder1):
    # subfolder_path1 = os.path.join(folder1, subfolder)
    # subfolder_path2 = os.path.join(folder2, subfolder)
    images1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img1, img2 in zip(images1, images2):
        image1 = imread(img1)
        image2 = imread(img2)
        results = compare_images(image1, image2)
        for key in metrics:
            metrics[key].append(results[key])
    average_results = {k: np.mean(v) for k, v in metrics.items()}
    return average_results

def main():
    parser = argparse.ArgumentParser(description="Calculate average SSIM, MSE, and MAE between images in two folders.")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second folder")
    args = parser.parse_args()

    results = average_metrics(args.folder1, args.folder2)
    print("Average Metrics:", results)

if __name__ == "__main__":
    main()
