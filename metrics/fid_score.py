from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats


def calculate_fid_score(image_path):
    """
    Calculate the FID score for a given path of images.

    Parameters:
    image_path (str): The path to the directory containing the generated images.

    Returns:
    float: The calculated FID score.
    """
    fid, _ = calculate_fid(image_path, get_coco_fid_stats())
    return fid


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python calculate_fid_score.py <image_path>")
    else:
        image_path = sys.argv[1]
        print(image_path)
        fid = calculate_fid_score(image_path)
        print(f"FID score: {fid}")
