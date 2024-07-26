import argparse

import hpsv2


def evaluate_images_with_hpsv2(image_path):
    """
    Evaluate images using hpsv2.

    Parameters:
    image_path (str): The path to the directory containing the images.

    Returns:
    None
    """
    hpsv2.evaluate(image_path)


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--image_path",
            type=str,
            default="./output_images",
            help="The path to the directory containing the images",
        )
        return parser.parse_args()

    args = parse_args()

    evaluate_images_with_hpsv2(args.image_path)
