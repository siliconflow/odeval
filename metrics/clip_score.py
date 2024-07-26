from glob import glob

import pandas as pd
from T2IBenchmark import calculate_clip_score


def odeval_clip_score(image_dir, csv_path):
    """
    Calculate the CLIP score for given images and captions.

    Parameters:
    image_dir (str): The directory containing the images.
    csv_path (str): The path to the CSV file containing the captions.

    Returns:
    float: The calculated CLIP score.
    """
    try:
        cat_paths = sorted(
            glob(f"{image_dir}/*.png"),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        print(f"Number of image files: {len(cat_paths)}")

        captions_df = pd.read_csv(csv_path)
        print(f"Number of captions read: {len(captions_df)}")

        if len(cat_paths) != len(captions_df):
            raise ValueError(
                "The number of images does not match the number of captions."
            )

        captions_mapping = {
            cat_paths[i]: captions_df.iloc[i, 1] for i in range(len(cat_paths))
        }
        print("Captions mapping created successfully.")

        clip_score = calculate_clip_score(cat_paths, captions_mapping=captions_mapping)
        return clip_score

    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: No data found in the CSV file.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python calculate_clip_score.py <image_dir> <csv_path>")
    else:
        image_dir = sys.argv[1]
        csv_path = sys.argv[2]
        score = odeval_clip_score(image_dir, csv_path)
        if score is not None:
            print(f"CLIP Score: {score}")
