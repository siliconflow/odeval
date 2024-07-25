import pandas as pd
from glob import glob
from T2IBenchmark import calculate_clip_score

cat_paths = sorted(glob('/home/lixiang/data/fid_kolors_nexfort/*.png'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

print(f"Number of image files: {len(cat_paths)}")

csv_path = '/home/lixiang/odeval/MS-COCO_val2014_30k_captions.csv'
try:
    captions_df = pd.read_csv(csv_path)
    print(f"Number of captions read: {len(captions_df)}")

    if len(cat_paths) != len(captions_df):
        print("Error: The number of images does not match the number of captions.")
    else:
        captions_mapping = {cat_paths[i]: captions_df.iloc[i, 1] for i in range(len(cat_paths))}
        print("Captions mapping created successfully.")

        clip_score = calculate_clip_score(cat_paths, captions_mapping=captions_mapping)
        print(f"CLIP Score: {clip_score}")

except FileNotFoundError:
    print(f"Error: The file {csv_path} was not found.")
except pd.errors.EmptyDataError:
    print("Error: No data found in the CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")
