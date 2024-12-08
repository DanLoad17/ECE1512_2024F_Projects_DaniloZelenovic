import pandas as pd
import os

# Paths
BASE_DIR = "./NIH_Dataset"
METADATA_FILE = os.path.join(BASE_DIR, "Data_Entry_2017_v2020.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_LABELS_FILE = os.path.join(BASE_DIR, "labels.csv")

# Load metadata
metadata = pd.read_csv(METADATA_FILE)

# Define binary labels: 1 for Pneumonia, 0 otherwise
metadata['Label'] = metadata['Finding Labels'].apply(lambda x: 1 if 'Pneumonia' in x else 0)

# Keep only images that exist in the extracted folder
existing_images = set(os.listdir(IMAGE_DIR))
metadata = metadata[metadata['Image Index'].isin(existing_images)]

# Save the updated labels
metadata[['Image Index', 'Label']].to_csv(OUTPUT_LABELS_FILE, index=False)

print(f"Labels prepared and saved to {OUTPUT_LABELS_FILE}.")
