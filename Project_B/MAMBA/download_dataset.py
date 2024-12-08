import os
import urllib.request
import tarfile

# Define the directory where images will be stored
BASE_DIR = "./NIH_Dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    ]

# Download and extract each file
for idx, link in enumerate(links):
    fn = f'images_{idx+1:02d}.tar.gz'
    filepath = os.path.join(BASE_DIR, fn)
    print(f"Downloading {fn}...")
    urllib.request.urlretrieve(link, filepath)
    print(f"Extracting {fn}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=IMAGE_DIR)
    os.remove(filepath)  # Remove the tar.gz file to save space
    print(f"{fn} processed.")

print("Download and extraction complete.")
