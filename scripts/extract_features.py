import os
import torch
import pickle
from models.efficientnet_extractor import load_efficientnet, extract_features
from tqdm import tqdm

# Paths
DATASET_PATH = "data/Flicker8k_Dataset"
CAPTIONS_FILE = "data/Flickr8k.token.txt"
FEATURES_OUTPUT = "outputs/image_features.pkl"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_efficientnet(device)

# Load captions to get image list
captions_dict = {}
with open(CAPTIONS_FILE, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        img_name, caption = parts[0], parts[1]
        img_name = img_name.split("#")[0]
        captions_dict.setdefault(img_name, []).append(caption)

# Extract and save features
image_features = {}
image_list = list(captions_dict.keys())

for img_name in tqdm(image_list, desc="Extracting features"):
    img_path = os.path.join(DATASET_PATH, img_name)
    if os.path.exists(img_path):
        features = extract_features(img_path, model, device)
        if features is not None:
            image_features[img_name] = features

# Save features
with open(FEATURES_OUTPUT, "wb") as f:
    pickle.dump(image_features, f)

print(f"Extracted features for {len(image_features)} images and saved to {FEATURES_OUTPUT}")
