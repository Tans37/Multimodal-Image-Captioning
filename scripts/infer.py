import os
import pickle
import torch
from transformers import T5Tokenizer
from models.caption_generator import CaptionGenerator
from utils.caption_utils import generate_caption_from_feature

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = CaptionGenerator().to(device)
model.load_state_dict(torch.load("outputs/model_checkpoints/caption_generator.pth", map_location=device))
model.eval()

# Load features
with open("data/image_features.pkl", "rb") as f:
    image_features = pickle.load(f)

# Inference on a few examples
sample_keys = list(image_features.keys())[:5]
for img_name in sample_keys:
    feature = image_features[img_name]
    caption = generate_caption_from_feature(feature, model, tokenizer, device)
    print(f"{img_name} => {caption}")
