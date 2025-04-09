import torch
import pickle
from transformers import T5Tokenizer

from models.caption_generator import CaptionGenerator
from utils.dataset import Flickr8kDataset
from utils.metrics import evaluate_test_loss, evaluate_bleu
from utils.caption_utils import generate_caption_from_feature

from torch.utils.data import DataLoader, random_split

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
with open("data/image_features.pkl", "rb") as f:
    image_features = pickle.load(f)

with open("data/Flickr8k.token.txt", "r") as f:
    lines = f.readlines()

captions_dict = {}
for line in lines:
    parts = line.strip().split("\t")
    img_name, caption = parts[0], parts[1]
    img_name = img_name.split("#")[0]
    captions_dict.setdefault(img_name, []).append(caption)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
dataset = Flickr8kDataset(captions_dict, image_features, tokenizer, device=device)

# Split same as training
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=16)

# Load model
model = CaptionGenerator().to(device)
model.load_state_dict(torch.load("outputs/model_checkpoints/caption_generator.pth", map_location=device))
model.eval()

# Evaluate
evaluate_test_loss(model, test_loader)
evaluate_bleu(model, tokenizer, image_features, captions_dict, test_dataset, 
              lambda feature: generate_caption_from_feature(feature, model, tokenizer, device))
