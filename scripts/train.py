import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer
import pickle
import os

from models.caption_generator import CaptionGenerator
from utils.dataset import Flickr8kDataset

# Load resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("outputs/image_features.pkl", "rb") as f:
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

# Split the dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model
model = CaptionGenerator().to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        image_features, input_ids, attention_mask = batch
        optimizer.zero_grad()
        loss, _ = model(image_features, input_ids, attention_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            image_features, input_ids, attention_mask = batch
            loss, _ = model(image_features, input_ids, attention_mask)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save model
os.makedirs("outputs/model_checkpoints", exist_ok=True)
torch.save(model.state_dict(), "outputs/model_checkpoints/caption_generator.pth")
print("Model saved.")
