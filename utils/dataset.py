import torch
from torch.utils.data import Dataset
import numpy as np
import random

class Flickr8kDataset(Dataset):
    def __init__(self, captions_dict, image_features, tokenizer, max_length=32, device='cpu'):
        self.image_features = {k: v for k, v in image_features.items() if k in captions_dict}
        self.captions_dict = captions_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_names = list(self.image_features.keys())
        self.device = device

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        feature = self.image_features[img_name]

        # Convert to numpy if needed
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        feature = np.array(feature)

        # Global average pool if shape is (1280, 7, 7)
        if feature.shape == (1280, 7, 7):
            feature = feature.mean(axis=(1, 2))

        assert feature.shape == (1280,), f"Feature shape mismatch: {feature.shape}"
        feature = torch.tensor(feature, dtype=torch.float).to(self.device)

        # Randomly choose one caption
        caption = random.choice(self.captions_dict[img_name])
        inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return feature, inputs.input_ids.squeeze(0).to(self.device), inputs.attention_mask.squeeze(0).to(self.device)
