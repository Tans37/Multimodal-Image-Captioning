import torch
from PIL import Image
from torchvision import transforms
import os

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_efficientnet(device):
    effnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                            'nvidia_efficientnet_b0',
                            pretrained=True)
    effnet = effnet.to(device)
    effnet.eval()
    return torch.nn.Sequential(*list(effnet.children())[:-1])

def extract_features(image_path, model, device):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(image).squeeze().cpu().numpy()

        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
