import torch
from transformers.modeling_outputs import BaseModelOutput

def generate_caption_from_feature(feature, model, tokenizer, device):
    model.eval()

    # Global Average Pooling if needed
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy()

    if feature.shape == (1280, 7, 7):
        feature = feature.mean(axis=(1, 2))

    feature_tensor = torch.tensor(feature, dtype=torch.float).unsqueeze(0).to(device)
    encoder_hidden = model.encoder_fc(feature_tensor).unsqueeze(1)  # Shape: [1, 1, 512]

    input_ids = tokenizer("caption:", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.decoder.generate(
            input_ids=input_ids,
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            max_length=32,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
