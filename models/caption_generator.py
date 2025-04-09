import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class CaptionGenerator(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, model_name="t5-small"):
        super(CaptionGenerator, self).__init__()
        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.decoder = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, image_features, input_ids, attention_mask):
        # Transform image features to match decoder hidden size
        image_features = self.encoder_fc(image_features).unsqueeze(1)  # [B, 1, 512]

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(image_features,),
            labels=input_ids  # Teacher forcing
        )
        return outputs.loss, outputs.logits
