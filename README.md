# Multimodal Image Captioning with T5 and EfficientNet

This project builds a compact yet effective image captioning pipeline using:
- 🧠 [**EfficientNet**](https://arxiv.org/abs/1905.11946) for visual feature extraction
- ✍️ [**T5-small**](https://huggingface.co/docs/transformers/en/model_doc/t5) for natural language generation

It combines computer vision and NLP into a streamlined, multimodal architecture — capable of generating human-like captions from image inputs.

---

## 🚀 Project Structure
```
image-captioning-t5/
├── models/                # Captioning model & feature extractor
├── scripts/               # Training, inference, evaluation, feature extraction
├── utils/                 # Dataset class, BLEU, captioning helpers
├── outputs/               # Model checkpoints (not included in repo)
├── notebooks/             # (Optional) end-to-end development notebook
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes weights and data
└── README.md              # This file
```
## Model Pipeline
![Resized_Flowchart](https://github.com/user-attachments/assets/dbfcf846-c370-4708-a9e2-293bd18faf2a)


We train the model on the **Flickr8k dataset**, which contains 8,000 images, each paired with five human-written captions. The visual encoder, **EfficientNet**, is used as a fixed feature extractor with its classification head removed. A **linear projection layer** then maps the 1280-dimensional image features to the 512-dimensional hidden space expected by the T5 encoder. We fine-tune the **T5-small model** end-to-end using these projected image embeddings and corresponding text captions, training with teacher forcing. The model learns to translate high-level visual semantics into fluent natural language descriptions.

---

## 📦 How to Run
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset & Weights
Due to GitHub's file size limitations, the dataset and pretrained weights are **not included in this repository**. Please download them manually:

- 📁 **Flickr8k Dataset**: Download the image folder and captions file from [official source](https://github.com/goodwillyoga/Flickr8k_dataset?tab=readme-ov-file)
- 💾 **Model Weights**: If you wish to use pretrained `.pth`(T5 weights) or `.pkl`(Extracted Image Features using EfficientNet) files, store them locally in `outputs/model_checkpoints/` and `outputs/`, respectively.

### 3. Extract Image Features
```bash
python scripts/extract_features.py
```

### 4. Train the Model
```bash
python scripts/train.py
```

### 5. Evaluate BLEU Score & Test Loss
```bash
python scripts/evaluate.py
```

### 6. Run Inference
```bash
python scripts/infer.py
```

---

## 📊 Example Output

![482098572_e83153b300](https://github.com/user-attachments/assets/95c20904-f268-4c1d-aea6-86983d0eece9)

A man in a red jacket is riding a yellow kayak in the water.

![1827560917_c8d3c5627f](https://github.com/user-attachments/assets/ec730a05-09d1-47d8-9fb0-6bc07b5285eb)

A black and white dog is carrying a stick in its mouth.

---

---

## 📈 Quantitative Results
- **Final Validation Loss**: 1.19
- **Final Test Loss**: 1.20
- **BLEU-4 Score**: 0.2359 on the test set

### 📉 Training and Validation Loss Curve

![train vs val](https://github.com/user-attachments/assets/ac9d5c54-dadd-4f85-b48f-974a58552068)

---


## 🔍 Future Plans
- Real-time video captioning with OpenCV
- T5-base / BLIP2 / CLIP-based captioning
- Multilingual caption generation
- Deploy via Gradio or Streamlit

---

## 📚 Dataset Credits

The **Flickr8k** dataset is provided by the University of Illinois at Urbana-Champaign (UIUC).  
Original source and license details: http://cs.stanford.edu/people/karpathy/deepimagesent/

- Dataset creators: Micah Hodosh, Peter Young, and Julia Hockenmaier  
- Citation:  
  Hodosh, M., Young, P., & Hockenmaier, J. (2013).  
  *Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics*.  
  Journal of Artificial Intelligence Research, 47, 853–899.

---

Made by: [Tanishq Sharma](#)
