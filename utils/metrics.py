import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate_test_loss(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img_feat, input_ids, attention_mask in test_loader:
            loss, _ = model(img_feat, input_ids, attention_mask)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f" Average Test Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_bleu(model, tokenizer, image_features, captions_dict, test_dataset, generate_fn):
    references = []
    hypotheses = []

    test_img_names = [test_dataset.dataset.image_names[i] for i in test_dataset.indices]

    for img_name in test_img_names:
        gt_captions = captions_dict[img_name]
        ref_tokens = [tokenizer.tokenize(cap.lower()) for cap in gt_captions]
        references.append(ref_tokens)

        feature = image_features[img_name]
        predicted_caption = generate_fn(feature)
        hyp_tokens = tokenizer.tokenize(predicted_caption.lower())
        hypotheses.append(hyp_tokens)

    smoothing = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    print(f" BLEU-4 Score on Test Set: {bleu_score:.4f}")
    return bleu_score
