import json
import torch
from transformers import AutoTokenizer,BertForSequenceClassification
from lexicore.configrations.config import settings

MODEL_DIR = settings.BERT_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


with open(f"{MODEL_DIR}/label_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}

def predict_contract(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = probs.argmax().item()

    return {
        "label": id2label[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
    }

result = predict_contract("""
The borrower shall repay the principal amount in monthly
installments together with interest at the agreed rate.
""")

print(result)