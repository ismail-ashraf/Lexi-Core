
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lexicore.configrations.config import settings


model_path = Path(settings.BERT_MODEL_PATH)
output_dir = model_path / "onnx"
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForSequenceClassification.from_pretrained(
    str(model_path)
)

model.eval()

sample_text = "This Loan Agreement is between a bank and a borrower."

inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=settings.MAX_LENGTH,
)

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    str(output_dir / "model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    dynamo=False,
)

tokenizer.save_pretrained(output_dir)

print(f"ONNX model exported to: {output_dir}")
