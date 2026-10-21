import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
from lexicore.configrations.config import settings
from pathlib import Path


class DocumentClassifier:
    def __init__(self):
        model_path = Path(settings.BERT_ONNX_MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        self.config = AutoConfig.from_pretrained(str(model_path))

        onnx_file = next(model_path.glob("*.onnx"), None)
        if not onnx_file:
            raise FileNotFoundError(f"No ONNX file found in {model_path}")

        self.ort_session = ort.InferenceSession(
            str(onnx_file),
            providers=["CPUExecutionProvider"]
        )

        self.model_inputs = [i.name for i in self.ort_session.get_inputs()]
        self.max_length = settings.MAX_LENGTH

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="np"
        )

        ort_inputs = {}
        for input_name in self.model_inputs:
            if input_name in inputs:
                ort_inputs[input_name] = inputs[input_name].astype(np.int64)

        outputs = self.ort_session.run(None, ort_inputs)
        logits = outputs[0]

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        predicted_id = int(np.argmax(probabilities, axis=-1)[0])
        confidence = float(probabilities[0][predicted_id])

        label = self.config.id2label.get(predicted_id, str(predicted_id))

        return {
            "label": label,
            "confidence": round(confidence, 4),
        }