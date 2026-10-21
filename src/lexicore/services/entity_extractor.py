from gliner import GLiNER
from lexicore.configrations.config import settings


class NERService:
    def __init__(self):
        self.model = GLiNER.from_pretrained(
            str(settings.NER_ONNX_MODEL_PATH),
            load_onnx_model=True,
            onnx_model_file="model.onnx",
            local_files_only=True,
        )

    def predict(self, text: str) -> list[dict]:
        entities = self.model.predict_entities(
            text,
            labels=settings.legal_ner_labels,
            threshold=settings.ner_threshold,
        )

        return [
            {
                "text": entity["text"],
                "label": entity["label"],
                "score": round(float(entity["score"]), 4),
                "start": entity.get("start"),
                "end": entity.get("end"),
            }
            for entity in entities
        ]
