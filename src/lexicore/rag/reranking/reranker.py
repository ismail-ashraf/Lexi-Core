
from sentence_transformers import CrossEncoder
import torch


class Reranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
        )

    def rerank(self, query, results, top_k=5):
        documents = results["documents"][0]

        if not documents:
            return {
                **results,
                "reranker_scores": [[]],
            }

        pairs = [(query, document) for document in documents]

        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        ranked_results = {}

        for key, values in results.items():
            if key in {
                "documents", "metadatas", "distances", "ids",
                "embeddings", "uris", "data",
            }:
                if values is not None and len(values) > 0:
                    ranked_results[key] = [
                        [values[0][index] for index in ranked_indices]
                    ]
                else:
                    ranked_results[key] = values
            else:
                ranked_results[key] = values

        ranked_results["reranker_scores"] = [
            [float(scores[index]) for index in ranked_indices]
        ]

        return ranked_results
