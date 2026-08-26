import torch.nn as nn


class SkipGramNN(nn.Module):
    def __init__(self , vocab_size:int , embedding_dim:int):
        super().__init__()

        self.input_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.output_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )


    def forward(self,target_id , context_id , negative_context_ids):
        target_vector = self.input_embeddings(target_id)

        context_vector = self.output_embeddings(context_id)

        negative_vectors = self.output_embeddings(negative_context_ids)

        positive_scores = (target_vector * context_vector).sum(dim=1)

        target_vector = target_vector.unsqueeze(1)

        negative_scores = (target_vector * negative_vectors).sum(dim=2)

        return positive_scores,negative_scores