import torch
import torch.nn.functional as f


class EmbeddingSimilarity:
    def __init__(self , model,vocab):
        self.model = model
        self.vocab = vocab

    def most_similar(self,word,top_k:int=5):
        word_id = self.vocab.word2idx.get(word)
        if word_id is None:
            raise ValueError(
                f"This word is not found in the vocabulary: {word}"
            )

        query_vector = self.model.input_embeddings.weight[word_id]

        embeddings = self.model.input_embeddings.weight

        cosine = f.cosine_similarity(query_vector,embeddings,dim=1)

        values,indices = torch.topk(cosine,top_k+1)


        results = []
        for score,idx in zip(values,indices):
            idx = idx.item()

            if idx == word_id:
                continue

            word = self.vocab.idx2word[idx]

            results.append((word,score.item()))

            if len(results) == top_k:
                break

        return results






