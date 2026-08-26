import torch


class NegativeSampling:
    def __init__(self , word_freq : dict[str,int] , word2idx : dict[str,int] , negative_sampler : int , power : float = 0.75):
        self.word_freq = word_freq
        self.word2idx = word2idx
        self.power = power
        self.negative_sampler = negative_sampler
        self.probabilities = self.build_distribution()

    def build_distribution(self):
        vocab_size = len(self.word2idx)

        probabilities = torch.zeros(vocab_size)

        for word,freq in self.word_freq.items():
            idx = self.word2idx[word]
            probabilities[idx] = freq**self.power

        probabilities = probabilities / probabilities.sum()

        return probabilities

    def sample(self, target_id: int, context_id: int):

        negatives = []

        while len(negatives) < self.negative_sampler:

            sampled_id = torch.multinomial(
                self.probabilities,
                num_samples=1
            ).item()

            if sampled_id != target_id and sampled_id != context_id and sampled_id not in negatives:
                negatives.append(sampled_id)

        return negatives