import re
from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merges = []

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    @staticmethod
    def initialize_word_freqs(words: list[str]) -> dict[tuple[str, ...], int]:
        word_counts = Counter(words)
        word_freqs = {}

        for word, count in word_counts.items():
            tokens = tuple(list(word[:-1]) + [word[-1] + "</w>"]) if word else ()
            if tokens:
                word_freqs[tokens] = count

        return word_freqs

    @staticmethod
    def get_pair_counts(word_freqs: dict[tuple[str, ...], int]) -> Counter:
        pair_counts = Counter()

        for word_tokens, freq in word_freqs.items():
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pair_counts[pair] += freq

        return pair_counts

    @staticmethod
    def get_best_pair(pair_counts: Counter) -> tuple[str, str] | None:
        if not pair_counts:
            return None

        return max(pair_counts, key=lambda p: (pair_counts[p], p))

    @staticmethod
    def merge_pair(
        word_freqs: dict[tuple[str, ...], int], best_pair: tuple[str, str]
    ) -> dict[tuple[str, ...], int]:
        new_word_freqs = {}
        bigram = best_pair

        for word_tokens, freq in word_freqs.items():
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if (
                    i < len(word_tokens) - 1
                    and word_tokens[i] == bigram[0]
                    and word_tokens[i + 1] == bigram[1]
                ):
                    new_tokens.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1

            new_word_freqs[tuple(new_tokens)] = freq

        return new_word_freqs

    def fit(self, text: str):

        self.merges = []
        words = self.pre_tokenize(text)
        word_freqs = self.initialize_word_freqs(words)

        self.vocab = set()
        for word_tokens in word_freqs.keys():
            for token in word_tokens:
                self.vocab.add(token)

        while len(self.vocab) < self.vocab_size:
            pair_counts = self.get_pair_counts(word_freqs)
            best_pair = self.get_best_pair(pair_counts)

            if not best_pair:
                break

            self.merges.append(best_pair)
            word_freqs = self.merge_pair(word_freqs, best_pair)

            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add(merged_token)

        return self

    def tokenize_word(self, word: str) -> list[str]:
        if not word:
            return []

        tokens = list(word[:-1]) + [word[-1] + "</w>"]

        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge[0]
                    and tokens[i + 1] == merge[1]
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def tokenize(self, text: str) -> list[str]:
        words = self.pre_tokenize(text)
        tokens = []
        for word in words:
            tokens.extend(self.tokenize_word(word))
        return tokens


class BPEVocabulary:

    def __init__(self, tokens: list[str]):
        self.word_freq = Counter(tokens)

        self.word2idx = {
            token: idx
            for idx, token in enumerate(
                self.word_freq.keys()
            )
        }

        self.idx2word = {
            idx: token
            for token, idx in self.word2idx.items()
        }

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens: list[str]) -> list[int]:

        return [
            self.word2idx[token]
            for token in tokens
        ]

    def decode(self, ids: list[int]) -> list[str]:

        return [
            self.idx2word[idx]
            for idx in ids
        ]

    def add_special_token(self, token: str):
        if token not in self.word2idx:
            idx = len(self.word2idx)

            self.word2idx[token] = idx
            self.idx2word[idx] = token

        return self.word2idx[token]


