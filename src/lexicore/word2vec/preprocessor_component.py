from lexicore.tokenization.bpe_tokenizer import BPETokenizer, BPEVocabulary
from lexicore.word2vec.context import ContextTargetGenerator
from lexicore.word2vec.corpus_data_loader import load_corpus
from pathlib import Path


class Word2VecPreprocessor:
    def __init__(self, tokenizer, context_generator):
        self.tokenizer = tokenizer
        self.context_generator = context_generator
        self.vocabulary = None

    def prepare(self, corpus: list[str]):

        full_text = "\n".join(corpus)

        self.tokenizer.fit(full_text)

        all_tokens = []

        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)

        self.vocabulary = BPEVocabulary(all_tokens)

        all_pairs = []

        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.vocabulary.encode(tokens)

            pairs = self.context_generator.generate(token_ids)

            all_pairs.extend(pairs)

        return all_pairs


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "word2vec_data.txt"

corpus = load_corpus(DATA_PATH)

tokenizer = BPETokenizer(
    vocab_size=1000
)

context = ContextTargetGenerator(
    context_window=2
)

preprocessor = Word2VecPreprocessor(
    tokenizer,
    context
)

pairs = preprocessor.prepare(corpus)

vocab = preprocessor.vocabulary