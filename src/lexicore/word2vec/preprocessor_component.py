from lexicore.tokenization.bpe_tokenizer import BPETokenizer, BPEVocabulary
from lexicore.word2vec.context import ContextTargetGenerator
from lexicore.training.data_preparation import prepare_data

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

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

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

pairs = preprocessor.prepare(X_train.tolist())

vocab = preprocessor.vocabulary