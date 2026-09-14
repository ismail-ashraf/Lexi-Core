import torch
import torch.nn as nn
import torch.optim as optim
from lexicore.word2vec.skipgram import SkipGramNN
from lexicore.word2vec.negative_sampling import NegativeSampling
from lexicore.word2vec.similarity import EmbeddingSimilarity
from lexicore.training.data_preparation import prepare_data
from lexicore.word2vec.dataset import create_dataloader
from lexicore.word2vec.preprocessor_component import Word2VecPreprocessor
from lexicore.word2vec.context import ContextTargetGenerator
from lexicore.tokenization.bpe_tokenizer import BPETokenizer



class SkipGramTrain:
    def __init__(self , model , dataloader,negative_sampler,optimizer , num_epochs , criterion):
        self.model = model
        self.dataloader = dataloader
        self.negative_sampler = negative_sampler
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.criterion = criterion

    def train(self):

        for epoch in range(self.num_epochs):
            self.model.train()

            total_train_loss = 0.0

            total_correct = 0.0
            total_samples = 0.0

            for target_ids,context_ids in self.dataloader:
                self.optimizer.zero_grad()

                negative_ids = []

                for target_id, context_id in zip(target_ids, context_ids):
                    negatives = self.negative_sampler.sample(
                        target_id.item(),
                        context_id.item()
                    )

                    negative_ids.append(negatives)

                negative_ids = torch.tensor(
                    negative_ids,
                    dtype=torch.long
                )

                positive_scores, negative_scores = self.model(
                    target_ids,
                    context_ids,
                    negative_ids
                )

                positive_labels = torch.ones_like(positive_scores)
                negative_labels = torch.zeros_like(negative_scores)

                positive_loss = self.criterion(
                    positive_scores,
                    positive_labels
                )

                negative_loss = self.criterion(
                    negative_scores,
                    negative_labels
                )

                loss = positive_loss + negative_loss

                loss.backward()

                self.optimizer.step()

                total_train_loss += loss.item()

                positive_predictions = (
                        torch.sigmoid(positive_scores) >= 0.5
                )

                negative_predictions = (
                        torch.sigmoid(negative_scores) < 0.5
                )

                correct_positive = (
                    positive_predictions.sum().item()
                )

                correct_negative = (
                    negative_predictions.sum().item()
                )

                total_correct += (
                        correct_positive +
                        correct_negative
                )

                total_samples += (
                        positive_scores.numel() +
                        negative_scores.numel()
                )
            average_loss = total_train_loss / len(self.dataloader)

            accuracy = (
                    total_correct /
                    total_samples
            )
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} "
                f"Loss: {average_loss:.4f}"
                f"| Accuracy: {accuracy:.4f}"
            )


X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

tokenizer = BPETokenizer(vocab_size=1000)

context = ContextTargetGenerator(context_window=2)

preprocessor = Word2VecPreprocessor(
    tokenizer,
    context
)

pairs = preprocessor.prepare(X_train.tolist())

vocab = preprocessor.vocabulary

dataloader = create_dataloader(
    pairs,
    batch_size=32
)

torch.manual_seed(42)
model = SkipGramNN(
    vocab_size=len(vocab.word2idx),
    embedding_dim=128
)

negative_sampler = NegativeSampling(
    word_freq=vocab.word_freq,
    word2idx=vocab.word2idx,
    negative_sampler=2
)
optimizer = optim.AdamW(model.parameters() , lr=0.003)
criterion = nn.BCEWithLogitsLoss()

train = SkipGramTrain(
    model=model,
    dataloader=dataloader,
    negative_sampler=negative_sampler,
    optimizer=optimizer,
    num_epochs=10,
    criterion=criterion
)
#
# train.train()


# embeddings = model.input_embeddings.weight.detach()
# print(embeddings.shape)
#
# similarity = EmbeddingSimilarity(
#     model,
#     vocab
# )
#
# # results = similarity.most_similar('employees</w>',top_k=5)
# # for word,score in results:
# #     print(f"{word} | {score:.2f}")
# #
# # # print(vocab.word2idx)
# #
# # print(
# #     similarity.most_similar(
# #         "employees",
# #         top_k=10
# #     )
# # )
# # print(vocab.word2idx)
# print(tokenizer.tokenize("employees"))