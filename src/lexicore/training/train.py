import torch
from lexicore.tokenization.bpe_tokenizer import BPETokenizer
from lexicore.word2vec.context import ContextTargetGenerator
from lexicore.word2vec.preprocessor_component import Word2VecPreprocessor
from lexicore.training.data_preparation import prepare_data
from lexicore.training.transformer_preprocessor import TransformerPreprocessor,TransformerDataset
from lexicore.word2vec.analyze_embeddings import checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from lexicore.core.model import TransformerEncoderModel


# =========================
# 1. Load Word2Vec embeddings
# =========================

state_dict = checkpoint["model_state_dict"]

embedding_weights = state_dict["input_embeddings.weight"]

vocab_size, embedding_dim = embedding_weights.shape

print("Embedding shape:", embedding_weights.shape)


# =========================
# 2. Prepare Transformer data
# =========================

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()


# =========================
# 3. Rebuild the exact BPE vocabulary
# =========================

tokenizer = BPETokenizer(vocab_size=1000)

context = ContextTargetGenerator(context_window=2)

word2vec_preprocessor = Word2VecPreprocessor(
    tokenizer,
    context
)

pairs = word2vec_preprocessor.prepare(
    X_train.tolist()
)

vocabulary = word2vec_preprocessor.vocabulary

print("Vocabulary size:", len(vocabulary))


# =========================
# 4. Add PAD token
# =========================

pad_id = vocabulary.add_special_token("<PAD>")

print("PAD ID:", pad_id)
print("Final vocabulary size:", len(vocabulary))

# =========================
# 5. Build Tokenizer Preprocessor
# =========================
MAX_SEQ_LEN = 414

preprocessor = TransformerPreprocessor(
    tokenizer=tokenizer,
    vocabulary=vocabulary,
    max_seq_len=MAX_SEQ_LEN
)

# =========================
# 6. Encode Dataset
# =========================

X_train_encoded = preprocessor.encode_dataset(
    X_train
)

X_val_encoded = preprocessor.encode_dataset(
    X_val
)

X_test_encoded = preprocessor.encode_dataset(
    X_test
)

# =========================
# 6. Class Dataset
# =========================

train_dataset = TransformerDataset(
    X_train_encoded,
    y_train
)

val_dataset = TransformerDataset(
    X_val_encoded,
    y_val
)

test_dataset = TransformerDataset(
    X_test_encoded,
    y_test
)

# =========================
# 6. DataLoader
# =========================

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# =========================
# 6. Adding the embeddings of word2vec to transformer
# =========================

model = TransformerEncoderModel(
    max_seq_len=MAX_SEQ_LEN,
    dim_model=embedding_dim,
    num_heads=8,
    num_layers=3,
    vocab_size=len(vocabulary),
    pad_id=pad_id
)

word2vec_vocab_size = embedding_weights.shape[0]

model.embedding = nn.Embedding(
    len(vocabulary),
    embedding_dim,
    padding_idx=pad_id
)

with torch.no_grad():
    model.embedding.weight[:word2vec_vocab_size] = embedding_weights


# =========================
# 6. Training Loop
# =========================


def train(model,dataloader,criterion,optimizer,num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct = 0
        total_samples = 0

        for logits, labels in dataloader:
            optimizer.zero_grad()

            pred = model(logits)

            loss = criterion(pred, labels)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

            predicted = pred.argmax(dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_train_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Epoch:{epoch + 1}|"
              f"Loss: {avg_loss}|"
              f"Accuracy: {accuracy}")



def validation(model,dataloader,criterion):
    model.eval()
    total_val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for logits, labels in dataloader:
            pred = model(logits)

            loss = criterion(pred, labels)

            total_val_loss += loss.item()

            predicted = pred.argmax(dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_val_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Loss:{avg_loss}|"
              f"Accuracy:{accuracy}|")


def test(model,dataloader,criterion):
    model.eval()
    total_test_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for logits, labels in dataloader:
            pred = model(logits)

            loss = criterion(pred, labels)

            total_test_loss += loss.item()

            predicted = pred.argmax(dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_test_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Loss:{avg_loss}|"
              f"Accuracy:{accuracy}|")


train(
    model=model,
    dataloader=train_dataloader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.AdamW(model.parameters(),lr=0.0003),
    num_epochs=10
)


validation(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    dataloader=val_dataloader
)