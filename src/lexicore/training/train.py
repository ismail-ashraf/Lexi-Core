import torch
from lexicore.tokenization.bpe_tokenizer import BPETokenizer
from lexicore.word2vec.context import ContextTargetGenerator
from lexicore.word2vec.preprocessor_component import Word2VecPreprocessor
from lexicore.training.data_preparation import prepare_data
from lexicore.training.transformer_preprocessor import TransformerPreprocessor, TransformerDataset
from lexicore.word2vec.analyze_embeddings import checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from lexicore.core.model import TransformerEncoderModel

# =========================
# 0. Setup GPU / Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
X_train_encoded = preprocessor.encode_dataset(X_train)
X_val_encoded = preprocessor.encode_dataset(X_val)
X_test_encoded = preprocessor.encode_dataset(X_test)


# =========================
# 7. Class Dataset & DataLoader
# =========================
train_dataset = TransformerDataset(X_train_encoded, y_train)
val_dataset = TransformerDataset(X_val_encoded, y_val)
test_dataset = TransformerDataset(X_test_encoded, y_test)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# =========================
# 8. Model Initialization & Moving to GPU
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
    model.embedding.weight[:word2vec_vocab_size] = embedding_weights.detach().cpu()

model = model.to(device)


# =========================
# 9. Training Loops with GPU Support
# =========================

def train(model, dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted = pred.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_train_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Epoch: {epoch + 1:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")


def validation(model, dataloader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            loss = criterion(pred, labels)

            total_val_loss += loss.item()
            predicted = pred.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_val_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Validation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")


def test(model, dataloader, criterion, device):
    model.eval()
    total_test_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            loss = criterion(pred, labels)

            total_test_loss += loss.item()
            predicted = pred.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_test_loss / len(dataloader)
        accuracy = total_correct / total_samples

        print(f"Test | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")


# =========================
# 10. Start Training
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003)

train(
    model=model,
    dataloader=train_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=10,
    device=device
)

validation(
    model=model,
    dataloader=val_dataloader,
    criterion=criterion,
    device=device
)