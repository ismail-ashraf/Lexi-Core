import torch

checkpoint_path = r"C:\Users\abdel\Downloads\word2vec_checkpoint.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

if checkpoint is None:
    raise FileNotFoundError("File Not Found")

print("CheckPoint loaded successfully")

print(f"Saved Epochs {checkpoint['epoch']}")

print(f"Saved Loss {checkpoint['loss']:.2f}")

print(f"Saved Accuracy {checkpoint['accuracy']:.2f}")


state_dict = checkpoint['model_state_dict'] # -> state for all parameters in the model

embedding_weights = state_dict['input_embeddings.weight'] # -> weights

vocab_size,embedding_dim = embedding_weights.shape

print("\nEmbedding information")
print("---------------------")

print(f"Vocabulary size: {vocab_size}")

print(f"Embedding dimension: {embedding_dim}")

print(f"Embedding shape: {embedding_weights.shape}")

print("\nLinear algebra")
print("--------------")

rank = torch.linalg.matrix_rank(
    embedding_weights
).item()

print(f"Rank: {rank}")


print("\nSingular values")
print("----------------")

svd = torch.linalg.svdvals(
    embedding_weights
)

print("singular values:")
print(svd)
print('SVD Shape:')
print(svd.shape)


print("------------------")
print("total energy for single vector:")
energy = svd**2
ratio = energy / energy.sum()
print(ratio)

print('-------------')
print("cumulative")
cumulative = ratio.cumsum(dim=0)
print(cumulative[:10])

indices = (cumulative >= 0.90).nonzero(as_tuple=True)[0][0].item()
print("All dimensions needed:")
print(indices+1)