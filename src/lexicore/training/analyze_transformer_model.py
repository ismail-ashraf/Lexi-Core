import torch
from lexicore.configrations.config import settings

checkpoint_path = settings.TRANSFORMER_MODEL_PATH

data = torch.load(
    checkpoint_path,
    map_location='cpu'
)

if data is None:
    raise FileNotFoundError("File Not Found")

print("CheckPoint Loaded Successfully")

# if isinstance(data,dict):
#     for key,value in data.items():
#         print(key,value.shape)

embedding_weight = data['embedding.weight']

print(embedding_weight.shape)

rank = torch.linalg.matrix_rank(
    embedding_weight
)

print(f"rank : {rank}")

svd = torch.linalg.svd(
    embedding_weight
)

print(f"SVD :{svd}")