import torch
import torch.nn as nn
import torch.nn.functional as f

class MultiHeadAttention(nn.Module):
    def __init__(self,dim_model,num_heads):
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = self.dim_model // self.num_heads

        self.query = nn.Linear(dim_model , dim_model)

        self.key = nn.Linear(dim_model,dim_model)

        self.value = nn.Linear(dim_model,dim_model)

        self.out = nn.Linear(dim_model,dim_model)

    def forward(self,x,padding_mask):
        b,s,d = x.shape
        # shape = [batch,token,dim_model]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(b,s,self.num_heads , self.head_dim)
        k = k.reshape(b,s,self.num_heads,self.head_dim)
        v = v.reshape(b,s,self.num_heads,self.head_dim)
        # shape -> [batch,token,num_heads,heads_dim]

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        #shape -> [batch,num_heads,token,heads_dim]

        attention_score = torch.matmul(q,k.transpose(-2,-1))
        # k.transpose(-2,-1) -> [batch,num_heads,heads_dim,token]
        # attention_score    -> [batch,num_heads,token,token]

        attention_score = attention_score / torch.sqrt(
            torch.tensor(self.head_dim , dtype=torch.float32)
        )

        attention_score = attention_score.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf")
        )
        weights_attention = f.softmax(attention_score,dim=-1)

        w_v = torch.matmul(weights_attention,v)
        w_v = w_v.transpose(1,2)
        # shape -> [batch,token,num_heads,head_dim]
        w_v = torch.flatten(w_v,start_dim=2)
        # shape -> [batch,token,model_dim]

        out = self.out(w_v)

        return out