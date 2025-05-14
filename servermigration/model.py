import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttenion(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
 
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=False
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, n_embd, bias=False, dropout=0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = SelfAttenion(n_embd, n_head, bias=bias, dropout=dropout)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





class Tokenizer(nn.Module):
    def __init__(self, n_embd, n_keypoints, max_len):
        super(Tokenizer, self).__init__()

        self.n_keypoints = n_keypoints
        self.n_embd = n_embd
        self.max_len = max_len
        self.fcs_spatial = nn.ModuleList([nn.Linear(3, n_embd) for _ in range(n_keypoints)])
        self.fcs_temporal = nn.ModuleList([nn.Linear(3, n_embd) for _ in range(max_len)])
        self.relu = nn.ReLU()

    def forward(self, keypoints, valid):
        B, T, n_keypoints, _ = keypoints.shape
        assert keypoints.shape[-2] == self.n_keypoints
        assert valid.shape[-1] == self.n_keypoints
        
        embeddings = torch.zeros(B, T+self.n_keypoints, self.n_embd, device=keypoints.device)

        # Spatial tokens: first T positions
        for i in range(self.n_keypoints):
            fc = self.fcs_spatial[i]
            kp_emb = fc(keypoints[:, :, i, :])  # (B, T, n_embd)
            kp_emb *= valid[:, :, i].unsqueeze(-1)
            embeddings[:, :T, :] += kp_emb

        spatial_valid_counts = valid.sum(dim=-1, keepdim=True)  # (B, T, 1)
        spatial_scale = (self.n_keypoints + 1) / (spatial_valid_counts + 1)
        embeddings[:, :T, :] *= spatial_scale

        # Temporal tokens: next n_keypoints positions
        for i in range(T):
            fc = self.fcs_temporal[i]
            kp_emb = fc(keypoints[:, i, :, :])  # (B, n_keypoints, n_embd)
            kp_emb *= valid[:, i, :].unsqueeze(-1)
            embeddings[:, T:, :] += kp_emb

        temporal_valid_counts = valid.sum(dim=-2, keepdim=True)  # (B, 1, n_keypoints)
        temporal_scale = (self.max_len + 1) / (temporal_valid_counts + 1)
        embeddings[:, T:, :] *= temporal_scale.transpose(1, 2)  # match dims

        return self.relu(embeddings) 

class WordProjection(nn.Module):
    def __init__(self, word_embd, n_embd):
        super(WordProjection, self).__init__()

        self.lm = nn.Linear(word_embd, n_embd)
        self.mlp_1 = MLP(n_embd, bias=True)
        self.ln_1 = LayerNorm(n_embd, bias=True)
        self.mlp_2 = MLP(n_embd, bias=True)
        self.ln_2 = LayerNorm(n_embd, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.lm(x)
        x = x + self.gelu(self.mlp_1(self.ln_1(x)))
        x = x + self.gelu(self.mlp_2(self.ln_2(x)))
        return x


class SLR(nn.Module):

    def __init__(
            self,
            n_embd=512, 
            n_cls_dict={'asl_citizen': 2305},
            n_head=8, 
            n_layer=10, 
            n_keypoints=63,
            dropout=0.0, 
            max_len=64,
            bias=True
        ):
        super(SLR, self).__init__()
        
        self.n_embd = n_embd
        self.max_len = max_len
        self.n_keypoints = n_keypoints

        self.cls_token = nn.Parameter(torch.rand(1, 1, n_embd))
        self.tokenizer = Tokenizer(n_embd, n_keypoints, max_len)
        self.pos_embd = nn.Embedding(max_len+n_keypoints, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, bias=True, dropout=dropout) 
            for _ in range(n_layer)
        ])

        self.layernorm = LayerNorm(n_embd, bias=False)
        self.heads = nn.ModuleDict({
            name: nn.Linear(n_embd, n_classes, bias=False)
            for name, n_classes in n_cls_dict.items()
        })
        self.default_name = next(iter(n_cls_dict))

        #self.word_proj = WordProjection(word_embd, n_embd)
        
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, keypoints, valid_keypoints, dataset_name=None):
        
        batch_size = len(keypoints)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        
        tok_emb = self.tokenizer(keypoints, valid_keypoints)  # shape (B, T+n_keypoints, n_embd)
        
        
        pos = torch.arange(0, tok_emb.size(1), dtype=torch.long, device=keypoints.device)
        pos_emb = self.pos_embd(pos)
        
        x = torch.cat([cls_token, tok_emb + pos_emb], dim=1)  # shape (B, T+n_keypoints+1, n_embd)
        
        for block in self.blocks:
            x = block(x)
        
        
        x = self.layernorm(x)
        output = x[:, 0]
        
        return output
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
