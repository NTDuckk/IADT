import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .subject_tokenizer import InversionNetwork

class FP16LayerNorm(nn.LayerNorm):
    """fp16-safe LayerNorm (CLIP-style)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)

class AttrTokenBlock(nn.Module):
    """A small cross-attention + FFN block (IADT-style) to update learnable queries."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = FP16LayerNorm(embed_dim)
        self.norm_kv = FP16LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        hidden = int(embed_dim * mlp_ratio)
        self.norm2 = FP16LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B,Q,D), kv: (B,M,D)
        q_ln = self.norm_q(q)
        kv_ln = self.norm_kv(kv)
        attn_out, _ = self.attn(q_ln, kv_ln, kv_ln, need_weights=False)
        q = q + self.drop(attn_out)
        q = q + self.mlp(self.norm2(q))
        return q

def orthogonal_token_loss(tokens: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Orthogonality regularization on attribute tokens (proxy for IADT L_ortho)."""
    if tokens is None:
        return None
    B, K, D = tokens.shape
    t = F.normalize(tokens, dim=-1)
    gram = torch.matmul(t, t.transpose(1, 2))  # (B,K,K)
    I = torch.eye(K, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
    return ((gram - I) ** 2).mean()

class AttributeTokenizer(nn.Module):
    """IADT-style attribute tokenization.

    - learnable queries X (num_queries=64)
    - cross-attn from queries to patch tokens (num_layers)
    - top-k filtering by cosine similarity with global embedding v (topk=16)
    - mapping network f_A to CLIP token embedding space (3-layer MLP + BN)
    """
    def __init__(
        self,
        embed_dim: int = 512,
        num_queries: int = 64,
        topk: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.topk = int(topk)

        self.query_embed = nn.Parameter(torch.empty(self.num_queries, self.embed_dim))
        trunc_normal_(self.query_embed, std=0.02)

        self.blocks = nn.ModuleList([
            AttrTokenBlock(embed_dim=self.embed_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(int(num_layers))
        ])

        # f_A mapping
        self.attr_mapper = InversionNetwork(dim=self.embed_dim)

    def forward(self, patch_tokens: torch.Tensor, v_global: torch.Tensor):
        """
        patch_tokens: (B,M,D)
        v_global:     (B,D)
        returns:
            attr_topk:  (B,topk,D)
            ortho_loss: scalar tensor
        """
        B, M, D = patch_tokens.shape
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)

        for blk in self.blocks:
            q = blk(q, patch_tokens)  # (B,Q,D)

        # top-k by cosine similarity with v_global
        qn = F.normalize(q, dim=-1)
        vn = F.normalize(v_global, dim=-1).unsqueeze(1)  # (B,1,D)
        sim = (qn * vn).sum(dim=-1)  # (B,Q)

        topk = min(self.topk, sim.size(1))
        idx = sim.topk(topk, dim=1, largest=True, sorted=False).indices  # (B,topk)
        attr = q.gather(1, idx.unsqueeze(-1).expand(B, topk, D))          # (B,topk,D)

        # map to token embedding space
        attr_mapped = self.attr_mapper(attr.reshape(B * topk, D)).reshape(B, topk, D)

        ortho = orthogonal_token_loss(attr_mapped)
        return attr_mapped, ortho
