import torch
import torch.nn as nn

class InversionNetwork(nn.Module):
    """PromptSG/IADT subject mapping network f_S (3-layer MLP + BN).

    Input:  (B, 512) global visual embedding (projected)
    Output: (B, 512) pseudo token S* in CLIP token-embedding space
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(v))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        return x
