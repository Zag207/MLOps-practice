from typing import Any

import torch
from torch import nn

class MyIrisNet(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.network_stack = nn.Sequential(
            nn.Linear(4, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )
    
    def forward(self, x):
        logits = self.network_stack(x)

        return logits
    
    def inference_proba(self, x):
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


