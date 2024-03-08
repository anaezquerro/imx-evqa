import torch.nn as nn
from modules.ffn import FFN
import torch
from typing import Optional, Tuple

class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        activation: nn.Module = nn.LeakyReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size//(2 if bidirectional else 1), num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True, bias=True)
        if output_size is None:
            self.ffn = activation
        else:
            self.ffn = FFN(hidden_size, output_size, activation)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, (hn, _) = self.lstm(x)
        if self.bidirectional:
            hn = self.ffn(hn.permute(1, 2, 0).flatten(-2, -1))
        h = self.ffn(h)
        return h, hn


    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)