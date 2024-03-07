import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: nn.Module = nn.LeakyReLU()
    ):
        super().__init__()
        self.mlp = nn.Linear(input_size, output_size)
        self.act = activation
        self.output_size = output_size
        self.input_size = input_size
        self.output_size = output_size
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 2:
            return self.act(self.mlp(x.contiguous().view(-1, x.shape[-1]))).view(x.shape[0], x.shape[1], self.output_size)
        else:
            return self.act(self.mlp(x))


    def reset_parameters(self):
        nn.init.orthogonal_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)