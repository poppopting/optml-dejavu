import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        drop= 0.1
    ):
        super().__init__()

        self.norm = nn.LayerNorm(input_size)

        self.fc1 = nn.Linear(input_size, 4*input_size)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(4*input_size, output_size)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        if mask is not None:
            x = x * mask
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# class FeedforwardNN(nn.Module):
#     def __init__(self, input_size, output_size=1, dropout_rate=0.1):
#         super(FeedforwardNN, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.LayerNorm(128), 
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         self.fc2 = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.LayerNorm(64),   
#             nn.ReLU(),
#             nn.Linear(64, output_size)  
#         )

#     def forward(self, x, mask=None):
#         x = self.fc1(x)
#         if mask is not None:
#             x = x * mask
#         x = self.fc2(x)
#         return x