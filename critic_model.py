import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, height=5, width=5):
        super().__init__()

        self.height = height
        self.width = width

        #CNN for board encoding
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        #Fully connected layers
        self.fc1 = nn.Linear(64 * height * width + 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, board, action):
        """
        board: (B, 3, H, W)
        action: (B, 2)  -- (row, col) normalized or raw
        """
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)   # (B, 64*H*W)
        x = torch.cat([x, action], dim=1)  # (B, 64*H*W + 2)

        x = F.relu(self.fc1(x))     # (B, 128)
        x = self.fc2(x)             # (B, 1)

        return x.squeeze(1)
