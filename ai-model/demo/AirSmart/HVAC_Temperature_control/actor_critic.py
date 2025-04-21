from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

# define Actor Model
class Actor(nn.Module):
    def __init__(self, action_dim, state_dim=2):
        super().__init__()
        # As we wish to process temp data of past 10 days, we need lstm layer to capture time features
        self.lstm = nn.LSTM(input_size=state_dim,
                    hidden_size=64,
                    num_layers=1,
                    batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.4),
            nn.Linear(32, action_dim)
        )

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        out, (h_n, _) = self.lstm(s)
        action_probs = F.softmax(self.fc(out[:,-1,:]), dim=-1)
        return action_probs

# define Critic Model
class Critic(nn.Module):
    def __init__(self, state_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(state_dim,64,batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        out, _ = self.lstm(s)
        value = self.fc(out[:, -1, :])

        return value.squeeze(-1)