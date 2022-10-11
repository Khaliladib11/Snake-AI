import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
   
class DuelingDQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()

        self.feature_extractor_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.value_q_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.advantage_q_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, X):
        features = self.feature_extractor_network(X)
        advantages = self.advantage_q_network(features)
        values = self.value_q_network(features)
        q_vals = values + (advantages - advantages.mean())
        return q_vals  