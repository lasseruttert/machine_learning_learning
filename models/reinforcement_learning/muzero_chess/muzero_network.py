import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class RepresentationNetwork(nn.Module):
    def __init__(self, input_channels=19, hidden_channels=256):
        super().__init__()
        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(16)
        ])
        
    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return x


class DynamicsNetwork(nn.Module):
    def __init__(self, hidden_channels=256, action_space_size=4672):
        super().__init__()
        # Action encoding
        self.action_encoder = nn.Embedding(action_space_size, hidden_channels)
        
        # Convolutional block for combining state and action
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(16)
        ])
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, hidden_state, action):
        # Encode action
        action_encoding = self.action_encoder(action)  # (batch, hidden_channels)
        action_encoding = action_encoding.view(-1, action_encoding.size(1), 1, 1)  # Reshape to spatial
        action_encoding = action_encoding.repeat(1, 1, 8, 8)  # Repeat to match spatial dimensions
        
        # Combine hidden state and action
        x = torch.cat([hidden_state, action_encoding], dim=1)
        
        # Process through convolutional blocks
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
            
        # Predict reward
        reward = self.reward_head(x)
        
        return x, reward


class PredictionNetwork(nn.Module):
    def __init__(self, hidden_channels=256, action_space_size=4672):
        super().__init__()
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, action_space_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, hidden_state):
        policy = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return policy, value


class MuZeroNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.representation = RepresentationNetwork(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels
        )
        self.dynamics = DynamicsNetwork(
            hidden_channels=config.hidden_channels,
            action_space_size=config.action_space_size
        )
        self.prediction = PredictionNetwork(
            hidden_channels=config.hidden_channels,
            action_space_size=config.action_space_size
        )
        
    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value
        
    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy, value