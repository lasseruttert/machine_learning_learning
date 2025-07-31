# MuZero Core Components Design

## Overview

This document describes the design of the core MuZero neural network components: representation, dynamics, and prediction networks. These components work together to learn a model of the environment and enable planning through MCTS.

## Network Architecture

### Input/Output Specifications

#### Representation Network
- Input: Chess board state tensor (8x8x19)
- Output: Hidden state (HxWxC) for the dynamics network

#### Dynamics Network
- Input: Hidden state + action (encoded)
- Output: Next hidden state + reward prediction

#### Prediction Network
- Input: Hidden state
- Output: Policy (action probabilities) + value (state evaluation)

## Neural Network Design

### Representation Network

```python
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
```

### Dynamics Network

```python
class DynamicsNetwork(nn.Module):
    def __init__(self, hidden_channels=256, action_space_size=4672):
        super().__init__()
        # Action encoding
        self.action_encoder = nn.Embedding(action_space_size, hidden_channels)
        
        # Convolutional block for combining state and action
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=3, padding=1),
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
        action_encoding = action_encoding.view(-1, 1, 8, 8)  # Reshape to spatial
        
        # Combine hidden state and action
        x = torch.cat([hidden_state, action_encoding], dim=1)
        
        # Process through convolutional blocks
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
            
        # Predict reward
        reward = self.reward_head(x)
        
        return x, reward
```

### Prediction Network

```python
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
```

### Residual Block

```python
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
```

## MuZero Model Integration

### Combined MuZero Network

```python
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
```

## Training Configuration

### Hyperparameters

```python
class MuZeroConfig:
    def __init__(self):
        # Network architecture
        self.input_channels = 19  # Chess state representation
        self.hidden_channels = 256
        self.action_space_size = 4672  # Chess action space
        
        # Training parameters
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.discount = 0.997
        
        # MCTS parameters
        self.num_simulations = 50
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Replay buffer
        self.replay_buffer_size = 100000
        self.batch_size = 1024
        
        # Optimization
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.learning_rate = 0.05
```

## Loss Function

The MuZero loss function combines three components:

1. **Policy Loss**: Cross-entropy between predicted and target policies
2. **Value Loss**: MSE between predicted and target values
3. **Reward Loss**: MSE between predicted and target rewards

```python
def muzero_loss(predictions, targets):
    policy_loss = cross_entropy_loss(predictions.policy, targets.policy)
    value_loss = mse_loss(predictions.value, targets.value)
    reward_loss = mse_loss(predictions.reward, targets.reward)
    
    total_loss = policy_loss + value_loss + reward_loss
    return total_loss, (policy_loss, value_loss, reward_loss)
```

## Implementation Plan

1. Implement the residual block component
2. Create the representation network
3. Create the dynamics network
4. Create the prediction network
5. Integrate all components into the MuZero network
6. Implement the loss function
7. Test with dummy data

## Key Considerations

1. **Computational Efficiency**: Use efficient operations and batch processing
2. **Memory Management**: Proper handling of hidden states and gradients
3. **Numerical Stability**: Appropriate initialization and normalization
4. **Scalability**: Design that can handle different board sizes and games
5. **Modularity**: Components that can be easily modified or replaced