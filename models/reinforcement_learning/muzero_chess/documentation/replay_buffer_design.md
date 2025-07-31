# Replay Buffer Design for MuZero Chess

## Overview

This document describes the design of the replay buffer for the MuZero chess agent. The replay buffer stores game histories generated during self-play and provides samples for training the neural network.

## Replay Buffer Requirements

The replay buffer for MuZero needs to:
1. Store complete game histories
2. Efficiently sample training batches
3. Manage memory usage with a fixed capacity
4. Support prioritized sampling (optional)
5. Handle the specific data structures used in MuZero

## Core Data Structures

### Game History Storage

```python
class GameHistory:
    def __init__(self):
        self.observations = []  # State representations
        self.actions = []       # Actions taken
        self.rewards = []       # Rewards received
        self.policies = []      # Policy targets
        self.values = []        # Value targets
        self.to_play = []       # Player to move
        self.action_probabilities = []  # MCTS action probabilities
        self.game_length = 0    # Length of the game
        
    def store_transition(self, observation, action, reward, policy, value, player):
        """Store a transition in the game history"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.to_play.append(player)
        self.game_length += 1
        
    def get_observation(self, index):
        """Get observation at index"""
        return self.observations[index]
        
    def get_action(self, index):
        """Get action at index"""
        return self.actions[index]
        
    def get_reward(self, index):
        """Get reward at index"""
        return self.rewards[index]
        
    def get_policy(self, index):
        """Get policy at index"""
        return self.policies[index]
        
    def get_value(self, index):
        """Get value at index"""
        return self.values[index]
```

## Replay Buffer Implementation

### Basic Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def save_game(self, game_history):
        """Save a game history to the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = game_history
        self.position = (self.position + 1) % self.capacity
        
    def sample_batch(self, batch_size):
        """Sample a batch of game histories"""
        # Sample games randomly
        games = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Sample positions from each game
        batch = Batch()
        for game in games:
            # Sample a position in the game
            game_pos = random.randint(0, game.game_length - 1)
            
            # Add data to batch
            batch.add_data(game, game_pos)
            
        return batch
        
    def __len__(self):
        return len(self.buffer)
        
    def is_ready(self, batch_size):
        """Check if buffer has enough data for training"""
        return len(self.buffer) >= batch_size
```

### Prioritized Replay Buffer (Advanced)

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=1.0):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def save_game(self, game_history, priority=1.0):
        """Save a game history with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(0)
            
        self.buffer[self.position] = game_history
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample_batch(self, batch_size, beta=1.0):
        """Sample a batch using priorities"""
        if len(self.buffer) == 0:
            return None
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample games
        indices = np.random.choice(len(self.buffer), 
                                 min(batch_size, len(self.buffer)), 
                                 p=probabilities)
        
        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Sample positions from each game
        batch = Batch()
        for i, idx in enumerate(indices):
            game = self.buffer[idx]
            game_pos = random.randint(0, game.game_length - 1)
            batch.add_data(game, game_pos, weight=weights[i])
            
        return batch, indices
        
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled games"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                
    def __len__(self):
        return len(self.buffer)
```

## Batch Data Structure

### Training Batch

```python
class Batch:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.weights = []  # For prioritized replay
        
    def add_data(self, game, position, weight=1.0):
        """Add data from a game position to the batch"""
        self.observations.append(game.get_observation(position))
        self.actions.append(game.get_action(position))
        self.rewards.append(game.get_reward(position))
        self.policies.append(game.get_policy(position))
        self.values.append(game.get_value(position))
        self.weights.append(weight)
        
    def to_tensor(self, device):
        """Convert batch data to PyTorch tensors"""
        return BatchTensor(
            observations=torch.FloatTensor(self.observations).to(device),
            actions=torch.LongTensor(self.actions).to(device),
            rewards=torch.FloatTensor(self.rewards).to(device),
            policies=torch.FloatTensor(self.policies).to(device),
            values=torch.FloatTensor(self.values).to(device),
            weights=torch.FloatTensor(self.weights).to(device)
        )
        
class BatchTensor:
    def __init__(self, observations, actions, rewards, policies, values, weights):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.policies = policies
        self.values = values
        self.weights = weights
```

## Memory Management

### Efficient Storage

```python
class EfficientGameHistory:
    def __init__(self, initial_capacity=100):
        self.capacity = initial_capacity
        self.size = 0
        
        # Pre-allocate arrays for better memory efficiency
        self.observations = [None] * self.capacity
        self.actions = [None] * self.capacity
        self.rewards = [None] * self.capacity
        self.policies = [None] * self.capacity
        self.values = [None] * self.capacity
        self.to_play = [None] * self.capacity
        
    def store_transition(self, observation, action, reward, policy, value, player):
        """Store transition with dynamic resizing"""
        if self.size >= self.capacity:
            self._resize()
            
        self.observations[self.size] = observation
        self.actions[self.size] = action
        self.rewards[self.size] = reward
        self.policies[self.size] = policy
        self.values[self.size] = value
        self.to_play[self.size] = player
        self.size += 1
        
    def _resize(self):
        """Double the capacity of storage arrays"""
        self.capacity *= 2
        
        self.observations.extend([None] * (self.capacity - len(self.observations)))
        self.actions.extend([None] * (self.capacity - len(self.actions)))
        self.rewards.extend([None] * (self.capacity - len(self.rewards)))
        self.policies.extend([None] * (self.capacity - len(self.policies)))
        self.values.extend([None] * (self.capacity - len(self.values)))
        self.to_play.extend([None] * (self.capacity - len(self.to_play)))
```

## Configuration

### Replay Buffer Settings

```python
class ReplayBufferConfig:
    def __init__(self):
        self.replay_buffer_size = 100000  # Maximum number of games
        self.batch_size = 1024            # Training batch size
        self.priority_alpha = 1.0         # Priority exponent (0 = uniform)
        self.priority_beta = 1.0          # Importance-sampling exponent
        self.use_prioritized_replay = False  # Enable prioritized replay
```

## Integration with Training Loop

### Buffer Usage in Training

```python
class TrainingLoop:
    def __init__(self, config):
        self.config = config
        if config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.replay_buffer_size, 
                config.priority_alpha
            )
        else:
            self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
            
    def _train_network(self):
        """Train network on batch from replay buffer"""
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return
            
        # Sample batch
        if self.config.use_prioritized_replay:
            batch, indices = self.replay_buffer.sample_batch(
                self.config.batch_size, 
                self.config.priority_beta
            )
        else:
            batch = self.replay_buffer.sample_batch(self.config.batch_size)
            
        # Convert to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_tensor = batch.to_tensor(device)
        
        # Calculate loss
        loss, loss_components = self._calculate_loss(batch_tensor)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.config.use_prioritized_replay:
            # Calculate new priorities based on TD errors
            priorities = self._calculate_priorities(batch_tensor, loss_components)
            self.replay_buffer.update_priorities(indices, priorities)
```

## Implementation Plan

1. Implement the basic GameHistory class
2. Create the ReplayBuffer class
3. Implement batch sampling functionality
4. Add prioritized replay buffer (optional)
5. Create efficient storage mechanisms
6. Integrate with training loop
7. Test with sample data

## Key Considerations

1. **Memory Efficiency**: Minimize memory usage for storing large game histories
2. **Sampling Efficiency**: Fast random sampling of game positions
3. **Scalability**: Handle large buffer sizes efficiently
4. **Thread Safety**: Safe concurrent access if needed
5. **Persistence**: Save/load buffer contents to disk
6. **Data Integrity**: Ensure consistent data structures
7. **Performance**: Optimize for the specific access patterns of MuZero