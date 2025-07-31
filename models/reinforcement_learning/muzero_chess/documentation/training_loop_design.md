# Training Loop Design for MuZero Chess

## Overview

This document describes the design of the training loop for the MuZero chess agent. The training loop orchestrates the self-play, data collection, and network training phases that are essential for the agent to learn to play chess.

## Training Process Overview

The MuZero training process follows these steps:
1. **Self-Play**: Generate games using MCTS with the current network
2. **Data Storage**: Store game trajectories in a replay buffer
3. **Network Training**: Train the network on sampled trajectories
4. **Evaluation**: Periodically evaluate the new network against the previous version

## Core Components

### Self-Play Manager

```python
class SelfPlay:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.environment = ChessEnvironment()
        
    def play_game(self, temperature=1.0):
        """Play a single game using MCTS"""
        observations = []
        actions = []
        rewards = []
        policies = []
        values = []
        
        # Reset environment
        observation = self.environment.reset()
        done = False
        
        # Initialize MCTS
        mcts = MCTS(self.network, self.config)
        
        while not done:
            # Get legal actions
            legal_actions = self.environment.get_legal_actions()
            
            # Run MCTS to get action probabilities
            action_probs = mcts.run(observation, legal_actions)
            
            # Select action based on temperature
            action = self._select_action(action_probs, temperature)
            
            # Store data
            observations.append(observation)
            actions.append(action)
            policies.append(action_probs)
            
            # Execute action
            next_observation, reward, done, _ = self.environment.step(action)
            rewards.append(reward)
            
            observation = next_observation
            
        # Calculate values using TD(lambda)
        values = self._calculate_values(rewards)
        
        return GameHistory(observations, actions, rewards, policies, values)
        
    def _select_action(self, action_probs, temperature):
        """Select action based on probabilities and temperature"""
        if temperature == 0:
            return max(action_probs, key=action_probs.get)
        else:
            # Apply temperature
            temp_probs = {a: p**(1/temperature) for a, p in action_probs.items()}
            total = sum(temp_probs.values())
            temp_probs = {a: p/total for a, p in temp_probs.items()}
            
            # Sample action
            actions = list(temp_probs.keys())
            probs = list(temp_probs.values())
            return np.random.choice(actions, p=probs)
            
    def _calculate_values(self, rewards):
        """Calculate values using TD(lambda)"""
        values = [0] * len(rewards)
        value = 0
        
        # Backward pass
        for i in reversed(range(len(rewards))):
            value = rewards[i] + self.config.discount * value
            values[i] = value
            
        return values
```

### Training Loop

```python
class TrainingLoop:
    def __init__(self, config):
        self.config = config
        self.network = MuZeroNetwork(config)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.self_play = SelfPlay(self.network, config)
        
    def train(self, num_games=10000):
        """Main training loop"""
        for game_idx in range(num_games):
            # Self-play phase
            temperature = self._get_temperature(game_idx)
            game_history = self.self_play.play_game(temperature)
            
            # Store in replay buffer
            self.replay_buffer.save_game(game_history)
            
            # Training phase
            if len(self.replay_buffer) >= self.config.batch_size:
                self._train_network()
                
            # Evaluation phase
            if game_idx % self.config.evaluation_interval == 0:
                self._evaluate_network()
                
            # Logging
            if game_idx % self.config.log_interval == 0:
                self._log_training_stats(game_idx, game_history)
                
    def _get_temperature(self, game_idx):
        """Get temperature based on training progress"""
        if game_idx < self.config.temperature_threshold:
            return 1.0
        else:
            return 0.5
            
    def _train_network(self):
        """Train network on batch of game data"""
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample_batch(self.config.batch_size)
        
        # Calculate loss
        loss, loss_components = self._calculate_loss(batch)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _calculate_loss(self, batch):
        """Calculate MuZero loss"""
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        reward_loss_total = 0
        
        # Unroll trajectories
        for i in range(len(batch.observations)):
            # Initial inference
            hidden_state, policy, value = self.network.initial_inference(
                batch.observations[i]
            )
            
            # Calculate losses for initial step
            policy_loss = cross_entropy_loss(policy, batch.policies[i])
            value_loss = mse_loss(value, batch.values[i])
            
            policy_loss_total += policy_loss
            value_loss_total += value_loss
            
            # Recurrent inferences
            for step in range(self.config.num_unroll_steps):
                if i + step < len(batch.actions):
                    hidden_state, reward = self.network.recurrent_inference(
                        hidden_state, batch.actions[i + step]
                    )
                    
                    policy, value = self.network.prediction(hidden_state)
                    
                    # Calculate losses
                    reward_loss = mse_loss(reward, batch.rewards[i + step])
                    policy_loss = cross_entropy_loss(policy, batch.policies[i + step])
                    value_loss = mse_loss(value, batch.values[i + step + 1] 
                                        if i + step + 1 < len(batch.values) 
                                        else 0)
                    
                    reward_loss_total += reward_loss
                    policy_loss_total += policy_loss
                    value_loss_total += value_loss
                    
        # Average losses
        policy_loss_avg = policy_loss_total / (len(batch) * (self.config.num_unroll_steps + 1))
        value_loss_avg = value_loss_total / (len(batch) * (self.config.num_unroll_steps + 1))
        reward_loss_avg = reward_loss_total / (len(batch) * self.config.num_unroll_steps)
        
        total_loss = policy_loss_avg + value_loss_avg + reward_loss_avg
        
        return total_loss, (policy_loss_avg, value_loss_avg, reward_loss_avg)
        
    def _evaluate_network(self):
        """Evaluate current network against previous version"""
        # Implementation details...
        pass
        
    def _log_training_stats(self, game_idx, game_history):
        """Log training statistics"""
        # Implementation details...
        pass
```

## Game History Storage

### Game History Class

```python
class GameHistory:
    def __init__(self, observations, actions, rewards, policies, values):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.policies = policies
        self.values = values
        self.game_length = len(observations)
        
    def make_image(self, state_index):
        """Create image representation for a state"""
        return self.observations[state_index]
        
    def make_target(self, state_index, td_steps):
        """Create training targets for a state"""
        # Calculate value target using TD steps
        bootstrap_index = state_index + td_steps
        if bootstrap_index < len(self.values):
            value = self.rewards[state_index] + \
                   self.config.discount ** td_steps * self.values[bootstrap_index]
        else:
            value = self.values[state_index]
            
        # Get policy and reward targets
        policy = self.policies[state_index]
        reward = self.rewards[state_index] if state_index > 0 else 0
        
        return policy, value, reward
```

## Training Configuration

### Key Training Parameters

```python
class TrainingConfig:
    def __init__(self):
        # Self-play parameters
        self.temperature_threshold = 10000  # Games before reducing temperature
        self.num_self_play_games = 100000  # Total games to play
        
        # Training parameters
        self.batch_size = 1024
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.discount = 0.997
        
        # Evaluation parameters
        self.evaluation_interval = 1000  # Games between evaluations
        self.evaluation_games = 200  # Games in evaluation
        
        # Logging parameters
        self.log_interval = 100  # Games between logging
        
        # Optimization parameters
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.weight_decay = 1e-4
```

## Implementation Plan

1. Implement the GameHistory class
2. Create the SelfPlay class
3. Implement the TrainingLoop class
4. Add training loss calculation
5. Implement evaluation mechanism
6. Add logging and monitoring
7. Test with simple scenarios

## Key Considerations

1. **Memory Management**: Efficient storage and sampling of game histories
2. **Parallelization**: Potential for parallel self-play games
3. **Checkpointing**: Save and restore training progress
4. **Monitoring**: Track training progress and performance
5. **Scalability**: Design that can handle large numbers of games
6. **Reproducibility**: Consistent random seeds and configurations
7. **Resource Management**: GPU memory and computational efficiency