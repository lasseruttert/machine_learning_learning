# Training and Evaluation Script Design

## Overview

This document describes the design for a complete training and evaluation script that:
1. Trains the MuZero chess model with visualization of training progress
2. Evaluates the trained model against Stockfish with full game visualization

## Requirements

1. Train the MuZero chess model with progress visualization
2. Show average game length and reward per training epoch
3. Evaluate the trained model against Stockfish with full game visualization
4. Save and load trained models
5. Support configurable training parameters

## Architecture

### Training with Progress Visualization

The training script will include real-time visualization of training metrics using matplotlib:

```python
import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.avg_lengths = []
        self.avg_rewards = []
        
        # Set up the plot
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('MuZero Chess Training Progress')
        
    def update_progress(self, epoch, avg_length, avg_reward):
        """Update training progress visualization"""
        self.epochs.append(epoch)
        self.avg_lengths.append(avg_length)
        self.avg_rewards.append(avg_reward)
        
        # Update plots
        self.ax1.clear()
        self.ax1.plot(self.epochs, self.avg_lengths, 'b-')
        self.ax1.set_title('Average Game Length per Epoch')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Average Length')
        self.ax1.grid(True)
        
        self.ax2.clear()
        self.ax2.plot(self.epochs, self.avg_rewards, 'r-')
        self.ax2.set_title('Average Reward per Epoch')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Average Reward')
        self.ax2.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
    def save_plot(self, filename):
        """Save the training progress plot"""
        plt.savefig(filename)
        
    def close(self):
        """Close the visualization"""
        plt.ioff()
        plt.close()
```

### Training Script

The main training script will orchestrate the training process:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from chess_environment import ChessEnvironment
from muzero_network import MuZeroNetwork
from replay_buffer import ReplayBuffer
from training_visualizer import TrainingVisualizer

class ChessTrainer:
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
        self.visualizer = TrainingVisualizer()
        
    def train(self, num_epochs=100):
        """Main training loop with visualization"""
        for epoch in range(num_epochs):
            # Self-play phase
            game_histories = self._self_play_phase()
            
            # Store games in replay buffer
            for history in game_histories:
                self.replay_buffer.save_game(history)
                
            # Training phase
            if self.replay_buffer.is_ready(self.config.batch_size):
                self._train_network()
                
            # Calculate and visualize metrics
            avg_length = np.mean([h.game_length for h in game_histories])
            avg_reward = np.mean([np.sum(h.rewards) for h in game_histories])
            self.visualizer.update_progress(epoch, avg_length, avg_reward)
            
            # Log progress
            print(f"Epoch {epoch}: Avg Length={avg_length:.2f}, Avg Reward={avg_reward:.2f}")
            
        # Save final model
        self._save_model()
        self.visualizer.close()
        
    def _self_play_phase(self):
        """Generate games through self-play"""
        game_histories = []
        for _ in range(self.config.games_per_epoch):
            history = self._play_game()
            game_histories.append(history)
        return game_histories
        
    def _play_game(self):
        """Play a single game using MCTS"""
        environment = ChessEnvironment()
        observation = environment.reset()
        done = False
        
        # Initialize game history
        history = GameHistory()
        
        while not done:
            legal_actions = environment.get_legal_actions()
            
            # Run MCTS to get action probabilities
            action_probs = self._run_mcts(observation, legal_actions)
            
            # Select action
            action = self._select_action(action_probs, temperature=1.0)
            
            # Execute action
            next_observation, reward, done, _ = environment.step(action)
            
            # Store transition
            history.store_transition(
                observation, action, reward, action_probs, 0, environment.board.turn
            )
            
            observation = next_observation
            
        # Calculate values using TD(lambda)
        history.values = self._calculate_values(history.rewards)
        
        return history
        
    def _run_mcts(self, observation, legal_actions):
        """Run MCTS to get action probabilities"""
        # Simplified implementation
        action_probs = {}
        for action in legal_actions:
            action_probs[action] = 1.0 / len(legal_actions)
        return action_probs
        
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
        
    def _train_network(self):
        """Train network on batch from replay buffer"""
        # Sample batch
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
        
    def _calculate_loss(self, batch):
        """Calculate MuZero loss"""
        # Implementation details...
        pass
        
    def _save_model(self):
        """Save the trained model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'trained_chess_model.pth')
```

### Evaluation Against Stockfish

After training, the script will evaluate the model against Stockfish:

```python
import chess
import chess.engine
from chess_visualization import ChessVisualizer
from evaluation_visualization import EvaluationWithVisualization

class StockfishEvaluator:
    def __init__(self, model_path, stockfish_path="stockfish"):
        self.model_path = model_path
        self.stockfish_path = stockfish_path
        self.visualizer = ChessVisualizer()
        
    def evaluate(self, num_games=10):
        """Evaluate trained model against Stockfish"""
        # Load trained model
        network = self._load_model()
        
        # Initialize Stockfish engine
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as e:
            print(f"Could not initialize Stockfish: {e}")
            return
            
        # Create evaluation with visualization
        evaluator = EvaluationWithVisualization(
            config=self._get_config(), 
            visualizer=self.visualizer
        )
        
        # Play games with visualization
        wins, draws, losses = evaluator.evaluate(network, engine, num_games)
        
        # Print results
        print(f"Evaluation Results ({num_games} games):")
        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"Win Rate: {wins/num_games*100:.1f}%")
        
        # Close engine
        engine.quit()
        
    def _load_model(self):
        """Load trained model"""
        # Implementation details...
        pass
        
    def _get_config(self):
        """Get configuration"""
        # Implementation details...
        pass
```

### Complete Script

The complete script will combine training and evaluation:

```python
#!/usr/bin/env python3

import argparse
import torch
from training_script import ChessTrainer
from stockfish_evaluation import StockfishEvaluator

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate MuZero chess agent')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate against Stockfish')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--games', type=int, default=10, help='Number of evaluation games')
    parser.add_argument('--model', type=str, default='trained_chess_model.pth', help='Model file path')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training...")
        # Initialize trainer
        config = get_training_config()
        trainer = ChessTrainer(config)
        
        # Train model
        trainer.train(num_epochs=args.epochs)
        print("Training completed!")
        
    if args.evaluate:
        print("Starting evaluation against Stockfish...")
        # Initialize evaluator
        evaluator = StockfishEvaluator(args.model)
        
        # Evaluate model
        evaluator.evaluate(num_games=args.games)
        print("Evaluation completed!")

def get_training_config():
    """Get training configuration"""
    class Config:
        def __init__(self):
            # Network parameters
            self.input_channels = 19
            self.hidden_channels = 256
            self.action_space_size = 4672
            
            # Training parameters
            self.learning_rate = 0.05
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.discount = 0.997
            self.games_per_epoch = 10
            
            # Replay buffer
            self.replay_buffer_size = 10000
            self.batch_size = 256
            
    return Config()

if __name__ == "__main__":
    main()
```

## Implementation Plan

1. Create the TrainingVisualizer class for progress visualization
2. Implement the ChessTrainer class with training loop
3. Create the StockfishEvaluator class for evaluation
4. Develop the complete training and evaluation script
5. Add model saving and loading functionality
6. Test with sample data
7. Optimize performance and memory usage

## Key Considerations

1. **Performance**: Training visualization should not significantly impact training speed
2. **Memory Management**: Efficient handling of game histories and replay buffer
3. **Error Handling**: Robust handling of Stockfish initialization and game errors
4. **Configurability**: Support for different training parameters and evaluation settings
5. **User Experience**: Clear progress indicators and informative output
6. **Compatibility**: Work with different versions of Stockfish and chess engines