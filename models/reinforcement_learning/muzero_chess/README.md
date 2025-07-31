# MuZero Chess Agent

This project implements a MuZero reinforcement learning agent for playing chess. The implementation includes training capabilities, evaluation against Stockfish, and visualization of both the training process and game play.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [File Descriptions](#file-descriptions)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Visualization](#visualization)

## Overview

The MuZero Chess Agent is a reinforcement learning system that learns to play chess through self-play. It uses the MuZero algorithm, which combines model-based planning with deep neural networks to make decisions. The agent can be trained to improve its play strength and evaluated against the Stockfish chess engine.

Key features:
- Real-time training progress visualization with matplotlib
- Full game visualization during evaluation with pygame
- Support for both training and evaluation in a single pipeline
- Device-agnostic implementation (works on both CPU and GPU)
- Fallback mechanisms for when Stockfish is not available

## Project Structure

```
muzero_chess/
├── chess_environment.py
├── muzero_network.py
├── chess_visualization.py
├── training_visualizer.py
├── chess_trainer.py
├── stockfish_evaluator.py
├── evaluation_visualization.py
├── train_and_evaluate.py
└── README.md
```

## File Descriptions

### chess_environment.py
Implements the chess environment using the python-chess library. This file provides:
- Board representation and state encoding (8x8x19 tensor)
- Action space definition (4672 possible moves)
- Move encoding/decoding between action indices and UCI notation
- Reward calculation based on game outcomes (+1 for win, -1 for loss, 0 for draw)
- Legal move generation and validation
- Game state tracking (check, checkmate, stalemate, etc.)

### muzero_network.py
Contains the neural network implementation for the MuZero algorithm:
- **ResidualBlock**: Basic building block for deep networks
- **RepresentationNetwork**: Encodes the current observation into a hidden state
- **DynamicsNetwork**: Predicts the next hidden state and reward given current state and action
- **PredictionNetwork**: Predicts policy (action probabilities) and value (expected future reward)
- **MuZeroNetwork**: Combines all components into a complete MuZero network

### chess_visualization.py
Provides visualization capabilities for chess games:
- Pygame-based chess board rendering
- Unicode chess piece symbols for visual representation
- Move animation and highlighting
- Interactive controls for viewing games
- Coordinate system for board positions

### training_visualizer.py
Implements real-time visualization of the training process:
- Matplotlib-based plots for training metrics
- Average game length per epoch tracking
- Average reward per epoch tracking
- Live updating of training progress
- Option to disable visualization for faster training

### chess_trainer.py
Implements the training loop for the MuZero chess agent:
- **TrainingConfig**: Configuration parameters for training
- **GameHistory**: Stores game trajectories for replay
- **ReplayBuffer**: Stores and samples game experiences
- **MCTS**: Monte Carlo Tree Search implementation for action selection
- **Node**: Tree node structure for MCTS
- **ChessTrainer**: Main training class with self-play and network training

### stockfish_evaluator.py
Provides evaluation capabilities against the Stockfish chess engine:
- Integration with Stockfish for benchmarking
- Game play visualization during evaluation
- Win/loss/draw statistics tracking
- Mock engines for testing when Stockfish is not available
- Automatic fallback to CPU for device compatibility

### evaluation_visualization.py
Implements visualization for evaluation games:
- Game evaluation with visualization capabilities
- Interactive chess board display for watching evaluation games
- Support for evaluating two different networks against each other
- Pygame-based visualization with move animation
- Device-agnostic implementation

### train_and_evaluate.py
Main script for training and evaluation:
- Command-line interface for controlling the agent
- Training and evaluation workflows
- Model saving and loading capabilities
- Options to enable/disable visualization

## Installation

1. Install the required dependencies:
```bash
pip install torch numpy matplotlib pygame python-chess
```

2. For Stockfish evaluation, download and install the Stockfish engine:
   - Download from: https://stockfishchess.org/download/
   - Make sure the `stockfish` executable is in your PATH

## Usage

The main entry point is `train_and_evaluate.py` which provides a command-line interface:

```bash
python train_and_evaluate.py [--train] [--evaluate] [--epochs EPOCHS] [--games GAMES] [--model MODEL] [--no-visualization]
```

Options:
- `--train`: Train the model
- `--evaluate`: Evaluate against Stockfish
- `--epochs`: Number of training epochs (default: 5)
- `--games`: Number of evaluation games (default: 3)
- `--model`: Model file path (default: 'trained_chess_model.pth')
- `--no-visualization`: Disable training visualization for faster training

## Training

To train the MuZero chess agent:
```bash
python train_and_evaluate.py --train --epochs 10
```

To train without visualization (faster):
```bash
python train_and_evaluate.py --train --epochs 10 --no-visualization
```

During training:
1. The agent plays games against itself (self-play)
2. Game trajectories are stored in a replay buffer
3. The neural network is trained on sampled game positions
4. Training progress is visualized in real-time with matplotlib
5. The trained model is saved to 'trained_chess_model.pth'

Training metrics:
- Average game length per epoch
- Average reward per epoch (ranges from -1.0 to +1.0)

## Evaluation

To evaluate the trained agent against Stockfish:
```bash
python train_and_evaluate.py --evaluate --games 5
```

To evaluate without visualization:
```bash
python train_and_evaluate.py --evaluate --games 5 --no-visualization
```

During evaluation:
1. The trained model plays games against Stockfish (or a random engine if Stockfish is not available)
2. Game play is visualized with pygame (unless disabled)
3. Results are displayed (wins, losses, draws)
4. Win rate is calculated

## Visualization

The project includes two types of visualization:

1. **Training Visualization**: Real-time plots showing training progress
   - Average game length per epoch
   - Average reward per epoch
   - Can be disabled with `--no-visualization` flag for faster training

2. **Game Visualization**: Interactive chess board display for watching games
   - Pygame window showing chess board with Unicode pieces
   - Move highlighting and animation
   - Can be disabled with `--no-visualization` flag during evaluation

To see game visualization during evaluation, simply run the evaluation command. The pygame window will show each move with animation and highlighting.

For training visualization, matplotlib windows will show the training metrics updating in real-time.