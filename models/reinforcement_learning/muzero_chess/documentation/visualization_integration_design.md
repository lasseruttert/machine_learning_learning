# Visualization Integration Design for MuZero Chess

## Overview

This document describes the design for integrating the chess visualization with the MuZero chess agent. The goal is to provide real-time visualization of games played by the agent, including the moves generated during self-play and evaluation.

## Requirements

1. Visualize games played by the MuZero agent in real-time
2. Show the MCTS search process and decision making
3. Support both self-play and evaluation modes
4. Allow for interactive controls (pause, step, etc.)
5. Maintain performance while providing visualization

## Architecture

### Visualization Wrapper

The visualization wrapper will be a decorator pattern that wraps around the existing chess environment and MCTS components to provide visualization capabilities without modifying the core logic.

```python
class VisualizationWrapper:
    def __init__(self, environment, visualizer):
        self.environment = environment
        self.visualizer = visualizer
        self.is_visualizing = False
        
    def enable_visualization(self):
        """Enable real-time visualization"""
        self.is_visualizing = True
        
    def disable_visualization(self):
        """Disable visualization"""
        self.is_visualizing = False
        
    def step(self, action):
        """Wrap environment step with visualization"""
        result = self.environment.step(action)
        
        if self.is_visualizing:
            # Visualize the move
            self.visualizer.animate_move(self.environment.board, action)
            self.visualizer.update_display(self.environment.board)
            
        return result
        
    def reset(self):
        """Wrap environment reset with visualization"""
        result = self.environment.reset()
        
        if self.is_visualizing:
            # Visualize the initial position
            self.visualizer.update_display(self.environment.board)
            
        return result
```

### Self-Play Integration

The self-play process will be modified to optionally include visualization:

```python
class SelfPlayWithVisualization:
    def __init__(self, network, config, visualizer=None):
        self.network = network
        self.config = config
        self.environment = ChessEnvironment()
        self.visualizer = visualizer
        self.is_visualizing = visualizer is not None
        
    def play_game(self, temperature=1.0):
        """Play a single game with optional visualization"""
        # Initialize visualization if enabled
        if self.is_visualizing:
            self.visualizer.initialize_board()
            self.visualizer.update_display(self.environment.board)
            pygame.time.wait(1000)  # Pause to show initial position
            
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
            
            # Visualize the move if enabled
            if self.is_visualizing:
                # Highlight the selected move
                move = self.environment._action_to_move(action)
                self.visualizer.update_display(
                    self.environment.board, 
                    [move.from_square, move.to_square]
                )
                pygame.time.wait(500)  # Pause before move
                
                # Animate the move
                self.visualizer.animate_move(self.environment.board, move)
                
            # Execute action
            next_observation, reward, done, _ = self.environment.step(action)
            
            # Update visualization
            if self.is_visualizing:
                self.visualizer.update_display(self.environment.board)
                pygame.time.wait(1000)  # Pause after move
                
            observation = next_observation
            
        # Keep window open at the end if visualizing
        if self.is_visualizing:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            
                pygame.time.wait(100)  # Small delay to prevent high CPU usage
                
            self.visualizer.close()
            
        return GameHistory(observations, actions, rewards, policies, values)
```

### Evaluation Mode with Visualization

A separate evaluation mode will be created to visualize games between different agent versions:

```python
class EvaluationWithVisualization:
    def __init__(self, config, visualizer=None):
        self.config = config
        self.visualizer = visualizer
        self.is_visualizing = visualizer is not None
        
    def evaluate(self, network1, network2, num_games=10):
        """Evaluate two networks with visualization"""
        wins = 0
        draws = 0
        losses = 0
        
        for game_idx in range(num_games):
            # Alternate colors
            if game_idx % 2 == 0:
                white_network = network1
                black_network = network2
            else:
                white_network = network2
                black_network = network1
                
            # Play game with visualization
            result = self._play_game(white_network, black_network, game_idx)
            
            # Update statistics
            if (game_idx % 2 == 0 and result == "1-0") or (game_idx % 2 == 1 and result == "0-1"):
                wins += 1
            elif result == "1/2-1/2":
                draws += 1
            else:
                losses += 1
                
        return wins, draws, losses
        
    def _play_game(self, white_network, black_network, game_id):
        """Play a single game with visualization"""
        # Initialize environment and visualization
        environment = ChessEnvironment()
        
        if self.is_visualizing:
            self.visualizer.initialize_board()
            self.visualizer.update_display(environment.board)
            pygame.time.wait(1000)
            
        # Initialize MCTS for both players
        white_mcts = MCTS(white_network, self.config)
        black_mcts = MCTS(black_network, self.config)
        
        done = False
        while not done:
            # Select network based on current player
            if environment.board.turn == chess.WHITE:
                network = white_network
                mcts = white_mcts
            else:
                network = black_network
                mcts = black_mcts
                
            # Get legal actions
            legal_actions = environment.get_legal_actions()
            
            # Run MCTS
            observation = environment._get_state()
            action_probs = mcts.run(observation, legal_actions)
            
            # Select action
            action = self._select_action(action_probs, temperature=0.0)  # Deterministic for evaluation
            
            # Visualize move
            if self.is_visualizing:
                move = environment._action_to_move(action)
                self.visualizer.update_display(
                    environment.board, 
                    [move.from_square, move.to_square]
                )
                pygame.time.wait(500)
                self.visualizer.animate_move(environment.board, move)
                
            # Execute move
            _, _, done, _ = environment.step(action)
            
            # Update visualization
            if self.is_visualizing:
                self.visualizer.update_display(environment.board)
                pygame.time.wait(1000)
                
        # Show result and wait for user input
        if self.is_visualizing:
            result = environment.get_result()
            print(f"Game {game_id} result: {result}")
            
            # Wait for user to close or press ESC
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            
                pygame.time.wait(100)
                
            self.visualizer.close()
            
        return environment.get_result()
```

## Implementation Plan

1. Create the VisualizationWrapper class
2. Implement SelfPlayWithVisualization class
3. Create EvaluationWithVisualization class
4. Add configuration options for visualization
5. Test integration with existing components
6. Add interactive controls (pause, step, speed control)
7. Add MCTS visualization features (search tree, visit counts)

## Key Considerations

1. **Performance**: Visualization should not significantly impact training performance
2. **Modularity**: Visualization components should be optional and easily disabled
3. **User Experience**: Provide clear controls and feedback during visualization
4. **Compatibility**: Ensure visualization works with existing training and evaluation workflows
5. **Scalability**: Design should support different visualization levels (full, minimal, etc.)