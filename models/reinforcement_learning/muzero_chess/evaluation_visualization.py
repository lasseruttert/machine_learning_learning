import pygame
import chess
import numpy as np
import torch
import math
import random
from chess_environment import ChessEnvironment
from chess_visualization import ChessVisualizer
from chess_trainer import Node, MCTS, TrainingConfig


class EvaluationWithVisualization:
    def __init__(self, config, visualizer=None):
        """
        Initialize evaluation with visualization.
        
        Args:
            config: Configuration parameters
            visualizer: ChessVisualizer instance (optional)
        """
        self.config = config
        self.visualizer = visualizer if visualizer is not None else ChessVisualizer()
        self.is_visualizing = visualizer is not None
        self.device = torch.device("cpu")  # Use CPU for evaluation to avoid device issues
        
    def evaluate(self, network1, network2, num_games=10):
        """
        Evaluate two networks with visualization.
        
        Args:
            network1: First network
            network2: Second network
            num_games (int): Number of games to play
            
        Returns:
            tuple: (wins, draws, losses) for network1
        """
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
        """
        Play a single game with visualization.
        
        Args:
            white_network: Network playing as white
            black_network: Network playing as black
            game_id (int): Game identifier
            
        Returns:
            str: Game result ("1-0", "0-1", "1/2-1/2")
        """
        # Initialize environment and visualization
        environment = ChessEnvironment()
        
        if self.is_visualizing:
            self.visualizer.initialize_board()
            self.visualizer.update_display(environment.board)
            pygame.time.wait(1000)
            
        done = False
        while not done:
            # Select network based on current player
            if environment.board.turn == chess.WHITE:
                network = white_network
            else:
                network = black_network
                
            # Get legal actions
            legal_actions = environment.get_legal_actions()
            
            # Run MCTS to get action probabilities
            action_probs = self._run_mcts(network, environment, legal_actions)
            
            # Select action (deterministic for evaluation)
            action = self._select_action(action_probs, temperature=0.0)
            
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
        result = environment.get_result()
        if self.is_visualizing:
            print(f"Game {game_id} result: {result}")
            self._wait_for_user_input()
            self.visualizer.close()
            
        return result
        
    def _run_mcts(self, network, environment, legal_actions):
        """
        Run MCTS to get action probabilities.
        
        Args:
            network: Network to use for MCTS
            environment: Chess environment
            legal_actions (list): List of legal action indices
            
        Returns:
            dict: Action probabilities
        """
        # Check if network is a mock network
        if hasattr(network, '__class__') and network.__class__.__name__ in ['MockNetwork', 'MockRandomEngine']:
            # Use random action selection for mock networks
            action_probs = {}
            for action in legal_actions:
                action_probs[action] = 1.0 / len(legal_actions)
            return action_probs
            
        # For actual networks, use MCTS
        try:
            # Get current observation
            observation = environment._get_state()
            
            # Create root node
            root = Node(0)
            
            # Initialize MCTS with the correct device
            config = TrainingConfig()
            mcts = MCTS(network, config, self.device)
            
            # Run MCTS
            action_probs, _ = mcts.run(root, observation, legal_actions)
            return action_probs
        except Exception as e:
            print(f"Error in MCTS: {e}")
            # Fallback to uniform random selection
            action_probs = {}
            for action in legal_actions:
                action_probs[action] = 1.0 / len(legal_actions)
            return action_probs
        
    def _select_action(self, action_probs, temperature):
        """
        Select action based on probabilities and temperature.
        
        Args:
            action_probs (dict): Action probabilities
            temperature (float): Temperature for selection
            
        Returns:
            int: Selected action
        """
        if not action_probs:
            return 0
            
        if temperature == 0:
            return max(action_probs, key=action_probs.get)
        else:
            # Apply temperature
            temp_probs = {a: p**(1/temperature) for a, p in action_probs.items()}
            total = sum(temp_probs.values())
            if total > 0:
                temp_probs = {a: p/total for a, p in temp_probs.items()}
            else:
                # If all probabilities are zero, assign uniform probabilities
                temp_probs = {a: 1.0/len(action_probs) for a in action_probs}
            
            # Sample action
            actions = list(temp_probs.keys())
            probs = list(temp_probs.values())
            return int(np.random.choice(actions, p=probs))
            
    def _wait_for_user_input(self):
        """Wait for user to close the visualization window."""
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