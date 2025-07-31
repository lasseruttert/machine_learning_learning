import torch
import torch.nn as nn
import numpy as np
import random
import math
from chess_environment import ChessEnvironment
from training_visualizer import TrainingVisualizer
from muzero_network import MuZeroNetwork
import chess


class TrainingConfig:
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
        self.games_per_epoch = 5
        
        # MCTS parameters
        self.num_simulations = 20
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Replay buffer
        self.replay_buffer_size = 1000
        self.batch_size = 64
        self.num_unroll_steps = 5
        self.td_steps = 10


class GameHistory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.to_play = []
        self.game_length = 0
        
    def store_transition(self, observation, action, reward, policy, value, player):
        """Store a transition in the game history"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.to_play.append(player)
        self.game_length += 1


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
        return games
        
    def __len__(self):
        return len(self.buffer)
        
    def is_ready(self, batch_size):
        """Check if buffer has enough data for training"""
        return len(self.buffer) >= batch_size


class Node:
    def __init__(self, prior):
        self.prior = prior  # Prior probability from neural network
        self.value_sum = 0  # Sum of values from simulations
        self.visit_count = 0  # Number of visits
        self.children = {}  # Action -> Node mapping
        self.hidden_state = None  # Hidden state for this node
        self.reward = 0  # Reward from parent to this node
        
    def expanded(self):
        """Check if node has been expanded"""
        return len(self.children) > 0
        
    def value(self):
        """Calculate mean value of node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, network, config, device):
        self.network = network
        self.config = config
        self.device = device
        
    def run(self, root, observation, legal_actions):
        """Run MCTS from root node"""
        # Add Dirichlet noise to root for exploration
        self._add_exploration_noise(root)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            actions = []
            
            # Selection phase
            while node.expanded():
                action, node = self._select_child(node)
                search_path.append(node)
                actions.append(action)
                
            # Expansion and evaluation
            parent = search_path[-2] if len(search_path) > 1 else root
            action = actions[-1] if actions else None
            
            # Use dynamics network to get next state and reward
            if action is not None and parent.hidden_state is not None:
                # Convert action to tensor
                action_tensor = torch.tensor([action], device=self.device)
                next_hidden_state, reward, policy, value = self.network.recurrent_inference(
                    parent.hidden_state, action_tensor
                )
            else:
                # Use representation network for root node
                # Transpose observation from (8, 8, 19) to (19, 8, 8)
                observation_transposed = np.transpose(observation, (2, 0, 1))
                observation_tensor = torch.tensor(observation_transposed, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_hidden_state, policy, value = self.network.initial_inference(observation_tensor)
                reward = torch.tensor([0.0], device=self.device)
            
            # Create new node
            node.hidden_state = next_hidden_state
            node.reward = reward.item() if isinstance(reward, torch.Tensor) else reward
            
            # Convert policy to numpy if it's a tensor
            if isinstance(policy, torch.Tensor):
                policy = policy.squeeze(0).cpu().detach().numpy()
            
            # Expand node with legal actions
            self._expand_node(node, policy, legal_actions)
            
            # Backup values
            self._backup(search_path, value.item() if isinstance(value, torch.Tensor) else value)
            
        return self._get_distributions(root)
        
    def _select_child(self, node):
        """Select child using UCB1"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def _ucb_score(self, parent, child):
        """Calculate UCB1 score for child node"""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / 
                        self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        value_score = child.value()
        
        return prior_score + value_score
        
    def _expand_node(self, node, policy, legal_actions):
        """Expand node with children for legal actions"""
        policy = self._mask_policy(policy, legal_actions)
        
        # Apply softmax to policy
        if isinstance(policy, torch.Tensor):
            policy = torch.softmax(policy, dim=0)
            policy = policy.cpu().detach().numpy()
        else:
            # Handle numpy array
            policy = np.exp(policy - np.max(policy))  # Numerical stability
            policy = policy / np.sum(policy)
        
        for action in legal_actions:
            if action < len(policy):
                node.children[action] = Node(prior=policy[action])
            
    def _mask_policy(self, policy, legal_actions):
        """Mask policy to only allow legal actions"""
        if isinstance(policy, torch.Tensor):
            masked_policy = torch.full_like(policy, -1e32)
            masked_policy[legal_actions] = policy[legal_actions]
        else:
            # Handle numpy array
            masked_policy = np.full_like(policy, -1e32)
            masked_policy[legal_actions] = policy[legal_actions]
        return masked_policy
        
    def _backup(self, search_path, value):
        """Backup values up the search path"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount * value
            
    def _add_exploration_noise(self, node):
        """Add Dirichlet noise to root node for exploration"""
        actions = list(node.children.keys())
        if actions:
            noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
            
            frac = self.config.root_exploration_fraction
            for a, n in zip(actions, noise):
                node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
                
    def _get_distributions(self, root):
        """Get action probabilities from visit counts"""
        visit_counts = [(action, child.visit_count) 
                       for action, child in root.children.items()]
        if not visit_counts:
            return {}, []
        actions, counts = zip(*visit_counts)
        total_visits = sum(counts)
        
        if total_visits == 0:
            probabilities = [1.0 / len(counts) for _ in counts]
        else:
            probabilities = [count / total_visits for count in counts]
        return dict(zip(actions, probabilities)), list(actions)


class ChessTrainer:
    def __init__(self, config=None, enable_visualization=True):
        """
        Initialize the chess trainer.
        
        Args:
            config: Training configuration
            enable_visualization: Whether to enable training visualization
        """
        self.config = config if config is not None else TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_visualization = enable_visualization
        
        # Initialize the MuZero network
        self.network = MuZeroNetwork(self.config).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.visualizer = TrainingVisualizer(enable_visualization)
        
    def train(self, num_epochs=10):
        """
        Main training loop with visualization.
        
        Args:
            num_epochs (int): Number of training epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        
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
        print("Training completed!")
        
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
        
        # Initialize MCTS
        mcts = MCTS(self.network, self.config, self.device)
        
        while not done and history.game_length < 100:  # Limit game length
            legal_actions = environment.get_legal_actions()
            
            if not legal_actions:
                # No legal moves - game over
                reward = -1.0 if environment.board.turn == chess.WHITE else 1.0
                history.rewards[-1] = reward  # Update last reward
                break
                
            # Create root node
            root = Node(0)
            
            # Run MCTS to get action probabilities
            action_probs, actions = mcts.run(root, observation, legal_actions)
            
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
        
    def _select_action(self, action_probs, temperature):
        """Select action based on probabilities and temperature"""
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
        games = self.replay_buffer.sample_batch(self.config.batch_size)
        
        # Calculate loss
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        reward_loss_total = 0
        
        # Process each game
        for game in games:
            if len(game.observations) == 0:
                continue
                
            # Sample a position in the game
            pos = np.random.randint(0, len(game.observations))
            
            # Initial inference
            observation = game.observations[pos]
            # Transpose observation from (8, 8, 19) to (19, 8, 8)
            observation_transposed = np.transpose(observation, (2, 0, 1))
            observation_tensor = torch.tensor(observation_transposed, dtype=torch.float32, device=self.device).unsqueeze(0)
            hidden_state, policy, value = self.network.initial_inference(observation_tensor)
            
            # Target values
            target_policy = np.zeros(self.config.action_space_size)
            if pos < len(game.policies):
                for action, prob in game.policies[pos].items():
                    if action < self.config.action_space_size:
                        target_policy[action] = prob
            target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32, device=self.device)
            
            target_value = game.values[pos] if pos < len(game.values) else 0
            target_value_tensor = torch.tensor([target_value], dtype=torch.float32, device=self.device)
            
            # Calculate losses for initial step
            policy_loss = nn.functional.cross_entropy(policy, target_policy_tensor.unsqueeze(0))
            value_loss = nn.functional.mse_loss(value, target_value_tensor)
            
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            
            # Recurrent inferences
            for step in range(self.config.num_unroll_steps):
                if pos + step < len(game.actions):
                    action = game.actions[pos + step]
                    action_tensor = torch.tensor([action], device=self.device)
                    
                    next_hidden_state, reward, policy, value = self.network.recurrent_inference(hidden_state, action_tensor)
                    
                    # Calculate losses
                    reward_target = game.rewards[pos + step] if pos + step < len(game.rewards) else 0
                    reward_target_tensor = torch.tensor([reward_target], dtype=torch.float32, device=self.device)
                    reward_loss = nn.functional.mse_loss(reward, reward_target_tensor)
                    
                    # For policy, we need a target
                    target_policy = np.zeros(self.config.action_space_size)
                    if pos + step + 1 < len(game.policies):
                        for action, prob in game.policies[pos + step + 1].items():
                            if action < self.config.action_space_size:
                                target_policy[action] = prob
                    target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32, device=self.device)
                    policy_loss = nn.functional.cross_entropy(policy, target_policy_tensor.unsqueeze(0))
                    
                    value_target = game.values[pos + step + 1] if pos + step + 1 < len(game.values) else 0
                    value_target_tensor = torch.tensor([value_target], dtype=torch.float32, device=self.device)
                    value_loss = nn.functional.mse_loss(value, value_target_tensor)
                    
                    reward_loss_total += reward_loss.item()
                    policy_loss_total += policy_loss.item()
                    value_loss_total += value_loss.item()
                    
        # Average losses
        if len(games) > 0:
            policy_loss_avg = policy_loss_total / (len(games) * (self.config.num_unroll_steps + 1))
            value_loss_avg = value_loss_total / (len(games) * (self.config.num_unroll_steps + 1))
            reward_loss_avg = reward_loss_total / (len(games) * self.config.num_unroll_steps) if self.config.num_unroll_steps > 0 else 0
            
            total_loss = policy_loss_avg + value_loss_avg + reward_loss_avg
            
            # Update network
            self.optimizer.zero_grad()
            # Create a loss tensor
            loss_tensor = torch.tensor(total_loss, requires_grad=True, dtype=torch.float32, device=self.device)
            loss_tensor.backward()
            self.optimizer.step()
            
            print(f"Training step completed with loss: {total_loss:.4f}")
        
    def _save_model(self):
        """Save the trained model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'trained_chess_model.pth')
        print("Model saved successfully!")