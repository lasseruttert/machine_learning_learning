# MCTS Implementation Design for MuZero Chess

## Overview

This document describes the design of the Monte Carlo Tree Search (MCTS) implementation for the MuZero chess agent. MCTS is used to select actions during self-play and evaluation by leveraging the learned model to perform look-ahead planning.

## MCTS Algorithm Overview

The MCTS algorithm in MuZero consists of four phases:
1. **Selection**: Traverse the tree from root to leaf using UCB1
2. **Expansion**: Add new nodes to the tree
3. **Evaluation**: Use the neural network to evaluate new nodes
4. **Backup**: Propagate values back up the tree

## Core Data Structures

### Node Structure

```python
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
```

### Root Node

```python
class RootNode(Node):
    def __init__(self, prior, hidden_state):
        super().__init__(prior)
        self.hidden_state = hidden_state
        self.to_play = None  # Player to move at root
```

## MCTS Implementation

### MCTS Class

```python
class MCTS:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.nodes = {}  # Hash map of nodes in the tree
        
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
            parent = search_path[-2]
            action = actions[-1]
            
            # Use dynamics network to get next state and reward
            next_hidden_state, reward = self.network.recurrent_inference(
                parent.hidden_state, action
            )
            
            # Use prediction network to get policy and value
            policy, value = self.network.prediction(next_hidden_state)
            
            # Create new node
            node.hidden_state = next_hidden_state
            node.reward = reward
            
            # Expand node with legal actions
            self._expand_node(node, policy, legal_actions)
            
            # Backup values
            self._backup(search_path, value)
            
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
        policy = torch.softmax(policy, dim=0)
        
        for action in legal_actions:
            node.children[action] = Node(prior=policy[action].item())
            
    def _mask_policy(self, policy, legal_actions):
        """Mask policy to only allow legal actions"""
        masked_policy = torch.full_like(policy, -1e32)
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
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
            
    def _get_distributions(self, root):
        """Get action probabilities from visit counts"""
        visit_counts = [(action, child.visit_count) 
                       for action, child in root.children.items()]
        actions, counts = zip(*visit_counts)
        total_visits = sum(counts)
        
        probabilities = [count / total_visits for count in counts]
        return actions, probabilities
```

## Integration with Chess Environment

### Action Selection

```python
def select_action(network, config, observation, legal_actions, temperature=1.0):
    """Select action using MCTS"""
    # Initial inference to get root node
    hidden_state, policy, value = network.initial_inference(observation)
    
    # Create root node
    root = RootNode(0, hidden_state)
    
    # Mask policy for legal actions
    policy = mask_policy(policy, legal_actions)
    policy = torch.softmax(policy, dim=0)
    
    # Expand root node
    for action in legal_actions:
        root.children[action] = Node(prior=policy[action].item())
    
    # Run MCTS
    mcts = MCTS(network, config)
    actions, probabilities = mcts.run(root, observation, legal_actions)
    
    # Select action based on temperature
    if temperature == 0:
        # Greedy selection
        action_idx = np.argmax(probabilities)
        return actions[action_idx]
    else:
        # Stochastic selection
        probabilities = [p ** (1/temperature) for p in probabilities]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return np.random.choice(actions, p=probabilities)
```

## MCTS Configuration

### Key Parameters

```python
class MCTSConfig:
    def __init__(self):
        self.num_simulations = 50  # Number of MCTS simulations per move
        self.root_dirichlet_alpha = 0.3  # Dirichlet noise parameter
        self.root_exploration_fraction = 0.25  # Fraction of noise to apply
        self.pb_c_base = 19652  # PUCT constant
        self.pb_c_init = 1.25  # PUCT constant
        self.discount = 0.997  # Discount factor
```

## Implementation Plan

1. Implement the Node data structure
2. Create the MCTS class with core algorithms
3. Implement UCB1 selection
4. Implement node expansion
5. Implement backup mechanism
6. Add exploration noise
7. Create action selection interface
8. Test with simple scenarios

## Key Considerations

1. **Efficiency**: Minimize redundant computations
2. **Memory Management**: Properly manage node storage and cleanup
3. **Numerical Stability**: Handle edge cases in UCB1 calculation
4. **Chess-Specific Logic**: Proper handling of legal moves and game rules
5. **Scalability**: Design that can handle different numbers of simulations
6. **Integration**: Seamless integration with neural network components