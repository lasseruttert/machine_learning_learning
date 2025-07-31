# Evaluation Framework Design for MuZero Chess

## Overview

This document describes the design of the evaluation framework for the MuZero chess agent. The evaluation framework is used to assess the performance of the agent during and after training, comparing it against baselines or previous versions.

## Evaluation Requirements

The evaluation framework needs to:
1. Test the agent's performance against various opponents
2. Track performance metrics over time
3. Support different evaluation scenarios
4. Provide detailed game analysis
5. Generate performance reports

## Core Components

### Evaluator Class

```python
class Evaluator:
    def __init__(self, config):
        self.config = config
        self.results = []
        
    def evaluate_agent(self, agent, opponent_type="random", num_games=100):
        """Evaluate agent against specified opponent"""
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(num_games):
            # Alternate colors
            if game_idx % 2 == 0:
                result = self._play_game(agent, opponent_type, agent_white=True)
            else:
                result = self._play_game(agent, opponent_type, agent_white=False)
                
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            else:
                draws += 1
                
        # Calculate statistics
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0
        draw_rate = draws / total_games if total_games > 0 else 0
        
        evaluation_result = {
            "opponent": opponent_type,
            "games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate
        }
        
        self.results.append(evaluation_result)
        return evaluation_result
        
    def _play_game(self, agent, opponent_type, agent_white=True):
        """Play a single game between agent and opponent"""
        environment = ChessEnvironment()
        observation = environment.reset()
        done = False
        
        # Initialize opponent
        opponent = self._create_opponent(opponent_type)
        
        while not done:
            legal_actions = environment.get_legal_actions()
            
            if (agent_white and environment.board.turn == chess.WHITE) or \
               (not agent_white and environment.board.turn == chess.BLACK):
                # Agent's turn
                action = self._get_agent_action(agent, observation, legal_actions)
            else:
                # Opponent's turn
                action = self._get_opponent_action(opponent, observation, legal_actions)
                
            # Execute action
            observation, reward, done, _ = environment.step(action)
            
        # Determine game result
        return self._determine_result(environment, agent_white)
        
    def _create_opponent(self, opponent_type):
        """Create opponent based on type"""
        if opponent_type == "random":
            return RandomOpponent()
        elif opponent_type == "stockfish":
            return StockfishOpponent()
        elif opponent_type == "previous_version":
            return PreviousVersionOpponent()
        else:
            return RandomOpponent()
            
    def _get_agent_action(self, agent, observation, legal_actions):
        """Get action from agent using MCTS"""
        # Use MCTS with low temperature for evaluation
        action = agent.select_action(observation, legal_actions, temperature=0.1)
        return action
        
    def _get_opponent_action(self, opponent, observation, legal_actions):
        """Get action from opponent"""
        return opponent.select_action(observation, legal_actions)
        
    def _determine_result(self, environment, agent_white):
        """Determine game result from environment"""
        result = environment.get_result()
        
        if result == "1-0":  # White wins
            return "win" if agent_white else "loss"
        elif result == "0-1":  # Black wins
            return "loss" if agent_white else "win"
        else:  # Draw
            return "draw"
```

## Opponent Implementations

### Random Opponent

```python
class RandomOpponent:
    def select_action(self, observation, legal_actions):
        """Select random legal action"""
        return random.choice(legal_actions)
```

### Stockfish Opponent (if available)

```python
class StockfishOpponent:
    def __init__(self, skill_level=10):
        try:
            import chess.engine
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
            self.engine.configure({"Skill Level": skill_level})
        except:
            # Fallback to random if Stockfish not available
            self.engine = None
            
    def select_action(self, observation, legal_actions):
        """Select action using Stockfish engine"""
        if self.engine is None:
            return random.choice(legal_actions)
            
        # Convert observation to chess.Board
        board = self._observation_to_board(observation)
        
        # Get best move from Stockfish
        result = self.engine.play(board, chess.engine.Limit(time=1.0))
        return result.move
        
    def _observation_to_board(self, observation):
        """Convert observation to chess.Board"""
        # Implementation details...
        pass
```

### Previous Version Opponent

```python
class PreviousVersionOpponent:
    def __init__(self, model_path):
        self.agent = MuZeroAgent()
        self.agent.load_model(model_path)
        
    def select_action(self, observation, legal_actions):
        """Select action using previous version of agent"""
        return self.agent.select_action(observation, legal_actions, temperature=0.1)
```

## Evaluation Metrics

### Key Performance Indicators

1. **Win Rate**: Percentage of games won
2. **Draw Rate**: Percentage of games drawn
3. **Elo Rating**: Relative strength rating
4. **Average Game Length**: How long games typically last
5. **Action Consistency**: How consistently the agent selects optimal moves

### Detailed Analysis

```python
class GameAnalyzer:
    def __init__(self):
        pass
        
    def analyze_game(self, game_history):
        """Analyze a single game"""
        analysis = {
            "length": game_history.game_length,
            "captures": self._count_captures(game_history),
            "checks": self._count_checks(game_history),
            "material_balance": self._calculate_material_balance(game_history),
            "opening_moves": self._get_opening_moves(game_history, 10),
            "endgame_phase": self._detect_endgame_phase(game_history)
        }
        return analysis
        
    def _count_captures(self, game_history):
        """Count number of captures in game"""
        # Implementation details...
        pass
        
    def _count_checks(self, game_history):
        """Count number of checks in game"""
        # Implementation details...
        pass
```

## Reporting System

### Evaluation Reports

```python
class EvaluationReport:
    def __init__(self, evaluation_results):
        self.results = evaluation_results
        
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            "summary": self._generate_summary(),
            "detailed_results": self._generate_detailed_results(),
            "trends": self._analyze_trends(),
            "recommendations": self._generate_recommendations()
        }
        return report
        
    def _generate_summary(self):
        """Generate summary of evaluation results"""
        # Implementation details...
        pass
        
    def save_report(self, filename):
        """Save report to file"""
        with open(filename, 'w') as f:
            json.dump(self.generate_report(), f, indent=2)
```

## Implementation Plan

1. Implement the Evaluator class
2. Create opponent implementations
3. Add game analysis capabilities
4. Implement reporting system
5. Test with different opponents
6. Validate metrics accuracy

## Key Considerations

1. **Fair Evaluation**: Ensure balanced conditions for all opponents
2. **Statistical Significance**: Use adequate sample sizes for reliable metrics
3. **Computational Efficiency**: Optimize evaluation to minimize resource usage
4. **Reproducibility**: Maintain consistent evaluation conditions
5. **Extensibility**: Design framework to support new opponents and metrics
