import chess
import numpy as np
import torch
from chess_visualization import ChessVisualizer
from evaluation_visualization import EvaluationWithVisualization
import os


# Mock configuration class
class EvalConfig:
    def __init__(self):
        self.discount = 0.997
        self.hidden_channels = 256
        self.action_space_size = 4672
        self.input_channels = 19
        self.num_simulations = 10
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


# Mock network class for fallback
class MockNetwork:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initial_inference(self, observation):
        # Return dummy values
        batch_size = observation.shape[0]
        hidden_state = torch.zeros(batch_size, 256, 8, 8, device=self.device)
        policy = torch.zeros(batch_size, 4672, device=self.device)
        value = torch.zeros(batch_size, 1, device=self.device)
        return hidden_state, policy, value
        
    def recurrent_inference(self, hidden_state, action):
        # Return dummy values
        batch_size = hidden_state.shape[0]
        next_hidden_state = torch.zeros(batch_size, 256, 8, 8, device=self.device)
        reward = torch.zeros(batch_size, 1, device=self.device)
        policy = torch.zeros(batch_size, 4672, device=self.device)
        value = torch.zeros(batch_size, 1, device=self.device)
        return next_hidden_state, reward, policy, value
        
    def prediction(self, hidden_state):
        # Return dummy values
        batch_size = hidden_state.shape[0]
        policy = torch.zeros(batch_size, 4672, device=self.device)
        value = torch.zeros(batch_size, 1, device=self.device)
        return policy, value
        
    def dynamics(self, hidden_state, action):
        # Return dummy values
        batch_size = hidden_state.shape[0]
        next_hidden_state = torch.zeros(batch_size, 256, 8, 8, device=self.device)
        reward = torch.zeros(batch_size, 1, device=self.device)
        return next_hidden_state, reward


class StockfishEvaluator:
    def __init__(self, model_path=None, stockfish_path="stockfish"):
        """
        Initialize the Stockfish evaluator.
        
        Args:
            model_path (str): Path to trained model
            stockfish_path (str): Path to Stockfish executable
        """
        self.model_path = model_path
        self.stockfish_path = stockfish_path
        self.visualizer = ChessVisualizer()
        
    def evaluate(self, num_games=5):
        """
        Evaluate trained model against Stockfish.
        
        Args:
            num_games (int): Number of games to play
        """
        print(f"Starting evaluation against Stockfish for {num_games} games...")
        
        # Load trained model (simplified)
        network = self._load_model()
        
        # Initialize Stockfish engine (simplified)
        try:
            # Try to import chess.engine
            import chess.engine
            # Try to initialize Stockfish
            if os.path.exists(self.stockfish_path) or self._is_stockfish_in_path():
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                print("Stockfish engine initialized successfully")
            else:
                raise Exception("Stockfish not found")
        except Exception as e:
            print(f"Could not initialize Stockfish: {e}")
            print("Using random engine as fallback")
            engine = MockRandomEngine()
            
        # Create evaluation with visualization
        evaluator = EvaluationWithVisualization(
            config=EvalConfig(), 
            visualizer=self.visualizer
        )
        
        # Play games with visualization
        wins, draws, losses = evaluator.evaluate(network, engine, num_games)
        
        # Print results
        print(f"Evaluation Results ({num_games} games):")
        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
        if num_games > 0:
            print(f"Win Rate: {wins/num_games*100:.1f}%")
        
        print("Evaluation completed!")
        
    def _load_model(self):
        """Load trained model (simplified)"""
        print(f"Loading model from {self.model_path if self.model_path else 'default location'}")
        try:
            # Try to load the actual trained model
            from chess_trainer import TrainingConfig
            from muzero_network import MuZeroNetwork
            
            config = TrainingConfig()
            # Use CPU for evaluation to avoid device issues
            device = torch.device("cpu")
            network = MuZeroNetwork(config).to(device)
            
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
                network.load_state_dict(checkpoint['network_state_dict'])
                print("Model loaded successfully")
            else:
                print("Model file not found, using untrained network")
                
            return network
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using mock network as fallback")
            return MockNetwork()
            
    def _is_stockfish_in_path(self):
        """Check if stockfish is in the system PATH"""
        import shutil
        return shutil.which("stockfish") is not None


# Mock random engine for fallback
class MockRandomEngine:
    def __init__(self):
        self.name = "Mock Random Engine"
        
    def play(self, board, limit):
        """Mock play function"""
        # Select a random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = np.random.choice(legal_moves)
            # Create a mock result object
            class MockResult:
                def __init__(self, move):
                    self.move = move
            return MockResult(move)
        return None
        
    def quit(self):
        """Mock quit function"""
        print("Mock random engine closed")