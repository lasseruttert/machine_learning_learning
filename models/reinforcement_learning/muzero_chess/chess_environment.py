import chess
import numpy as np
import torch


class ChessEnvironment:
    def __init__(self):
        """
        Initialize the chess environment using python-chess library.
        """
        self.board = chess.Board()  # Initialize a new chess board
        self.action_space_size = 4672  # 8×8×73 action space
        self.action_space = self._get_action_space()
        
        # Define directions for queen moves (8 directions)
        self.queen_directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Southwest, South, Southeast
            (0, -1),           (0, 1),   # West, East
            (1, -1),  (1, 0),  (1, 1)    # Northwest, North, Northeast
        ]
        
        # Define knight moves (8 moves)
        self.knight_moves = [
            (-2, -1), (-2, 1),  # 2 squares vertical, 1 square horizontal
            (-1, -2), (-1, 2),  # 1 square vertical, 2 squares horizontal
            (1, -2),  (1, 2),
            (2, -1),  (2, 1)
        ]
        
        # Define underpromotion moves (3 directions, 3 pieces)
        self.underpromotion_moves = [
            (0, -1), (0, 0), (0, 1)  # Forward, Forward-Left, Forward-Right
        ]
        self.underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            np.array: Initial state tensor representation
        """
        self.board = chess.Board()
        return self._get_state()
        
    def step(self, action):
        """
        Execute an action and return (next_state, reward, done, info).
        
        Args:
            action (int): Action index in the action space
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Convert action index to UCI move
        move = self._action_to_move(action)
        
        # Apply move if legal
        if move in self.board.legal_moves:
            self.board.push(move)
            reward = self._get_reward()
            done = self._is_game_over()
            return self._get_state(), reward, done, {}
        else:
            # Illegal move penalty
            return self._get_state(), -1.0, True, {"illegal_move": True}
            
    def _get_state(self):
        """
        Convert current board state to tensor representation.
        
        Returns:
            np.array: 8x8x19 tensor representation of the board state
        """
        # Initialize the state tensor (8x8x19)
        state = np.zeros((8, 8, 19), dtype=np.float32)
        
        # 1. Piece-centric representation (12 channels)
        for i in range(64):
            row, col = divmod(i, 8)
            piece = self.board.piece_at(i)
            if piece:
                # Map piece types to channels
                # White pieces: 0-5, Black pieces: 6-11
                piece_type = piece.piece_type - 1  # 0-5 for pawn-king
                piece_color = 0 if piece.color == chess.WHITE else 6
                channel = piece_type + piece_color
                state[row, col, channel] = 1
        
        # 2. Player to move (1 channel)
        state[:, :, 12] = 1 if self.board.turn == chess.WHITE else 0
        
        # 3. Castling rights (4 channels)
        if self.board.has_kingside_castling_rights(chess.WHITE):
            state[:, :, 13] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            state[:, :, 14] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            state[:, :, 15] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            state[:, :, 16] = 1
            
        # 4. En passant target (1 channel)
        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            state[row, col, 17] = 1
            
        # 5. Halfmove clock (1 channel)
        state[:, :, 18] = self.board.halfmove_clock / 100.0  # Normalize
        
        return state
        
    def _get_action_space(self):
        """
        Define the action space for chess.
        
        Returns:
            list: List of all possible actions
        """
        # For now, return the size of action space
        # Actual implementation would map all 4672 actions
        return list(range(self.action_space_size))
        
    def _action_to_move(self, action):
        """
        Convert action index to chess.Move object.
        
        Args:
            action (int): Action index (0-4671)
            
        Returns:
            chess.Move: Corresponding chess move
        """
        # Convert action index to from_square, to_square, and promotion
        # Action space: 8×8×73 = 4672
        # 8×8 for the from square
        # 73 for move types:
        #   - 56 queen moves (up to 7 squares in 8 directions)
        #   - 8 knight moves
        #   - 9 underpromotions (3 pieces × 3 directions)
        
        # Extract components from action index
        from_square = action // 73
        move_type = action % 73
        
        # Convert from_square to row, col
        from_row, from_col = divmod(from_square, 8)
        from_chess_square = chess.square(from_col, 7 - from_row)  # chess uses bottom-left origin
        
        # Handle different move types
        if move_type < 56:  # Queen moves
            direction_idx = move_type // 7
            distance = (move_type % 7) + 1
            dr, dc = self.queen_directions[direction_idx]
            
            to_row = from_row + dr * distance
            to_col = from_col + dc * distance
            
            # Check if the target square is valid
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_chess_square = chess.square(to_col, 7 - to_row)
                return chess.Move(from_chess_square, to_chess_square)
                
        elif move_type < 64:  # Knight moves
            knight_idx = move_type - 56
            dr, dc = self.knight_moves[knight_idx]
            
            to_row = from_row + dr
            to_col = from_col + dc
            
            # Check if the target square is valid
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_chess_square = chess.square(to_col, 7 - to_row)
                return chess.Move(from_chess_square, to_chess_square)
                
        else:  # Underpromotions (move_type 64-72)
            underpromotion_idx = move_type - 64
            direction_idx = underpromotion_idx // 3
            piece_idx = underpromotion_idx % 3
            
            dr, dc = self.underpromotion_moves[direction_idx]
            promotion_piece = self.underpromotion_pieces[piece_idx]
            
            to_row = from_row + dr
            to_col = from_col + dc
            
            # Check if the target square is valid
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_chess_square = chess.square(to_col, 7 - to_row)
                # Check if this is a pawn promotion (moving to the last rank)
                if (self.board.piece_at(from_chess_square).piece_type == chess.PAWN and
                    (to_row == 0 or to_row == 7)):
                    return chess.Move(from_chess_square, to_chess_square, promotion=promotion_piece)
                else:
                    return chess.Move(from_chess_square, to_chess_square)
        
        # If we couldn't decode the action, return a null move
        return chess.Move.null()
        
    def _move_to_action(self, move):
        """
        Convert chess.Move object to action index.
        
        Args:
            move (chess.Move): Chess move
            
        Returns:
            int: Action index (0-4671)
        """
        # Convert chess squares to row, col
        from_chess_square = move.from_square
        to_chess_square = move.to_square
        
        from_col = chess.square_file(from_chess_square)
        from_row = 7 - chess.square_rank(from_chess_square)  # Convert to top-left origin
        to_col = chess.square_file(to_chess_square)
        to_row = 7 - chess.square_rank(to_chess_square)
        
        # Calculate from_square index (0-63)
        from_square = from_row * 8 + from_col
        
        # Calculate move type
        dr = to_row - from_row
        dc = to_col - from_col
        
        # Check for underpromotion
        piece = self.board.piece_at(from_chess_square)
        if (move.promotion is not None and
            piece is not None and
            piece.piece_type == chess.PAWN):
            # Underpromotion move
            # Find direction
            direction = (dr, dc)
            if direction in self.underpromotion_moves:
                direction_idx = self.underpromotion_moves.index(direction)
                # Find piece
                if move.promotion in self.underpromotion_pieces:
                    piece_idx = self.underpromotion_pieces.index(move.promotion)
                    move_type = 64 + direction_idx * 3 + piece_idx
                    return from_square * 73 + move_type
        
        # Check for knight move
        move_vector = (dr, dc)
        if move_vector in self.knight_moves:
            knight_idx = self.knight_moves.index(move_vector)
            move_type = 56 + knight_idx
            return from_square * 73 + move_type
            
        # Check for queen move (including rook and bishop moves)
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):  # Horizontal, vertical, or diagonal
            # Determine direction
            if dr != 0:
                dr_sign = dr // abs(dr) if dr != 0 else 0
            else:
                dr_sign = 0
                
            if dc != 0:
                dc_sign = dc // abs(dc) if dc != 0 else 0
            else:
                dc_sign = 0
                
            direction = (dr_sign, dc_sign)
            
            if direction in self.queen_directions:
                direction_idx = self.queen_directions.index(direction)
                distance = max(abs(dr), abs(dc)) - 1  # 0-6 for distances 1-7
                move_type = direction_idx * 7 + distance
                return from_square * 73 + move_type
        
        # If we couldn't encode the move, return 0
        return 0
        
    def _get_reward(self):
        """
        Calculate reward based on game outcome.
        
        Returns:
            float: Reward value (+1 for win, -1 for loss, 0 for draw/ongoing)
        """
        if self.board.is_checkmate():
            # If it's checkmate, the player who just moved won
            # Since we just executed a move, if it's checkmate, the current player lost
            return -1.0 if self.board.turn == chess.WHITE else 1.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
            return 0.0
        else:
            return 0.0
            
    def _is_game_over(self):
        """
        Check if game has ended.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.board.is_game_over()
        
    def get_legal_actions(self):
        """
        Get list of legal action indices.
        
        Returns:
            list: List of legal action indices
        """
        legal_moves = list(self.board.legal_moves)
        legal_actions = []
        
        for move in legal_moves:
            action = self._move_to_action(move)
            if 0 <= action < self.action_space_size:
                legal_actions.append(action)
                
        return legal_actions
        
    def get_legal_moves_mask(self):
        """
        Get a mask of legal actions for the current board position.
        
        Returns:
            np.array: Boolean array of shape (4672,) where True indicates legal action
        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        legal_actions = self.get_legal_actions()
        mask[legal_actions] = True
        return mask
        
    def is_action_legal(self, action):
        """
        Check if an action is legal in the current position.
        
        Args:
            action (int): Action index
            
        Returns:
            bool: True if action is legal, False otherwise
        """
        if action < 0 or action >= self.action_space_size:
            return False
            
        move = self._action_to_move(action)
        return move in self.board.legal_moves
        
    def state_to_board(self, state):
        """
        Convert state tensor back to a chess.Board representation (for debugging/visualization).
        
        Args:
            state (np.array): 8x8x19 tensor representation
            
        Returns:
            chess.Board: Corresponding chess board
        """
        # Create a new board
        board = chess.Board(None)  # Empty board
        
        # Set up the board position from the state tensor
        # This is a simplified reconstruction for visualization purposes
        board.clear()
        
        # Place pieces based on channels 0-11
        for row in range(8):
            for col in range(8):
                # Convert from tensor coordinates to chess coordinates
                chess_square = chess.square(col, 7 - row)
                
                # Check piece channels (0-11)
                piece = None
                for channel in range(12):
                    if state[row, col, channel] == 1:
                        # Determine piece type and color
                        piece_type = (channel % 6) + 1  # 1-6 for pawn-king
                        piece_color = chess.WHITE if channel < 6 else chess.BLACK
                        piece = chess.Piece(piece_type, piece_color)
                        break
                        
                if piece:
                    board.set_piece_at(chess_square, piece)
                    
        # Set player to move (channel 12)
        if state[0, 0, 12] == 0:  # If black to move
            board.turn = chess.BLACK
            
        # Set castling rights (channels 13-16)
        # This is a simplified implementation
        if state[0, 0, 13] == 1:  # White kingside
            board.castling_rights |= chess.BB_H1
        if state[0, 0, 14] == 1:  # White queenside
            board.castling_rights |= chess.BB_A1
        if state[0, 0, 15] == 1:  # Black kingside
            board.castling_rights |= chess.BB_H8
        if state[0, 0, 16] == 1:  # Black queenside
            board.castling_rights |= chess.BB_A8
            
        # Set en passant target (channel 17)
        for row in range(8):
            for col in range(8):
                if state[row, col, 17] == 1:
                    chess_square = chess.square(col, 7 - row)
                    board.ep_square = chess_square
                    break
                    
        return board
        
    def visualize_state(self, state=None):
        """
        Visualize the board state.
        
        Args:
            state (np.array, optional): 8x8x19 tensor representation.
                                       If None, uses current board state.
        """
        if state is None:
            print(self.board)
        else:
            board = self.state_to_board(state)
            print(board)
            
    def get_state_statistics(self, state):
        """
        Get statistics about the state tensor.
        
        Args:
            state (np.array): 8x8x19 tensor representation
            
        Returns:
            dict: Dictionary with state statistics
        """
        stats = {
            'piece_count': np.sum(state[:, :, :12]),
            'white_pieces': np.sum(state[:, :, :6]),
            'black_pieces': np.sum(state[:, :, 6:12]),
            'player_to_move': 'White' if state[0, 0, 12] == 1 else 'Black',
            'castling_rights': np.sum(state[:, :, 13:17]),
            'en_passant_targets': np.sum(state[:, :, 17]),
            'halfmove_clock': np.mean(state[:, :, 18]) * 100
        }
        return stats
        
    def get_result(self):
        """
        Get the result of the game.
        
        Returns:
            str: Game result ("1-0", "0-1", "1/2-1/2", or "*" for ongoing)
        """
        if self.board.is_checkmate():
            return "0-1" if self.board.turn == chess.WHITE else "1-0"
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
            return "1/2-1/2"
        else:
            return "*"