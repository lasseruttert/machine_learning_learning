# Chess Environment Design for MuZero

## Overview

This document describes the design of the chess environment that will be used with the MuZero agent. The environment will be built using the python-chess library, which provides a comprehensive implementation of chess rules and game state management.

## Python-Chess Library Integration

The python-chess library (https://python-chess.readthedocs.io/) provides:
- Complete chess rules implementation
- Board representation and move generation
- FEN and PGN support
- Move validation
- Game state tracking (check, checkmate, stalemate, etc.)

## Environment Class Design

```python
class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()  # Initialize a new chess board
        self.action_space = self._get_action_space()
        
    def reset(self):
        """Reset the environment to initial state"""
        self.board = chess.Board()
        return self._get_state()
        
    def step(self, action):
        """Execute an action and return (next_state, reward, done, info)"""
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
        """Convert current board state to tensor representation"""
        # Implementation details...
        pass
        
    def _get_action_space(self):
        """Define the action space for chess"""
        # Implementation details...
        pass
        
    def _action_to_move(self, action):
        """Convert action index to chess.Move object"""
        # Implementation details...
        pass
        
    def _get_reward(self):
        """Calculate reward based on game outcome"""
        # Implementation details...
        pass
        
    def _is_game_over(self):
        """Check if game has ended"""
        return self.board.is_game_over()
```

## State Representation

### Board Encoding
The chess board will be encoded as a tensor representation suitable for neural networks:

1. **Piece-centric representation**:
   - 8x8x12 tensor (8 rows, 8 columns, 12 piece types)
   - Each piece type has a channel: P, N, B, R, Q, K for both colors
   - 1 for occupied squares, 0 for empty squares

2. **Additional state information**:
   - Player to move (1 channel)
   - Castling rights (4 channels)
   - En passant target (1 channel)
   - Halfmove clock and fullmove number (2 channels)

### Total State Tensor
- Shape: 8x8x19 (8 rows, 8 columns, 19 channels)
- Channels:
  - 12 channels for piece positions
  - 1 channel for player to move
  - 4 channels for castling rights
  - 1 channel for en passant target
  - 1 channel for halfmove clock

## Action Space

### Move Encoding
Chess has a large action space (~2000 possible moves in typical positions). We'll use:

1. **Underpromotion handling**:
   - Queen promotions (default)
   - Knight, Bishop, Rook promotions (special encoding)

2. **Action indexing**:
   - Map all possible moves to indices
   - Use a fixed-size action space (e.g., 4672 actions)
   - Include illegal moves that will be masked during action selection

### Action Space Size
- Total actions: 8×8×73 = 4672
- 8×8 for the from square
- 73 for move types:
  - 56 queen moves (up to 7 squares in 8 directions)
  - 8 knight moves
  - 9 underpromotions (3 pieces × 3 directions)

## Reward Structure

### Game Outcome Rewards
- Win: +1.0
- Loss: -1.0
- Draw: 0.0

### Intermediate Rewards (Optional)
- Small positive reward for captures
- Small negative reward for losing material
- Positional rewards based on piece values

## Integration with MuZero

### State Representation for MuZero
The state tensor will be used as input to the representation network:
- Shape: (batch_size, channels, height, width)
- Channels: 19 (as defined above)
- Height: 8
- Width: 8

### Action Representation
Actions will be represented as indices in the fixed action space:
- Integer from 0 to 4671
- Converted to/from UCI moves as needed

## Implementation Plan

1. Create ChessEnvironment class
2. Implement state encoding (board to tensor)
3. Implement action encoding (index to move)
4. Implement reward calculation
5. Add game state tracking
6. Test with simple random play

## Key Considerations

1. **Performance**: Efficient encoding/decoding of states and actions
2. **Correctness**: Proper handling of all chess rules
3. **Compatibility**: Integration with MuZero's requirements
4. **Extensibility**: Easy to modify or extend functionality