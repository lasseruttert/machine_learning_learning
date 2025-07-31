import pygame
import chess
import sys
import os
import time


class ChessVisualizer:
    def __init__(self, width=640, height=640):
        """
        Initialize the chess visualization system.
        
        Args:
            width (int): Width of the visualization window
            height (int): Height of the visualization window
        """
        self.width = width
        self.height = height
        self.square_size = width // 8
        
        # Colors
        self.light_square = (240, 217, 181)
        self.dark_square = (181, 136, 99)
        self.highlight_color = (100, 200, 100, 128)  # Semi-transparent green
        self.text_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MuZero Chess Visualization")
        self.clock = pygame.time.Clock()
        
        # Font for text rendering
        # Try to use a font that supports Unicode chess symbols
        try:
            # Try to use a system font that supports chess symbols
            self.font = pygame.font.SysFont('segoe ui symbol,dejavusans,arial', 36)
        except:
            # Fallback to default font if specific fonts are not available
            self.font = pygame.font.Font(None, 36)
        
        # Piece symbols for text rendering (Unicode chess symbols)
        # White pieces (uppercase) and black pieces (lowercase)
        self.piece_symbols = {
            chess.PAWN: ('♙', '♟'),
            chess.KNIGHT: ('♘', '♞'),
            chess.BISHOP: ('♗', '♝'),
            chess.ROOK: ('♖', '♜'),
            chess.QUEEN: ('♕', '♛'),
            chess.KING: ('♔', '♚')
        }
        
        # Fallback ASCII symbols for cases where Unicode symbols don't render
        self.fallback_symbols = {
            chess.PAWN: ('P', 'p'),
            chess.KNIGHT: ('N', 'n'),
            chess.BISHOP: ('B', 'b'),
            chess.ROOK: ('R', 'r'),
            chess.QUEEN: ('Q', 'q'),
            chess.KING: ('K', 'k')
        }
        
        # Animation settings
        self.animation_speed = 2  # Moves per second
        self.is_animating = False
        self.animation_progress = 0.0
        self.animation_start = None
        self.animation_end = None
        self.moving_piece = None
        
    def initialize_board(self):
        """Set up the pygame window and board display."""
        self.screen.fill(self.bg_color)
        pygame.display.flip()
        
    def draw_board(self, board, highlighted_squares=None):
        """
        Draw the chess board with pieces.
        
        Args:
            board (chess.Board): Current board state
            highlighted_squares (list): List of squares to highlight
        """
        if highlighted_squares is None:
            highlighted_squares = []
            
        # Draw squares
        for row in range(8):
            for col in range(8):
                # Convert to pygame coordinates (top-left origin)
                x = col * self.square_size
                y = row * self.square_size
                
                # Determine square color
                is_light = (row + col) % 2 == 0
                color = self.light_square if is_light else self.dark_square
                
                # Draw square
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                
                # Highlight squares if needed
                square = chess.square(col, 7 - row)
                if square in highlighted_squares:
                    # Create a semi-transparent surface for highlighting
                    highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                    highlight.fill(self.highlight_color)
                    self.screen.blit(highlight, (x, y))
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                
                if piece:
                    self.draw_piece(piece, col, row)
                    
        # Draw coordinates - removed as per task requirements
        # self.draw_coordinates()
        
    def draw_piece(self, piece, col, row):
        """
        Draw a chess piece at the specified position.
        
        Args:
            piece (chess.Piece): The piece to draw
            col (int): Column position (0-7)
            row (int): Row position (0-7)
        """
        # For now, use text rendering as a simple implementation
        # In a more advanced version, we would use actual piece images
        
        x = col * self.square_size
        y = row * self.square_size
        
        # Determine piece symbol
        piece_type = piece.piece_type
        symbol_tuple = self.piece_symbols.get(piece_type, ('?', '?'))
        symbol = symbol_tuple[0] if piece.color == chess.WHITE else symbol_tuple[1]
        
        # Determine color (white pieces are white, black pieces are black)
        piece_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
        
        # Try to render the Unicode symbol
        try:
            text = self.font.render(symbol, True, piece_color)
            # Check if the rendered text is empty or contains only a placeholder
            if text.get_size()[0] == 0 or symbol == '?':
                # Use fallback symbols
                fallback_tuple = self.fallback_symbols.get(piece_type, ('?', '?'))
                symbol = fallback_tuple[0] if piece.color == chess.WHITE else fallback_tuple[1]
                text = self.font.render(symbol, True, piece_color)
        except:
            # If Unicode rendering fails, use fallback symbols
            fallback_tuple = self.fallback_symbols.get(piece_type, ('?', '?'))
            symbol = fallback_tuple[0] if piece.color == chess.WHITE else fallback_tuple[1]
            text = self.font.render(symbol, True, piece_color)
        
        text_rect = text.get_rect(center=(x + self.square_size // 2, y + self.square_size // 2))
        self.screen.blit(text, text_rect)
        
            
    def start_move_animation(self, from_square, to_square, piece):
        """
        Start animating a piece move.
        
        Args:
            from_square (int): Starting square
            to_square (int): Ending square
            piece (chess.Piece): The piece being moved
        """
        self.is_animating = True
        self.animation_progress = 0.0
        self.animation_start = self.square_to_coords(from_square)
        self.animation_end = self.square_to_coords(to_square)
        self.moving_piece = piece
        
    def update_animation(self, dt):
        """
        Update the animation progress.
        
        Args:
            dt (float): Time delta in seconds
        """
        if self.is_animating:
            # Update progress based on animation speed
            self.animation_progress += dt * self.animation_speed
            
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                self.is_animating = False
                
    def draw_animated_piece(self):
        """Draw the moving piece at its current animated position."""
        if self.is_animating and self.moving_piece:
            # Calculate current position
            start_x, start_y = self.animation_start
            end_x, end_y = self.animation_end
            
            current_x = start_x + (end_x - start_x) * self.animation_progress
            current_y = start_y + (end_y - start_y) * self.animation_progress
            
            # Draw the piece at the animated position
            self.draw_piece_at_position(self.moving_piece, current_x, current_y)
            
    def draw_piece_at_position(self, piece, x, y):
        """
        Draw a piece at a specific position (for animation).
        
        Args:
            piece (chess.Piece): The piece to draw
            x (float): X position
            y (float): Y position
        """
        piece_type = piece.piece_type
        symbol_tuple = self.piece_symbols.get(piece_type, ('?', '?'))
        symbol = symbol_tuple[0] if piece.color == chess.WHITE else symbol_tuple[1]
        piece_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
        
        # Try to render the Unicode symbol
        try:
            text = self.font.render(symbol, True, piece_color)
            # Check if the rendered text is empty or contains only a placeholder
            if text.get_size()[0] == 0 or symbol == '?':
                # Use fallback symbols
                fallback_tuple = self.fallback_symbols.get(piece_type, ('?', '?'))
                symbol = fallback_tuple[0] if piece.color == chess.WHITE else fallback_tuple[1]
                text = self.font.render(symbol, True, piece_color)
        except:
            # If Unicode rendering fails, use fallback symbols
            fallback_tuple = self.fallback_symbols.get(piece_type, ('?', '?'))
            symbol = fallback_tuple[0] if piece.color == chess.WHITE else fallback_tuple[1]
            text = self.font.render(symbol, True, piece_color)
        
        text_rect = text.get_rect(center=(x + self.square_size // 2, y + self.square_size // 2))
        self.screen.blit(text, text_rect)
        
    def square_to_coords(self, square):
        """
        Convert a chess square to screen coordinates.
        
        Args:
            square (int): Chess square (0-63)
            
        Returns:
            tuple: (x, y) screen coordinates
        """
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)  # Flip for pygame coordinates
        return (col * self.square_size, row * self.square_size)
        
    def coords_to_square(self, x, y):
        """
        Convert screen coordinates to a chess square.
        
        Args:
            x (int): X screen coordinate
            y (int): Y screen coordinate
            
        Returns:
            int: Chess square (0-63)
        """
        col = x // self.square_size
        row = y // self.square_size
        return chess.square(col, 7 - row)  # Flip for chess coordinates
        
    def update_display(self, board, highlighted_squares=None):
        """
        Update the display with the current board state.
        
        Args:
            board (chess.Board): Current board state
            highlighted_squares (list): List of squares to highlight
        """
        self.screen.fill(self.bg_color)
        self.draw_board(board, highlighted_squares)
        pygame.display.flip()
        
    def animate_move(self, board, move, duration=0.5):
        """
        Animate a chess move.
        
        Args:
            board (chess.Board): Current board state
            move (chess.Move): Move to animate
            duration (float): Duration of animation in seconds
        """
        # Get the piece being moved
        piece = board.piece_at(move.from_square)
        if not piece:
            return
            
        # Start animation
        self.start_move_animation(move.from_square, move.to_square, piece)
        
        # Animation loop
        start_time = time.time()
        while self.is_animating:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
            # Calculate time delta
            current_time = time.time()
            dt = current_time - start_time
            start_time = current_time
            
            # Update animation
            self.update_animation(dt)
            
            # Draw everything
            self.screen.fill(self.bg_color)
            # Draw board without the moving piece
            temp_board = board.copy()
            temp_board.remove_piece_at(move.from_square)
            self.draw_board(temp_board)
            # Draw animated piece
            self.draw_animated_piece()
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(60)
            
    def run_game(self, moves, delay=1.0):
        """
        Run a complete game with visualization.
        
        Args:
            moves (list): List of chess moves
            delay (float): Delay between moves in seconds
        """
        board = chess.Board()
        
        # Initialize display
        self.initialize_board()
        self.update_display(board)
        
        # Process each move
        for i, move in enumerate(moves):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
            # Highlight the move
            self.update_display(board, [move.from_square, move.to_square])
            pygame.time.wait(500)  # Brief pause before move
            
            # Animate the move
            self.animate_move(board, move)
            
            # Execute move on board
            board.push(move)
            
            # Update display
            self.update_display(board)
            
            # Delay before next move
            pygame.time.wait(int(delay * 1000))
            
        # Keep window open at the end
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            self.clock.tick(60)
            
        pygame.quit()
        
    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                    
    def close(self):
        """Close the visualization window."""
        pygame.quit()


def main():
    """Test the chess visualization with a sample game."""
    # Create a simple game
    moves = [
        chess.Move.from_uci("e2e4"),  # e4
        chess.Move.from_uci("e7e5"),  # e5
        chess.Move.from_uci("g1f3"),  # Nf3
        chess.Move.from_uci("b8c6"),  # Nc6
        chess.Move.from_uci("f1c4"),  # Bc4
        chess.Move.from_uci("f8c5"),  # Bc5
        chess.Move.from_uci("e1g1"),  # O-O
        chess.Move.from_uci("e8g8"),  # O-O
    ]
    
    # Create visualizer and run game
    visualizer = ChessVisualizer()
    visualizer.run_game(moves, delay=1.0)


if __name__ == "__main__":
    main()