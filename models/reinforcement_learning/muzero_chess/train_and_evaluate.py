#!/usr/bin/env python3

import argparse
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chess_trainer import ChessTrainer
from stockfish_evaluator import StockfishEvaluator


def main():
    """Main function to train and evaluate the MuZero chess agent"""
    parser = argparse.ArgumentParser(description='Train and evaluate MuZero chess agent')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate against Stockfish')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--games', type=int, default=3, help='Number of evaluation games')
    parser.add_argument('--model', type=str, default='trained_chess_model.pth', help='Model file path')
    parser.add_argument('--no-visualization', action='store_true', help='Disable training visualization')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not args.train and not args.evaluate:
        parser.print_help()
        return
    
    if args.train:
        print("=" * 50)
        print("Starting MuZero Chess Training")
        print("=" * 50)
        
        # Initialize trainer
        trainer = ChessTrainer(enable_visualization=not args.no_visualization)
        
        # Train model
        trainer.train(num_epochs=args.epochs)
        
        print("=" * 50)
        print("Training Completed!")
        print("=" * 50)
        
    if args.evaluate:
        print("=" * 50)
        print("Starting Evaluation Against Stockfish")
        print("=" * 50)
        
        # Initialize evaluator
        evaluator = StockfishEvaluator(args.model)
        
        # Evaluate model
        evaluator.evaluate(num_games=args.games)
        
        print("=" * 50)
        print("Evaluation Completed!")
        print("=" * 50)


if __name__ == "__main__":
    main()