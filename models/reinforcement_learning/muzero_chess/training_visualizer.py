import matplotlib.pyplot as plt
import numpy as np
import threading
import time


class TrainingVisualizer:
    def __init__(self, enable_visualization=True):
        """Initialize the training visualizer"""
        self.epochs = []
        self.avg_lengths = []
        self.avg_rewards = []
        self.enable_visualization = enable_visualization
        self.lock = threading.Lock()
        
        if self.enable_visualization:
            # Set up the plot
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
            self.fig.suptitle('MuZero Chess Training Progress')
            
    def update_progress(self, epoch, avg_length, avg_reward):
        """
        Update training progress visualization.
        
        Args:
            epoch (int): Current epoch number
            avg_length (float): Average game length
            avg_reward (float): Average reward
        """
        if not self.enable_visualization:
            return
            
        with self.lock:
            self.epochs.append(epoch)
            self.avg_lengths.append(avg_length)
            self.avg_rewards.append(avg_reward)
            
            # Update plots
            self.ax1.clear()
            self.ax1.plot(self.epochs, self.avg_lengths, 'b-')
            self.ax1.set_title('Average Game Length per Epoch')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Average Length')
            self.ax1.grid(True)
            
            self.ax2.clear()
            self.ax2.plot(self.epochs, self.avg_rewards, 'r-')
            self.ax2.set_title('Average Reward per Epoch')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Average Reward')
            self.ax2.grid(True)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
            
    def save_plot(self, filename):
        """
        Save the training progress plot.
        
        Args:
            filename (str): Filename to save the plot
        """
        if self.enable_visualization:
            plt.savefig(filename)
            
    def close(self):
        """Close the visualization"""
        if self.enable_visualization:
            plt.ioff()
            plt.close()