import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class DQN(nn.Module):
    """Deep Q-Network für Breakout"""
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # Convolutional layers für Bildverarbeitung
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Berechne die Größe für den fully connected layer
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def _get_conv_out(self, shape):
        """Berechnet die Ausgabegröße der Convolutional Layers"""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def preprocess_frame(frame):
    """Preprocesse das Atari-Frame für das Netzwerk"""
    # Konvertiere zu Graustufen und resize
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame

class BreakoutAgent:
    def __init__(self, state_shape, n_actions, learning_rate=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        # Netzwerke
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(100000)
        
        # Hyperparameter
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000
        self.step_count = 0
        
        # Für Frame Stacking (4 Frames)
        self.frame_stack_size = 4
        
    def get_action(self, state):
        """Epsilon-greedy Action Selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def train_step(self):
        """Trainiere das Netzwerk mit einem Batch"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss berechnen
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Target network update
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Speichere das trainierte Modell"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        print(f"Modell gespeichert: {filepath}")
    
    def load_model(self, filepath):
        """Lade ein trainiertes Modell"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        print(f"Modell geladen: {filepath}")

def create_frame_stack(frames):
    """Erstelle einen Stack von 4 Frames"""
    return np.stack(frames, axis=0)

def train_breakout_agent(episodes=1000, render=False):
    """Haupttraining-Loop"""
    # Environment erstellen - verwende die klassische Breakout-Umgebung
    try:
        env = gym.make('ALE/Breakout-v5', render_mode='human' if render else None)
    except:
        # Fallback auf die klassische Version
        env = gym.make('Breakout-v4', render_mode='human' if render else None)
    
    # Agent initialisieren
    state_shape = (4, 84, 84)  # 4 gestapelte Frames
    n_actions = env.action_space.n
    agent = BreakoutAgent(state_shape, n_actions)
    
    # Training statistics
    scores = []
    episode_lengths = []
    
    print(f"Training beginnt mit {n_actions} möglichen Aktionen")
    print(f"Aktionen: {list(range(n_actions))}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        
        # Initialisiere Frame Stack mit dem ersten Frame
        frame_stack = [state] * 4
        current_state = create_frame_stack(frame_stack)
        
        total_reward = 0
        steps = 0
        
        while True:
            # Action auswählen
            action = agent.get_action(current_state)
            
            # Step in der Umgebung
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Frame preprocessen
            next_state = preprocess_frame(next_state)
            
            # Frame Stack aktualisieren
            frame_stack.pop(0)
            frame_stack.append(next_state)
            next_state_stack = create_frame_stack(frame_stack)
            
            # Experience zum Replay Buffer hinzufügen
            agent.replay_buffer.push(current_state, action, reward, next_state_stack, done)
            
            # Training step
            agent.train_step()
            
            current_state = next_state_stack
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        episode_lengths.append(steps)
        
        # Statistiken ausgeben
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Avg Length: {avg_length:.1f}, Epsilon: {agent.epsilon:.3f}")
        
        # Modell regelmäßig speichern
        if episode % 500 == 0 and episode > 0:
            agent.save_model(f'breakout_model_episode_{episode}.pth')
    
    env.close()
    
    # Finale Statistiken plotten
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    # Moving average für bessere Visualisierung
    window = 100
    if len(scores) >= window:
        moving_avg = [np.mean(scores[i:i+window]) for i in range(len(scores)-window+1)]
        plt.plot(moving_avg)
        plt.title(f'Moving Average Score (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.show()
    
    return agent, scores

def test_agent(model_path, episodes=5):
    """Teste einen trainierten Agent"""
    try:
        env = gym.make('ALE/Breakout-v5', render_mode='human')
    except:
        env = gym.make('Breakout-v4', render_mode='human')
    
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n
    agent = BreakoutAgent(state_shape, n_actions)
    agent.load_model(model_path)
    agent.epsilon = 0  # Keine Exploration beim Testen
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_frame(state)
        
        frame_stack = [state] * 4
        current_state = create_frame_stack(frame_stack)
        
        total_reward = 0
        
        while True:
            action = agent.get_action(current_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocess_frame(next_state)
            frame_stack.pop(0)
            frame_stack.append(next_state)
            current_state = create_frame_stack(frame_stack)
            
            total_reward += reward
            
            if done:
                break
        
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    print("Breakout DQN Training")
    print("=" * 50)
    
    # Training starten
    # Für schnelles Testen: weniger Episodes
    agent, scores = train_breakout_agent(episodes=500, render=True)
    
    # Finales Modell speichern
    agent.save_model('breakout_final_model.pth')
    
    print("\nTraining abgeschlossen!")
    print(f"Finaler Score: {scores[-1]}")
    print(f"Durchschnittlicher Score (letzte 100 Episodes): {np.mean(scores[-100:]):.2f}")
    
    # Optional: Teste das trainierte Modell
    test_agent('breakout_final_model.pth', episodes=3)