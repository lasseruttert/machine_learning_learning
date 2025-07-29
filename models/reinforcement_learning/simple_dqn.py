import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# Neuronales Netzwerk für Q-Werte
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        """Speichere Erfahrungen für späteren Replay"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Wähle eine Aktion: Exploration vs Exploitation"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Trainiere das Netzwerk mit gespeicherten Erfahrungen"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reduziere Exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Kopiere Weights vom Haupt- zum Target-Netzwerk"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training
def train_dqn():
    # Environment Setup
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Agent erstellen
    agent = DQNAgent(state_size, action_size)
    
    # Training Parameters
    episodes = 500
    max_steps = 500
    target_update_freq = 10
    
    scores = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Gym gibt manchmal (state, info) zurück
        
        total_reward = 0
        
        for step in range(max_steps):
            # Action wählen und ausführen
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Modifiziere Reward für besseres Lernen
            reward = reward if not done else -10
            
            # Erfahrung speichern
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Training
            if len(agent.memory) > 32:
                agent.replay()
            
            if done:
                break
        
        scores.append(total_reward)
        avg_score = np.mean(scores)
        
        # Target Network Update
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Progress ausgeben
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {total_reward:.0f}, " f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Erfolgreich gelöst?
        if avg_score >= 195.0 and len(scores) >= 100:
            print(f"\nUmgebung gelöst in {episode} Episoden!")
            break
    
    env.close()
    return agent

# Teste den trainierten Agent
def test_agent(agent, episodes=5):
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        done = False
        
        while not done:
            env.render()
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    print("Training DQN Agent auf CartPole...")
    trained_agent = train_dqn()
    
    print("\nTeste den trainierten Agent...")
    trained_agent.epsilon = 0  # Keine Exploration beim Testen
    test_agent(trained_agent)