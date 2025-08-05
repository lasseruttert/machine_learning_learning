import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, List
import time
import os


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    env_name: str = "CartPole-v1"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    total_timesteps: int = 500000
    num_envs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    torch_deterministic: bool = True
    normalize_observations: bool = True
    normalize_rewards: bool = True
    log_dir: str = "runs"
    save_freq: int = 10000
    eval_freq: int = 10000
    eval_episodes: int = 10
    use_async_envs: bool = False
    lr_schedule: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.n_steps % self.batch_size == 0, "n_steps must be divisible by batch_size"
        assert self.batch_size >= 1, "batch_size must be at least 1"
        assert self.num_envs >= 1, "num_envs must be at least 1"


class RunningMeanStd:
    """Tracks running mean and std for observation/reward normalization"""
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize network layers with orthogonal weights"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteAgent(nn.Module):
    """Agent for discrete action spaces"""
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        n_actions = envs.single_action_space.n
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class ContinuousAgent(nn.Module):
    """Agent for continuous action spaces"""
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PPO:
    """Proximal Policy Optimization implementation"""
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Set seeds
        self._set_seeds()
        
        # Create environments
        self.envs = self._make_envs()
        
        # Determine if continuous or discrete
        self.is_continuous = isinstance(self.envs.single_action_space, gym.spaces.Box)
        
        # Create agent
        if self.is_continuous:
            self.agent = ContinuousAgent(self.envs).to(config.device)
        else:
            self.agent = DiscreteAgent(self.envs).to(config.device)
            
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        
        # Learning rate scheduler
        self.num_updates = config.total_timesteps // config.n_steps // config.num_envs
        if config.lr_schedule:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.num_updates
            )
        
        # Normalization
        if config.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=self.envs.single_observation_space.shape)
        if config.normalize_rewards:
            self.reward_rms = RunningMeanStd(shape=())
            
        # Storage (keep on CPU to save GPU memory)
        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape
        
        self.obs = torch.zeros((config.n_steps, config.num_envs) + obs_shape)
        self.actions = torch.zeros((config.n_steps, config.num_envs) + act_shape)
        self.logprobs = torch.zeros((config.n_steps, config.num_envs))
        self.rewards = torch.zeros((config.n_steps, config.num_envs))
        self.dones = torch.zeros((config.n_steps, config.num_envs))
        self.values = torch.zeros((config.n_steps, config.num_envs))
        
        # Logging
        self.writer = SummaryWriter(f"{config.log_dir}/{config.env_name}_{config.seed}_{int(time.time())}")
        self.global_step = 0
        self.start_time = time.time()
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic
        np.random.seed(self.config.seed)
        
    def _make_envs(self) -> gym.vector.VectorEnv:
        """Create vectorized environments"""
        def make_env(seed: int):
            def thunk():
                env = gym.make(self.config.env_name)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            return thunk
        
        if self.config.use_async_envs and self.config.num_envs > 1:
            envs = gym.vector.AsyncVectorEnv([make_env(self.config.seed + i) for i in range(self.config.num_envs)])
        else:
            envs = gym.vector.SyncVectorEnv([make_env(self.config.seed + i) for i in range(self.config.num_envs)])
            
        return envs
    
    def collect_rollouts(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect experience rollouts"""
        for step in range(self.config.n_steps):
            self.global_step += self.config.num_envs
            
            # Normalize observations if enabled
            if self.config.normalize_observations:
                next_obs_np = next_obs.cpu().numpy()
                self.obs_rms.update(next_obs_np)
                norm_obs = self.obs_rms.normalize(next_obs_np)
                norm_obs = torch.from_numpy(norm_obs).float().to(self.config.device)
            else:
                norm_obs = next_obs.to(self.config.device)
            
            self.obs[step] = next_obs.cpu()
            self.dones[step] = next_done.cpu()
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(norm_obs)
                self.values[step] = value.flatten().cpu()
                
            self.actions[step] = action.cpu()
            self.logprobs[step] = logprob.cpu()
            
            # Execute action
            if self.is_continuous:
                clipped_action = action.cpu().numpy()
                # Clip actions to valid range
                clipped_action = np.clip(clipped_action, 
                                       self.envs.single_action_space.low,
                                       self.envs.single_action_space.high)
                next_obs, reward, terminations, truncations, infos = self.envs.step(clipped_action)
            else:
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                
            done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward)
            next_obs = torch.from_numpy(next_obs).float()
            next_done = torch.from_numpy(done).float()
            
            # Track episode statistics - this is the critical part
            for idx, info in enumerate(infos):
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.writer.add_scalar("charts/episodic_return", ep_reward, self.global_step)
                    self.writer.add_scalar("charts/episodic_length", ep_length, self.global_step)
                    
                    # Debug logging
                    if len(self.episode_rewards) % 100 == 0:
                        recent_rewards = self.episode_rewards[-100:]
                        perfect_count = sum(1 for r in recent_rewards if r >= 499)
                        print(f"[DEBUG] Episodes: {len(self.episode_rewards)}, "
                              f"Recent 100 avg: {np.mean(recent_rewards):.1f}, "
                              f"Perfect scores: {perfect_count}/100")
                        
        return next_obs, next_done
    
    def compute_gae(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        with torch.no_grad():
            # Normalize next observation if needed
            if self.config.normalize_observations:
                next_obs_np = next_obs.cpu().numpy()
                norm_next_obs = self.obs_rms.normalize(next_obs_np)
                norm_next_obs = torch.from_numpy(norm_next_obs).float().to(self.config.device)
            else:
                norm_next_obs = next_obs.to(self.config.device)
                
            next_value = self.agent.get_value(norm_next_obs).reshape(1, -1).cpu()
            
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            
            for t in reversed(range(self.config.n_steps)):
                if t == self.config.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                    
                delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                
            returns = advantages + self.values
                
        return advantages, returns
    
    def update_policy(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """Update policy and value networks"""
        # Flatten batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # Normalize advantages for entire batch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Move returns to device for value loss computation
        b_returns = b_returns.to(self.config.device)
        
        # Optimization
        b_inds = np.arange(self.config.n_steps * self.config.num_envs)
        clipfracs = []
        
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, len(b_inds), self.config.batch_size):
                end = start + self.config.batch_size
                mb_inds = b_inds[start:end]
                
                # Normalize observations if needed
                mb_obs = b_obs[mb_inds]
                if self.config.normalize_observations:
                    mb_obs_np = mb_obs.numpy()
                    norm_mb_obs = self.obs_rms.normalize(mb_obs_np)
                    mb_obs = torch.from_numpy(norm_mb_obs).float()
                    
                mb_obs = mb_obs.to(self.config.device)
                mb_actions = b_actions[mb_inds].to(self.config.device)
                
                if not self.is_continuous:
                    mb_actions = mb_actions.long()
                    
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(mb_obs, mb_actions)
                logratio = newlogprob - b_logprobs[mb_inds].to(self.config.device)
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds].to(self.config.device)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds].to(self.config.device)
                
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values,
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                print(f"Early stopping at epoch {epoch} due to reaching target KL: {approx_kl:.3f}")
                break
        
        # Learning rate scheduling
        if self.config.lr_schedule:
            self.scheduler.step()
            
        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log metrics
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        
        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }
    
    def evaluate(self, n_episodes: int = 10) -> List[float]:
        """Evaluate the current policy"""
        eval_env = gym.make(self.config.env_name)
        eval_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    # Normalize observation if needed
                    if self.config.normalize_observations:
                        norm_obs = self.obs_rms.normalize(obs)
                        obs_tensor = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.config.device)
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                        
                    action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                    
                    if self.is_continuous:
                        action = action.cpu().numpy()[0]
                        action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
                    else:
                        action = action.cpu().numpy().item()
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            
        eval_env.close()
        return eval_rewards
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }
        
        if self.config.normalize_observations:
            checkpoint['obs_rms'] = self.obs_rms
        if self.config.normalize_rewards:
            checkpoint['reward_rms'] = self.reward_rms
            
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if 'obs_rms' in checkpoint:
            self.obs_rms = checkpoint['obs_rms']
        if 'reward_rms' in checkpoint:
            self.reward_rms = checkpoint['reward_rms']
            
        print(f"Checkpoint loaded from {path}")
    
    def train(self) -> List[float]:
        """Main training loop"""
        print(f"Starting PPO training with config:")
        print(f"  Total steps: {self.config.total_timesteps}")
        print(f"  Updates: {self.num_updates}")
        print(f"  Minibatch updates per epoch: {self.config.n_steps * self.config.num_envs // self.config.batch_size}")
        print(f"Device: {self.config.device}")
        print(f"Number of environments: {self.config.num_envs}")
        
        # Initialize
        next_obs, _ = self.envs.reset(seed=self.config.seed)
        next_obs = torch.from_numpy(next_obs).float()
        next_done = torch.zeros(self.config.num_envs)
        
        # Training loop
        for update in range(1, self.num_updates + 1):
            # Collect rollouts
            next_obs, next_done = self.collect_rollouts(next_obs, next_done)
            
            # Compute advantages
            advantages, returns = self.compute_gae(next_obs, next_done)
            
            # Update policy
            metrics = self.update_policy(advantages, returns)
            
            # Logging
            if update % 10 == 0:
                sps = int(self.global_step / (time.time() - self.start_time))
                print(f"Update {update:4d}/{self.num_updates} | Steps {self.global_step:7d} | "
                      f"SPS {sps:4d} | PG Loss {metrics['pg_loss']:.4f} | "
                      f"V Loss {metrics['v_loss']:.4f} | KL {metrics['approx_kl']:.4f}")
                
                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-100:]
                    print(f"  Avg Return (last 100 eps): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
            
            # Quick evaluation to verify training performance
            if update % 50 == 0:
                print("\n[Quick Eval] Testing current policy...")
                quick_test_rewards = []
                for _ in range(10):
                    test_obs, _ = self.envs.reset()
                    test_obs = torch.from_numpy(test_obs).float()
                    done = False
                    total_reward = np.zeros(self.config.num_envs)
                    
                    while not done:
                        with torch.no_grad():
                            norm_test_obs = test_obs.to(self.config.device)
                            action, _, _, _ = self.agent.get_action_and_value(norm_test_obs)
                            test_obs, reward, term, trunc, _ = self.envs.step(action.cpu().numpy())
                            test_obs = torch.from_numpy(test_obs).float()
                            total_reward += reward
                            done = np.any(np.logical_or(term, trunc))
                    
                    quick_test_rewards.extend(total_reward)
                
                print(f"[Quick Eval] Mean: {np.mean(quick_test_rewards):.1f} ± {np.std(quick_test_rewards):.1f}")
                print(f"[Quick Eval] Perfect scores: {sum(1 for r in quick_test_rewards if r >= 499)}/{len(quick_test_rewards)}\n")
            
            # Evaluation
            if self.config.eval_freq > 0 and self.global_step % self.config.eval_freq == 0:
                eval_rewards = self.evaluate(self.config.eval_episodes)
                eval_mean = np.mean(eval_rewards)
                eval_std = np.std(eval_rewards)
                print(f"\nEvaluation at step {self.global_step}: {eval_mean:.2f} ± {eval_std:.2f}\n")
                self.writer.add_scalar("eval/mean_reward", eval_mean, self.global_step)
                self.writer.add_scalar("eval/std_reward", eval_std, self.global_step)
            
            # Save checkpoint
            if self.config.save_freq > 0 and self.global_step % self.config.save_freq == 0:
                os.makedirs("checkpoints", exist_ok=True)
                self.save_checkpoint(f"checkpoints/{self.config.env_name}_step_{self.global_step}.pt")
        
        self.envs.close()
        self.writer.close()
        return self.episode_rewards
    
    def test(self, n_episodes: int = 10, deterministic: bool = True) -> List[float]:
        """Test the trained policy"""
        test_env = gym.make(self.config.env_name, render_mode="human")
        test_rewards = []
        
        print(f"\nTesting for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    if self.config.normalize_observations:
                        norm_obs = self.obs_rms.normalize(obs)
                        obs_tensor = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.config.device)
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                    
                    if deterministic and not self.is_continuous:
                        # For discrete actions, take argmax during testing
                        logits = self.agent.actor(obs_tensor)
                        action = torch.argmax(logits, dim=1)
                    elif deterministic and self.is_continuous:
                        # For continuous actions, use mean
                        action = self.agent.actor_mean(obs_tensor)
                    else:
                        # Sample from distribution (same as training)
                        action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                    
                    if self.is_continuous:
                        action = action.cpu().numpy()[0]
                        action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
                    else:
                        action = action.cpu().numpy().item()
                
                obs, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            test_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        test_env.close()
        
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        print(f"\nAverage Test Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min: {min(test_rewards):.2f}, Max: {max(test_rewards):.2f}")
        
        return test_rewards


def main():
    # Example configurations for different environments
    
    # CartPole-v1 - Fast convergence config
    # config = PPOConfig(
    #     env_name="CartPole-v1",
    #     total_timesteps=50000,   # Should be enough for CartPole
    #     learning_rate=3e-4,      # Standard LR for simple tasks
    #     n_steps=32,              # Very short rollouts for fast updates
    #     batch_size=32,           # Full batch updates
    #     n_epochs=10,             # Thorough optimization per update
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_coef=0.2,
    #     ent_coef=0.00,            # No entropy needed for CartPole
    #     vf_coef=1.0,             # Strong value function fitting
    #     max_grad_norm=0.5,
    #     num_envs=4,
    #     normalize_observations=True,
    #     normalize_rewards=True,
    #     eval_freq=10000,
    #     target_kl=0.02,
    #     lr_schedule=False,       # No decay needed for short training
    #     seed=42
    # )
    
    # # Alternative: Stable but slightly slower
    # config = PPOConfig(
    #     env_name="CartPole-v1",
    #     total_timesteps=75000,
    #     learning_rate=2.5e-4,
    #     n_steps=64,
    #     batch_size=64,
    #     n_epochs=4,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_coef=0.2,
    #     ent_coef=0.01,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     num_envs=4,
    #     normalize_observations=False,
    #     normalize_rewards=False,
    #     target_kl=0.02,
    #     seed=42
    # )
    
    # # Pendulum-v1 (Continuous)
    config = PPOConfig(
        env_name="Pendulum-v1",
        total_timesteps=200000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        num_envs=4,
        normalize_observations=True,
        normalize_rewards=False,
        seed=42
    )
    
    # # LunarLander-v2 (Discrete)
    # config = PPOConfig(
    #     env_name="LunarLander-v2",
    #     total_timesteps=500000,
    #     learning_rate=2.5e-4,
    #     n_steps=1024,
    #     batch_size=64,
    #     n_epochs=4,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_coef=0.2,
    #     ent_coef=0.01,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     num_envs=8,
    #     normalize_observations=False,
    #     normalize_rewards=False,
    #     target_kl=0.015,
    #     seed=42
    # )
    
    # Initialize and train
    print(f"Starting PPO training with config:")
    print(f"  Total steps: {config.total_timesteps}")
    print(f"  Updates: {config.total_timesteps // (config.n_steps * config.num_envs)}")
    print(f"  Minibatch updates per epoch: {config.n_steps * config.num_envs // config.batch_size}")
    
    ppo = PPO(config)
    episode_rewards = ppo.train()
    
    # Test the trained agent
    print("\nRunning final evaluation...")
    test_rewards = ppo.test(n_episodes=30, deterministic=True)  # Deterministic testing
    
    # Also test with stochastic policy to match training
    print("\nRunning stochastic evaluation (matching training behavior)...")
    stochastic_rewards = ppo.test(n_episodes=30, deterministic=False)
    
    print(f"\nTraining completed!")
    if episode_rewards:
        # Check when solved (avg >= 195 for 100 consecutive episodes)
        solved_at = None
        if len(episode_rewards) >= 100:
            for i in range(100, len(episode_rewards) + 1):
                if np.mean(episode_rewards[i-100:i]) >= 195:
                    solved_at = i
                    break
                
        final_100_mean = np.mean(episode_rewards[-100:])
        final_100_std = np.std(episode_rewards[-100:])
        print(f"Final 100 training episodes: {final_100_mean:.2f} ± {final_100_std:.2f}")
        
        if solved_at:
            # Calculate actual environment steps when solved
            episodes_per_env = solved_at / config.num_envs
            steps_when_solved = episodes_per_env * config.num_envs * np.mean([len(e) for e in episode_rewards[:solved_at] if e > 0])
            print(f"Environment solved at episode {solved_at} (approx. step {int(steps_when_solved)})")
        
        # Check consistency
        perfect_scores = sum(1 for r in episode_rewards[-100:] if r >= 499)
        print(f"Perfect scores (≥499) in last 100 episodes: {perfect_scores}%")
    
    # Detailed test analysis
    print(f"\nDeterministic Test Performance:")
    perfect_test_scores = sum(1 for r in test_rewards if r >= 499)
    print(f"Perfect scores (≥499): {perfect_test_scores}/{len(test_rewards)} ({100*perfect_test_scores/len(test_rewards):.1f}%)")
    print(f"Mean: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    
    print(f"\nStochastic Test Performance (matches training):")
    perfect_stochastic = sum(1 for r in stochastic_rewards if r >= 499)
    print(f"Perfect scores (≥499): {perfect_stochastic}/{len(stochastic_rewards)} ({100*perfect_stochastic/len(stochastic_rewards):.1f}%)")
    print(f"Mean: {np.mean(stochastic_rewards):.2f} ± {np.std(stochastic_rewards):.2f}")
    
    # Expected performance benchmarks
    print(f"\nPerformance vs Expected:")
    print(f"Expected: Solve (<50k steps), >95% perfect scores")
    print(f"Actual: {config.total_timesteps} steps, {100*perfect_stochastic/len(stochastic_rewards):.1f}% perfect scores (stochastic)")
    
    # This is the real metric that matters
    if np.mean(stochastic_rewards) < 450:
        print("\n⚠️  WARNING: Agent performs well with deterministic actions but poorly with stochastic policy.")
        print("This suggests the policy hasn't properly converged - it's relying on argmax to hide poor exploration.")
        
    # Plot learning curve if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        if len(episode_rewards) > 10:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Learning curve
            window = min(100, len(episode_rewards) // 10)
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label='Smoothed')
            ax1.axhline(y=500, color='r', linestyle='--', label='Max possible')
            ax1.axhline(y=195, color='g', linestyle='--', label='Solved threshold')
            ax1.set_title(f'Learning Curve - {config.env_name}')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True)
            
            # Test performance distribution
            ax2.hist(test_rewards, bins=20, edgecolor='black')
            ax2.axvline(x=np.mean(test_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(test_rewards):.1f}')
            ax2.set_title('Test Performance Distribution')
            ax2.set_xlabel('Reward')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{config.env_name}_training_results.png')
            plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()