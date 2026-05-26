"""
Utility classes for PPO training.
Includes rollout buffer, metrics reporting, and baseline policies.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer


class PPOBuffer:
    """Buffer for storing rollout data using torch tensors for efficiency"""
    
    def __init__(self, size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)

    def store(self, obs, action, reward, value, log_prob, done):
        """Store a single transition"""
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def compute_advantages(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE advantages and returns"""
        buffer_size = self.size if self.full else self.ptr

        rewards = self.rewards[:buffer_size]
        values = self.values[:buffer_size]
        dones = self.dones[:buffer_size]

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(buffer_size)):
            next_non_terminal = 1.0 - dones[t].float()
            next_value = last_value if t == buffer_size - 1 else values[t + 1]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        returns = advantages + values

        # Don't normalize here - will normalize per minibatch during training
        self.advantages[:buffer_size] = advantages
        self.returns[:buffer_size] = returns

    def get_batch(self):
        """Get all data as a batch for training"""
        buffer_size = self.size if self.full else self.ptr
        return {
            'obs': self.obs[:buffer_size],
            'actions': self.actions[:buffer_size],
            'log_probs': self.log_probs[:buffer_size],
            'advantages': self.advantages[:buffer_size],
            'returns': self.returns[:buffer_size],
            'values': self.values[:buffer_size]
        }

    def clear(self):
        """Reset the buffer"""
        self.ptr = 0
        self.full = False


class MetricsReporter:
    """Handles logging and reporting of training metrics"""
    
    def __init__(self, writer: SummaryWriter, ppo_writer: SummaryWriter = None, 
                 baseline_writer: SummaryWriter = None, test_interval: int = 10):
        self.writer = writer
        self.ppo_writer = ppo_writer
        self.baseline_writer = baseline_writer
        self.test_interval = test_interval
        self.training_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

    def log_training_metrics(self, update_count: int, rollout_metrics: Dict, 
                            update_metrics: Dict, timesteps_collected: int, fps: float):
        """Log training metrics to console and TensorBoard"""
        avg_reward = np.mean(rollout_metrics['rewards'])
        avg_value = np.mean(rollout_metrics['values'])
        avg_entropy = np.mean(rollout_metrics['entropies'])
        avg_total_loss = np.mean(update_metrics['total_loss'])
        avg_policy_loss = np.mean(update_metrics['policy_loss'])
        avg_value_loss = np.mean(update_metrics['value_loss'])
        avg_entropy_loss = np.mean(update_metrics['entropy_loss'])
        avg_approx_kl = np.mean(update_metrics['approx_kl'])
        
        # Enhanced console output with more meaningful metrics
        if 'episode_returns' in rollout_metrics and rollout_metrics['episode_returns']:
            avg_episode_return = np.mean(rollout_metrics['episode_returns'])
            num_episodes = len(rollout_metrics['episode_returns'])
            max_episode_return = np.max(rollout_metrics['episode_returns'])
            print(f"Update {update_count:4d} | Timesteps {timesteps_collected:7d} | FPS {fps:6.0f} | "
                  f"Episodes {num_episodes:2d} | EpReturn {avg_episode_return:7.1f} (max: {max_episode_return:7.1f})")
        else:
            # Show cumulative reward (now meaningful with utilization rewards)
            total_reward = sum(rollout_metrics['rewards'])
            avg_reward = np.mean(rollout_metrics['rewards'])

            # Show bucket metrics if available
            bucket_info = ""
            if 'current_buckets' in rollout_metrics and rollout_metrics['current_buckets']:
                avg_buckets = int(np.mean(rollout_metrics['current_buckets']))
                bucket_info += f" | AvgBuckets {avg_buckets:2d}"
            if 'max_buckets' in rollout_metrics and rollout_metrics['max_buckets']:
                max_buckets = max(rollout_metrics['max_buckets'])
                bucket_info += f" | MaxBuckets {max_buckets:2d}"

            print(f"Update {update_count:4d} | Timesteps {timesteps_collected:7d} | FPS {fps:6.0f} | "
                  f"TotalReward {total_reward:7.1f} | AvgReward {avg_reward:6.3f}" + bucket_info)

        self.writer.add_scalar('Training/Total_Loss', avg_total_loss, update_count)
        self.writer.add_scalar('Training/Policy_Loss', avg_policy_loss, update_count)
        self.writer.add_scalar('Training/Value_Loss', avg_value_loss, update_count)
        self.writer.add_scalar('Training/Entropy_Loss', avg_entropy_loss, update_count)
        self.writer.add_scalar('Training/Approx_KL', avg_approx_kl, update_count)
        self.writer.add_scalar('Performance/Reward_Per_Step', avg_reward, update_count)
        self.writer.add_scalar('Performance/Value', avg_value, update_count)
        self.writer.add_scalar('Performance/Entropy', avg_entropy, update_count)
        self.writer.add_scalar('Performance/FPS', fps, update_count)
        
        # Log episode returns if available
        if 'episode_returns' in rollout_metrics and rollout_metrics['episode_returns']:
            avg_episode_return = np.mean(rollout_metrics['episode_returns'])
            max_episode_return = np.max(rollout_metrics['episode_returns'])
            min_episode_return = np.min(rollout_metrics['episode_returns'])
            self.writer.add_scalar('Performance/Episode_Return_Avg', avg_episode_return, update_count)
            self.writer.add_scalar('Performance/Episode_Return_Max', max_episode_return, update_count)
            self.writer.add_scalar('Performance/Episode_Return_Min', min_episode_return, update_count)
            self.writer.add_scalar('Performance/Episodes_Completed', len(rollout_metrics['episode_returns']), update_count)
            
            if 'episode_lengths' in rollout_metrics:
                avg_episode_length = np.mean(rollout_metrics['episode_lengths'])
                self.writer.add_scalar('Performance/Episode_Length_Avg', avg_episode_length, update_count)

        # Log environment metrics collected during rollouts
        for key, values in rollout_metrics.items():
            if key.startswith('env_') and values:
                avg_value = np.mean(values)
                self.writer.add_scalar(f'Environment/{key}', avg_value, update_count)
        
        # Store key metrics for tracking
        self.training_metrics['update_count'].append(update_count)
        self.training_metrics['timesteps'].append(timesteps_collected)
        self.training_metrics['reward'].append(avg_reward)
        self.training_metrics['value'].append(avg_value)
        self.training_metrics['entropy'].append(avg_entropy)
        self.training_metrics['total_loss'].append(avg_total_loss)
        
        # Log learning rate if available
        if 'learning_rate' in update_metrics:
            current_lr = update_metrics['learning_rate'][0]
            self.writer.add_scalar('Training/Learning_Rate', current_lr, update_count)
        
        # Log exploration noise if available
        if 'exploration_noise' in update_metrics:
            current_noise = update_metrics['exploration_noise'][0]
            self.writer.add_scalar('Training/Exploration_Noise', current_noise, update_count)

    def log_env_metrics(self, env_metrics: Dict, update_count: int, prefix: str = 'Environment'):
        """Log environment-specific metrics"""
        if env_metrics:
            if isinstance(env_metrics, dict):
                for key, value in env_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'{prefix}/{key}', value, update_count)

    def should_run_test(self, update_count: int) -> bool:
        """Check if test should be run at this update"""
        return update_count % self.test_interval == 0

    def log_test_results(self, test_rewards: List[float], env_metrics: Dict, update_count: int):
        """Log test episode results"""
        avg_test_reward = np.mean(test_rewards)
        std_test_reward = np.std(test_rewards)
        min_test_reward = np.min(test_rewards)
        max_test_reward = np.max(test_rewards)

        print(f"[TEST] Update {update_count:4d} | "
              f"Avg: {avg_test_reward:.3f} ± {std_test_reward:.3f} | "
              f"Range: [{min_test_reward:.3f}, {max_test_reward:.3f}]")

        self.writer.add_scalar('Test/Avg_Reward', avg_test_reward, update_count)
        self.writer.add_scalar('Test/Std_Reward', std_test_reward, update_count)
        self.writer.add_scalar('Test/Min_Reward', min_test_reward, update_count)
        self.writer.add_scalar('Test/Max_Reward', max_test_reward, update_count)

        for metric_name, values in env_metrics.items():
            if values:
                avg_value = np.mean(values)
                self.writer.add_scalar(f'Test/{metric_name}', avg_value, update_count)

        self.test_metrics['update_count'].append(update_count)
        self.test_metrics['avg_reward'].append(avg_test_reward)
        self.test_metrics['std_reward'].append(std_test_reward)

        return avg_test_reward


class KLAdaptiveLR:
    """
    Adaptive learning rate adjustment based on KL divergence.
    Works on top of existing schedulers by applying a multiplicative factor.
    """
    
    def __init__(
        self, 
        optimizer: Optimizer,
        kl_target: float = 0.02,
        initial_adjustment: float = 1.0,
        min_adjustment: float = 0.1,
        max_adjustment: float = 1.5  # Reduced from 2.0 to prevent instability
    ):
        """
        Initialize KL-adaptive learning rate manager.
        
        Args:
            optimizer: PyTorch optimizer
            kl_target: Target KL divergence value
            initial_adjustment: Initial multiplicative factor
            min_adjustment: Minimum allowed adjustment factor
            max_adjustment: Maximum allowed adjustment factor
        """
        self.optimizer = optimizer
        self.kl_target = kl_target
        self.adjustment_factor = initial_adjustment
        self.min_adjustment = min_adjustment
        self.max_adjustment = max_adjustment
        
        # Store initial learning rates for reference
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def update(self, kl_divergence: float, verbose: bool = True) -> float:
        """
        Update learning rate based on KL divergence.
        
        Args:
            kl_divergence: Current KL divergence value
            verbose: Whether to print adjustment messages
            
        Returns:
            Current adjustment factor
        """
        old_factor = self.adjustment_factor
        
        # Adjust factor based on KL divergence
        if kl_divergence > self.kl_target * 2:  # KL much too high
            self.adjustment_factor *= 0.5
            if verbose:
                print(f"  [KL Adaptive] High KL ({kl_divergence:.4f}), reducing LR adjustment to {self.adjustment_factor:.3f}")
                
        elif kl_divergence > self.kl_target * 1.5:  # KL somewhat high
            self.adjustment_factor *= 0.9
            
        elif kl_divergence < self.kl_target * 0.5:  # KL too low
            self.adjustment_factor *= 1.1
            if verbose and self.adjustment_factor != old_factor:
                print(f"  [KL Adaptive] Low KL ({kl_divergence:.4f}), increasing LR adjustment to {self.adjustment_factor:.3f}")
        
        # Clamp adjustment factor
        self.adjustment_factor = max(self.min_adjustment, 
                                    min(self.max_adjustment, self.adjustment_factor))
        
        return self.adjustment_factor
    
    def apply_adjustment(self, scheduled_lr: float = None) -> float:
        """
        Apply KL adjustment to learning rate.
        
        Args:
            scheduled_lr: Learning rate from scheduler (if any)
            
        Returns:
            Adjusted learning rate
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            if scheduled_lr is not None:
                # Apply adjustment on top of scheduled LR
                adjusted_lr = scheduled_lr * self.adjustment_factor
            else:
                # Apply adjustment to current LR
                adjusted_lr = param_group['lr'] * self.adjustment_factor
            
            param_group['lr'] = adjusted_lr
            
        return self.optimizer.param_groups[0]['lr']
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for logging."""
        return {
            'adjustment_factor': self.adjustment_factor,
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
    
    def reset(self):
        """Reset adjustment factor to initial value."""
        self.adjustment_factor = 1.0


class LRSchedulerManager:
    """
    Unified learning rate scheduler manager that handles:
    - Traditional schedulers (linear, cosine, constant)
    - KL-divergence adaptive adjustment
    - Universal warmup for all schedulers
    - Hybrid combinations of schedulers + KL adaptive
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        schedule_type: str = "constant",
        total_updates: int = 1000,
        warmup_steps: int = 0,
        use_kl_adaptive: bool = False,
        kl_target: float = 0.02,
        combine_kl_with_scheduler: bool = False,
        lr_min_factor: float = 0.01
    ):
        """
        Initialize the LR scheduler manager.

        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            schedule_type: Type of scheduler ("constant", "linear", "cosine")
            total_updates: Total number of training updates
            warmup_steps: Number of warmup steps (applies to all schedulers)
            use_kl_adaptive: Whether to use KL-divergence based adjustment
            kl_target: Target KL divergence for adaptive adjustment
            combine_kl_with_scheduler: Whether to combine KL with traditional scheduler
            lr_min_factor: Minimum LR as fraction of base_lr (for cosine/linear)
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.schedule_type = schedule_type
        self.total_updates = total_updates
        self.warmup_steps = warmup_steps
        self.use_kl_adaptive = use_kl_adaptive
        self.combine_kl_with_scheduler = combine_kl_with_scheduler
        self.lr_min_factor = lr_min_factor
        
        # Initialize KL adaptive if needed
        self.kl_adaptive = None
        if use_kl_adaptive:
            from torch.optim.lr_scheduler import LinearLR
            self.kl_adaptive = KLAdaptiveLR(optimizer, kl_target=kl_target)
        
        # Create scheduler based on configuration
        self.scheduler = self._create_scheduler()
        self.current_step = 0
        self.warmup_complete = False
        
    def _create_scheduler(self):
        """Create the appropriate scheduler based on configuration."""
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        # If using pure KL-adaptive (no combination), only create warmup if needed
        if self.use_kl_adaptive and not self.combine_kl_with_scheduler:
            if self.warmup_steps > 0:
                # Just warmup for pure KL-adaptive
                return LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                              total_iters=self.warmup_steps)
            else:
                return None

        # Create main scheduler
        main_scheduler = None
        effective_updates = max(1, self.total_updates - self.warmup_steps)

        if self.schedule_type == "constant":
            # Constant doesn't need a scheduler (unless warmup is used)
            main_scheduler = None
        elif self.schedule_type == "linear":
            main_scheduler = LinearLR(self.optimizer, start_factor=1.0,
                                     end_factor=self.lr_min_factor,
                                     total_iters=effective_updates)
        elif self.schedule_type == "cosine":
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=effective_updates,
                                              eta_min=self.base_lr * self.lr_min_factor)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        # Add warmup if specified
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                                       total_iters=self.warmup_steps)
            if main_scheduler is not None:
                # Combine warmup with main scheduler
                return SequentialLR(self.optimizer, [warmup_scheduler, main_scheduler],
                                  milestones=[self.warmup_steps])
            else:
                # Just warmup (for constant schedule)
                return warmup_scheduler
        
        return main_scheduler
    
    def step(self, kl_divergence: Optional[float] = None, verbose: bool = False) -> float:
        """
        Step the learning rate scheduler.
        
        Args:
            kl_divergence: Current KL divergence (required if using KL adaptive)
            verbose: Whether to print adjustment messages
            
        Returns:
            Current learning rate
        """
        self.current_step += 1
        
        # Check if warmup is complete
        if not self.warmup_complete and self.current_step >= self.warmup_steps:
            self.warmup_complete = True
            if verbose and self.warmup_steps > 0:
                print(f"  [LR Scheduler] Warmup complete at step {self.current_step}")
        
        # Handle different scheduling modes
        current_lr = self.base_lr
        
        if self.use_kl_adaptive and not self.combine_kl_with_scheduler:
            # Pure KL-adaptive mode
            if self.scheduler:
                # Apply warmup if still in warmup phase
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Apply KL adjustment after warmup
            if self.warmup_complete and kl_divergence is not None:
                self.kl_adaptive.update(kl_divergence, verbose=verbose)
                current_lr = self.kl_adaptive.apply_adjustment()
                
        elif self.combine_kl_with_scheduler:
            # Combined mode: scheduler + KL adaptive
            if self.scheduler:
                self.scheduler.step()
                scheduled_lr = self.optimizer.param_groups[0]['lr']
            else:
                scheduled_lr = self.base_lr
            
            # Apply KL adjustment on top after warmup
            if self.warmup_complete and kl_divergence is not None and self.kl_adaptive is not None:
                self.kl_adaptive.update(kl_divergence, verbose=verbose)
                current_lr = self.kl_adaptive.apply_adjustment(scheduled_lr)
            else:
                current_lr = scheduled_lr
                
        else:
            # Pure scheduler mode (no KL)
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.base_lr
        
        return current_lr
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_description(self) -> str:
        """Get a human-readable description of the scheduling strategy."""
        parts = []
        
        if self.use_kl_adaptive:
            if self.kl_adaptive:
                parts.append(f"KL-adaptive (target={self.kl_adaptive.kl_target})")
            if self.combine_kl_with_scheduler:
                parts.append(f"{self.schedule_type} scheduler")
        else:
            parts.append(f"{self.schedule_type} scheduler")
        
        strategy = " + ".join(parts) if parts else "constant"
        
        # Add warmup info
        if self.warmup_steps > 0:
            strategy += f" with {self.warmup_steps}-step warmup"
        else:
            strategy += " (no warmup)"
        
        return strategy
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        state = {
            'current_step': self.current_step,
            'warmup_complete': self.warmup_complete
        }
        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()
        if self.kl_adaptive:
            state['kl_adjustment_factor'] = self.kl_adaptive.adjustment_factor
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.current_step = state_dict.get('current_step', 0)
        self.warmup_complete = state_dict.get('warmup_complete', False)
        if self.scheduler and 'scheduler_state' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
        if self.kl_adaptive and 'kl_adjustment_factor' in state_dict:
            self.kl_adaptive.adjustment_factor = state_dict['kl_adjustment_factor']


class SequentialBucketBaseline:
    """Baseline policy that selects first valid bucket sequentially.

    Simple baseline for job ordering environment with discrete actions.
    Uses num_valid_buckets from state to determine which buckets exist.
    """

    def __init__(self, num_buckets):
        self.num_buckets = num_buckets

    def get_action_and_value(self, obs, deterministic=True):
        """Select first valid bucket (sequential bucket selection).

        Args:
            obs: Observation tensor with num_valid_buckets at the end
            deterministic: Ignored (baseline is always deterministic)

        Returns:
            action: Bucket index of first valid bucket (0 to num_valid-1)
            log_prob: None (not applicable for baseline)
            entropy: None (not applicable for baseline)
            value: None (not applicable for baseline)
        """
        # State format: [bucket0_cores, bucket0_mem, bucket0_count, ..., global(5), num_valid_buckets(1)]
        # Extract num_valid_buckets and create mask
        num_valid_idx = obs.shape[-1] - 1

        if obs.dim() == 1:
            # Single observation
            num_valid = int(obs[num_valid_idx].item())
            # Create mask: indices >= num_valid are invalid
            mask = torch.arange(self.num_buckets, device=obs.device) >= num_valid

            # Find first valid bucket (mask == False)
            for i in range(self.num_buckets):
                if not mask[i]:
                    return torch.tensor(i), None, None, None
            # Fallback: if all invalid, return 0
            return torch.tensor(0), None, None, None
        else:
            # Batch of observations
            batch_size = obs.shape[0]
            actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
            num_valid = obs[:, num_valid_idx].long()  # (batch_size,)

            # Create mask for batch: indices >= num_valid are invalid
            indices = torch.arange(self.num_buckets, device=obs.device)
            mask = indices.unsqueeze(0) >= num_valid.unsqueeze(1)  # (batch_size, num_buckets)

            for b in range(batch_size):
                # Find first valid bucket
                for i in range(self.num_buckets):
                    if not mask[b, i]:
                        actions[b] = i
                        break
                # If all invalid, defaults to 0

            return actions, None, None, None