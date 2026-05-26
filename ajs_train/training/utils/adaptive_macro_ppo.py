import torch
import torch.optim as optim
import numpy as np
from collections import defaultdict
import time
from typing import Dict, List, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
import json
import os

# Import utility classes (reuse from existing utils)
from .utils import MetricsReporter, LRSchedulerManager


class AdaptiveMacroPPOBuffer:
    """PPO rollout buffer for adaptive macro actions (bucket + macro level)."""

    def __init__(self, buffer_size: int, obs_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device
        
        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.bucket_actions = np.zeros(buffer_size, dtype=np.int64)  # Bucket selection
        self.macro_actions = np.zeros(buffer_size, dtype=np.int64)   # Macro level (0,1,2)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False

    def store(self, obs, bucket_action, macro_action, reward, value, log_prob, done):
        """Store transition in buffer."""
        self.observations[self.ptr] = obs
        self.bucket_actions[self.ptr] = bucket_action
        self.macro_actions[self.ptr] = macro_action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        """Compute GAE advantages."""
        size = self.buffer_size if self.full else self.ptr
        
        advantages = np.zeros(size, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
        self.advantages[:size] = advantages
        self.returns[:size] = advantages + self.values[:size]

    def get(self):
        """Get all data from buffer."""
        size = self.buffer_size if self.full else self.ptr
        return {
            'observations': torch.from_numpy(self.observations[:size]).to(self.device),
            'bucket_actions': torch.from_numpy(self.bucket_actions[:size]).to(self.device),
            'macro_actions': torch.from_numpy(self.macro_actions[:size]).to(self.device),
            'old_log_probs': torch.from_numpy(self.log_probs[:size]).to(self.device),
            'advantages': torch.from_numpy(self.advantages[:size]).to(self.device),
            'returns': torch.from_numpy(self.returns[:size]).to(self.device),
            'old_values': torch.from_numpy(self.values[:size]).to(self.device),
        }

    def reset(self):
        """Reset buffer pointer."""
        self.ptr = 0
        self.full = False


class AdaptiveMacroPPOTrainer:
    """PPO Trainer for adaptive macro action space (bucket + macro level)."""

    @torch.inference_mode()
    def test_with_metrics(self, num_episodes: int = 1, update_count: int = 0,
                         test_seeds: List[int] = None, policy_name: str = "PPO") -> Dict:
        """Test hierarchical policy and collect environment metrics."""
        if test_seeds is None:
            test_seeds = [42]

        if self.first_test_run:
            self.save_test_env_data(test_seeds)
            self.first_test_run = False

        self.policy.eval()
        episode_metrics = []
        seed_to_metrics = {}

        # Track macro levels across all test episodes
        all_macro_levels = []

        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env

        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            test_env = train_env.create_test_env(seed)

            # Get and print cluster information (only on first test)
            if hasattr(test_env, 'get_cluster_info') and update_count == 0:
                cluster_info = test_env.get_cluster_info()
                print(f"\n[TEST {policy_name}] Cluster Info for seed {seed}:")
                print(f"  Total Cluster Cores: {cluster_info.get('total_cluster_cores', 'N/A')}")
                print(f"  Total Cluster Memory: {cluster_info.get('total_cluster_memory', 'N/A')} MB")
                print(f"  Number of Hosts: {cluster_info.get('num_hosts', 'N/A')}")
                print(f"  Host Cores Range: {cluster_info.get('host_cores_range', 'N/A')}")
                print(f"  Host Memory Range: {cluster_info.get('host_memory_range', 'N/A')} MB")

            obs, _ = test_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            final_info = None
            max_steps = 10000

            while not (terminated or truncated) and episode_length < max_steps:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                (bucket_action, macro_action), _, _, _ = self.policy.get_action_and_value(obs_tensor, deterministic=True)

                # Track macro level
                all_macro_levels.append(macro_action.item())

                # Pack adaptive macro action for environment
                action_value = np.array([bucket_action.item(), macro_action.item()], dtype=np.int64)
                obs, reward, terminated, truncated, info = test_env.step(action_value)
                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    final_info = info

            if episode_length >= max_steps:
                print(f"WARNING: {policy_name} episode hit max_steps limit ({max_steps}) for seed {seed}")

            metrics = None
            if final_info and 'final_metrics' in final_info:
                metrics = final_info['final_metrics']
            elif hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()

            if isinstance(metrics, dict):
                metrics['episode_return'] = episode_return
                metrics['episode_length'] = episode_length
                episode_metrics.append(metrics)
                seed_to_metrics[seed] = metrics.copy()

            test_env.close()

        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m and m.get(key) is not None]
                if values:
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}

        # Calculate and log macro level distribution for test episodes
        if all_macro_levels:
            total_actions = len(all_macro_levels)
            level_0_count = all_macro_levels.count(0)
            level_1_count = all_macro_levels.count(1)
            level_2_count = all_macro_levels.count(2)

            level_0_pct = level_0_count / total_actions * 100
            level_1_pct = level_1_count / total_actions * 100
            level_2_pct = level_2_count / total_actions * 100

            # Log to TensorBoard - Test macro action distribution
            self.writer.add_scalar('MacroAction_Test/Level_0_Pct', level_0_pct, update_count)
            self.writer.add_scalar('MacroAction_Test/Level_1_Pct', level_1_pct, update_count)
            self.writer.add_scalar('MacroAction_Test/Level_2_Pct', level_2_pct, update_count)

            # Add to avg_metrics for reporting
            avg_metrics['macro_level_0_pct'] = level_0_pct
            avg_metrics['macro_level_1_pct'] = level_1_pct
            avg_metrics['macro_level_2_pct'] = level_2_pct

        result = {
            'average': avg_metrics,
            'per_seed': seed_to_metrics
        }

        self.policy.train()
        return result

    @torch.inference_mode()
    def test_baseline_with_metrics(self, num_episodes: int = 1, update_count: int = 0,
                                  test_seeds: List[int] = None, debug_fcfs: bool = False) -> Dict:
        """Test baseline policy - assumes baseline also returns adaptive macro actions."""
        if test_seeds is None:
            test_seeds = [42]

        if self.baseline_policy is None:
            return {'average': {}, 'per_seed': {}}

        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env

        baseline = self.baseline_policy
        episode_metrics = []
        seed_to_metrics = {}

        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            test_env = train_env.create_test_env(seed)

            # Enable debug logging for first episode only (to avoid too much output)
            if debug_fcfs and ep == 0 and hasattr(test_env, 'set_debug_fcfs'):
                log_file = "fcfs_debug.log"
                test_env.set_debug_fcfs(True, log_file)
                print(f"\n[DEBUG] FCFS debug logging enabled for seed {seed}, writing to {log_file}")

            obs, _ = test_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            final_info = None
            max_steps = 30000

            while not (terminated or truncated) and episode_length < max_steps:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                (bucket_action, macro_action), _, _, _ = baseline.get_action_and_value(obs_tensor)

                # Pack adaptive macro action
                action_value = np.array([bucket_action.item(), macro_action.item()], dtype=np.int64)
                obs, reward, terminated, truncated, info = test_env.step(action_value)
                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    final_info = info

            if episode_length >= max_steps:
                print(f"WARNING: Baseline episode hit max_steps limit ({max_steps}) for seed {seed}")

            metrics = None
            if final_info and 'final_metrics' in final_info:
                metrics = final_info['final_metrics']
            elif hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()

            if isinstance(metrics, dict):
                metrics['episode_return'] = episode_return
                metrics['episode_length'] = episode_length
                episode_metrics.append(metrics)
                seed_to_metrics[seed] = metrics.copy()

            test_env.close()

        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m and m.get(key) is not None]
                if values:
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}

        # Note: Baseline macro action logging removed - baseline only uses level 2 (100%)
        # which is not meaningful to track

        return {
            'average': avg_metrics,
            'per_seed': seed_to_metrics
        }

    def log_metric_comparison(self, ppo_metrics: Dict, baseline_metrics: Dict, update_count: int):
        """Log comparison between PPO and baseline performance metrics"""

        # Extract average and per-seed metrics
        ppo_avg = ppo_metrics.get('average', ppo_metrics)  # Backward compatibility
        baseline_avg = baseline_metrics.get('average', baseline_metrics)
        ppo_per_seed = ppo_metrics.get('per_seed', {})
        baseline_per_seed = baseline_metrics.get('per_seed', {})

        # Log per-seed metrics
        if ppo_per_seed and baseline_per_seed:
            seeds = list(ppo_per_seed.keys())
            print(f"\n[Test Results - Update {update_count}]")
            print("=" * 60)

            # Log per-seed average waiting time and jobs completed
            for seed in seeds:
                ppo_seed_metrics = ppo_per_seed.get(seed, {})
                baseline_seed_metrics = baseline_per_seed.get(seed, {})

                # Average waiting time per seed
                ppo_wait = ppo_seed_metrics.get('avg_waiting_time', 0)
                baseline_wait = baseline_seed_metrics.get('avg_waiting_time', 0)
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Avg_Waiting_Time', ppo_wait, update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Avg_Waiting_Time', baseline_wait, update_count)

                # Log waiting time comparison
                print(f"  Seed {seed} Avg Waiting Time: PPO={ppo_wait:.1f}, Baseline={baseline_wait:.1f}")

                # Jobs completed per seed
                ppo_jobs = ppo_seed_metrics.get('total_jobs_completed', 0)
                baseline_jobs = baseline_seed_metrics.get('total_jobs_completed', 0)
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Jobs_Completed', ppo_jobs, update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Jobs_Completed', baseline_jobs, update_count)

                # Log jobs completed comparison
                print(f"  Seed {seed} Jobs Completed: PPO={ppo_jobs}, Baseline={baseline_jobs}")

                # Log other per-seed metrics
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Episode_Return',
                                          ppo_seed_metrics.get('episode_return', 0), update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Episode_Return',
                                               baseline_seed_metrics.get('episode_return', 0), update_count)

                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Core_Utilization',
                                          ppo_seed_metrics.get('avg_host_core_utilization', 0), update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Core_Utilization',
                                               baseline_seed_metrics.get('avg_host_core_utilization', 0), update_count)

                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Memory_Utilization',
                                          ppo_seed_metrics.get('avg_host_memory_utilization', 0), update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Memory_Utilization',
                                               baseline_seed_metrics.get('avg_host_memory_utilization', 0), update_count)

            print("-" * 60)

        # Log average metrics (existing behavior)
        # Episode return comparison
        ppo_return = ppo_avg.get('episode_return', 0)
        baseline_return = baseline_avg.get('episode_return', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Episode_Return', ppo_return, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Episode_Return', baseline_return, update_count)

        # Average waiting time comparison (IMPORTANT metric)
        ppo_wait = ppo_avg.get('avg_waiting_time', 0)
        baseline_wait = baseline_avg.get('avg_waiting_time', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Avg_Waiting_Time', ppo_wait, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Avg_Waiting_Time', baseline_wait, update_count)

        print(f"  AVERAGE Waiting Time: PPO={ppo_wait:.1f}, Baseline={baseline_wait:.1f}")

        # Jobs completed comparison (IMPORTANT metric)
        ppo_jobs = ppo_avg.get('total_jobs_completed', 0)
        baseline_jobs = baseline_avg.get('total_jobs_completed', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Jobs_Completed', ppo_jobs, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Jobs_Completed', baseline_jobs, update_count)

        print(f"  AVERAGE Jobs Completed: PPO={ppo_jobs}, Baseline={baseline_jobs}")

        # Makespan comparison (if available)
        ppo_makespan = ppo_avg.get('makespan')
        baseline_makespan = baseline_avg.get('makespan')
        if ppo_makespan is not None and baseline_makespan is not None:
            self.ppo_writer.add_scalar('Comparison_Avg/Makespan', ppo_makespan, update_count)
            self.baseline_writer.add_scalar('Comparison_Avg/Makespan', baseline_makespan, update_count)

            print(f"  AVERAGE Makespan: PPO={ppo_makespan:.0f}, Baseline={baseline_makespan:.0f}")

        print("=" * 60)

        # Host core utilization comparison
        ppo_core_util = ppo_avg.get('avg_host_core_utilization', 0)
        baseline_core_util = baseline_avg.get('avg_host_core_utilization', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Host_Core_Utilization', ppo_core_util, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Host_Core_Utilization', baseline_core_util, update_count)

        # Host memory utilization comparison
        ppo_mem_util = ppo_avg.get('avg_host_memory_utilization', 0)
        baseline_mem_util = baseline_avg.get('avg_host_memory_utilization', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Host_Memory_Utilization', ppo_mem_util, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Host_Memory_Utilization', baseline_mem_util, update_count)

        # Average reward per step
        ppo_avg_reward = ppo_avg.get('episode_return', 0) / max(ppo_avg.get('episode_length', 1), 1)
        baseline_avg_reward = baseline_avg.get('episode_return', 0) / max(baseline_avg.get('episode_length', 1), 1)
        self.ppo_writer.add_scalar('Comparison_Avg/Avg_Reward_Per_Step', ppo_avg_reward, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Avg_Reward_Per_Step', baseline_avg_reward, update_count)


    def __init__(
        self,
        policy,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        buffer_size: int = 2048,
        device: str = "auto",
        clip_value_loss: bool = True,
        tensorboard_log_dir: str = "logs/adaptive_macro_ppo",
        lr_schedule: str = "constant",
        lr_warmup_steps: int = 0,
        early_stopping_patience: int = 50,
        early_stopping_threshold: float = 0.01,
        value_norm_decay: float = 0.99,
        checkpoint_dir: str = None,
        save_freq: int = 100,
        test_seeds: List[int] = None,
        use_kl_adaptive_lr: bool = False,
        kl_target: float = 0.02,
        combine_kl_with_scheduler: bool = False,
        baseline_policy = None
    ):
        self.policy = policy
        self.env = env
        self.baseline_policy = baseline_policy
        self.gamma = gamma
        self.lam = lam
        self.test_seeds = test_seeds if test_seeds is not None else [42, 43, 44]
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.clip_value_loss = clip_value_loss
        
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.policy.to(self.device)
        
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.value_norm_decay = value_norm_decay
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # LR scheduler manager
        self.lr_manager = LRSchedulerManager(
            optimizer=self.optimizer,
            base_lr=lr,
            schedule_type=lr_schedule,
            total_updates=1000,
            warmup_steps=lr_warmup_steps,
            use_kl_adaptive=use_kl_adaptive_lr,
            kl_target=kl_target,
            combine_kl_with_scheduler=combine_kl_with_scheduler,
            lr_min_factor=0.01
        )
        
        # Early stopping with EMA
        self.best_test_reward = float('-inf')
        self.best_update = 0
        self.patience_counter = 0
        self.should_stop = False

        # EMA for smoothed early stopping
        self.ema_reward = None  # Will be initialized on first test
        self.best_ema_reward = float('-inf')
        self.ema_alpha = 0.3  # Higher = more weight on recent (0.3 means 30% new, 70% old)
        
        self.value_mean = 0.0
        self.value_var = 1.0
        
        # Checkpointing
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check if vectorized
        self.is_vectorized = hasattr(env, 'num_envs')
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # Buffer(s) - hierarchical buffers
        if self.is_vectorized:
            obs_dim = env.single_observation_space.shape[0]
            self.buffers = [AdaptiveMacroPPOBuffer(buffer_size, obs_dim, self.device)
                           for _ in range(self.num_envs)]
        else:
            obs_dim = env.observation_space.shape[0]
            self.buffer = AdaptiveMacroPPOBuffer(buffer_size, obs_dim, self.device)
        
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.ppo_writer = SummaryWriter(f"{tensorboard_log_dir}/PPO")
        self.baseline_writer = SummaryWriter(f"{tensorboard_log_dir}/Baseline")
        
        self.metrics_reporter = MetricsReporter(self.writer, self.ppo_writer, self.baseline_writer, test_interval=10)
        self.grad_norms = []
        self.lr_history = []
        self.first_test_run = True
    
    def set_test_interval(self, interval: int):
        """Set the interval for running test episodes."""
        self.metrics_reporter.test_interval = interval
    
    def save_test_env_data(self, test_seeds: List[int] = None):
        """Save test environment data."""
        if test_seeds is None:
            test_seeds = [42]
        
        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env
        
        test_data = {"test_environments": {}}
        
        print("Saving test environment data...")
        for seed in test_seeds:
            test_env = train_env.create_test_env(seed)
            obs, _ = test_env.reset()
            
            rust_env = test_env.rust_env
            hosts = rust_env.get_host_configs()
            job_schedule = rust_env.get_job_schedule()
            
            test_data["test_environments"][seed] = {
                "hosts": hosts,
                "job_schedule": job_schedule
            }
            
            test_env.close()
        
        output_file = os.path.join(self.writer.log_dir, "test_env_data.json")
        
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Test environment data saved to: {output_file}")
        return output_file
    
    def get_training_summary(self) -> Dict:
        """Get summary of training metrics."""
        return {
            'training_metrics': dict(self.metrics_reporter.training_metrics),
            'test_metrics': dict(self.metrics_reporter.test_metrics)
        }
    
    def collect_rollouts(self, num_steps: int) -> Dict:
        """Collect rollouts from the environment."""
        if self.is_vectorized:
            return self._collect_rollouts_vectorized(num_steps)
        else:
            return self._collect_rollouts_single(num_steps)
    
    def _collect_rollouts_single(self, num_steps: int) -> Dict:
        """Collect rollouts from single environment with adaptive macro actions."""
        self.env.set_random_seed(None)
        obs, _ = self.env.reset()
        rollout_metrics = defaultdict(list)
        episode_ended = False
        
        current_episode_return = 0.0
        current_episode_length = 0
        
        for step in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                (bucket_action, macro_action), log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)
            
            # Pack adaptive macro action for environment
            action_for_env = np.array([bucket_action.cpu().item(), macro_action.cpu().item()], dtype=np.int64)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_for_env)
            done = terminated or truncated

            # Store in buffer
            self.buffer.store(obs, bucket_action.cpu().item(), macro_action.cpu().item(),
                            reward, value.cpu().item(), log_prob.cpu().item(), done)

            # Metrics
            rollout_metrics['rewards'].append(reward)
            rollout_metrics['values'].append(value.cpu().item())
            rollout_metrics['entropies'].append(entropy.cpu().item())

            if isinstance(info, dict):
                if 'macro_level' in info:
                    rollout_metrics['macro_levels'].append(info['macro_level'])
                if 'macro_jobs_scheduled' in info:
                    rollout_metrics['macro_jobs_scheduled'].append(info['macro_jobs_scheduled'])
                if 'num_buckets' in info:
                    rollout_metrics['num_buckets'].append(info['num_buckets'])
                if 'max_buckets_in_episode' in info:
                    rollout_metrics['max_buckets_in_episode'].append(info['max_buckets_in_episode'])
                # Track macro level distribution
                if 'macro_level_0_count' in info:
                    rollout_metrics['macro_level_0_count'].append(info['macro_level_0_count'])
                if 'macro_level_1_count' in info:
                    rollout_metrics['macro_level_1_count'].append(info['macro_level_1_count'])
                if 'macro_level_2_count' in info:
                    rollout_metrics['macro_level_2_count'].append(info['macro_level_2_count'])
            
            current_episode_return += reward
            current_episode_length += 1
            
            obs = next_obs
            
            if done:
                rollout_metrics['episode_returns'].append(current_episode_return)
                rollout_metrics['episode_lengths'].append(current_episode_length)
                
                if hasattr(self.env, 'get_metrics'):
                    env_metrics = self.env.get_metrics()
                    for key, val in env_metrics.items():
                        if isinstance(val, (int, float)):
                            rollout_metrics[f'env_{key}'].append(val)
                
                current_episode_return = 0.0
                current_episode_length = 0
                
                self.env.set_random_seed(None)
                obs, _ = self.env.reset()
                episode_ended = True
            else:
                episode_ended = False
        
        # Compute advantages
        if episode_ended:
            last_value = 0.0
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_value = self.policy.get_value(obs_tensor).cpu().item()
        
        self.buffer.compute_advantages(last_value, self.gamma, self.lam)
        return rollout_metrics
    
    def _collect_rollouts_vectorized(self, num_steps: int) -> Dict:
        """Collect rollouts from vectorized environments."""
        self.env.set_random_seed(None)
        obs, _ = self.env.reset()
        rollout_metrics = defaultdict(list)
        
        episode_returns = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        
        for step in range(num_steps):
            obs_batch = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                (batch_bucket_actions, batch_macro_actions), batch_log_probs, batch_entropies, batch_values = self.policy.get_action_and_value(obs_batch)
                
                # Pack actions for environment
                actions_np = np.column_stack([
                    batch_bucket_actions.cpu().numpy(),
                    batch_macro_actions.cpu().numpy()
                ])
            
            # Step all environments
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds)
            
            # Store in buffers
            for env_idx in range(self.num_envs):
                self.buffers[env_idx].store(
                    obs[env_idx],
                    batch_bucket_actions[env_idx].cpu().item(),
                    batch_macro_actions[env_idx].cpu().item(),
                    rewards[env_idx],
                    batch_values[env_idx].cpu().item(),
                    batch_log_probs[env_idx].cpu().item(),
                    dones[env_idx]
                )

                rollout_metrics['rewards'].append(rewards[env_idx])
                rollout_metrics['values'].append(batch_values[env_idx].cpu().item())
                rollout_metrics['entropies'].append(batch_entropies[env_idx].cpu().item())
                episode_returns[env_idx] += rewards[env_idx]
                episode_lengths[env_idx] += 1

                # Track bucket statistics from info
                env_info = None
                if isinstance(infos, (list, tuple)) and env_idx < len(infos):
                    env_info = infos[env_idx]
                elif isinstance(infos, dict):
                    # Handle vectorized env format where infos is a dict with arrays
                    env_info = {k: v[env_idx] if hasattr(v, '__getitem__') else v for k, v in infos.items()}

                if isinstance(env_info, dict):
                    if 'num_buckets' in env_info:
                        rollout_metrics['num_buckets'].append(env_info['num_buckets'])
                    if 'max_buckets_in_episode' in env_info:
                        rollout_metrics['max_buckets_in_episode'].append(env_info['max_buckets_in_episode'])
                    if 'macro_level' in env_info:
                        rollout_metrics['macro_levels'].append(env_info['macro_level'])
                    # Track macro level distribution
                    if 'macro_level_0_count' in env_info:
                        rollout_metrics['macro_level_0_count'].append(env_info['macro_level_0_count'])
                    if 'macro_level_1_count' in env_info:
                        rollout_metrics['macro_level_1_count'].append(env_info['macro_level_1_count'])
                    if 'macro_level_2_count' in env_info:
                        rollout_metrics['macro_level_2_count'].append(env_info['macro_level_2_count'])

                if dones[env_idx]:
                    rollout_metrics['episode_returns'].append(episode_returns[env_idx])
                    rollout_metrics['episode_lengths'].append(episode_lengths[env_idx])
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0

            obs = next_obs
        
        # Compute advantages for all buffers
        obs_batch = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            last_values = self.policy.get_value(obs_batch).cpu().numpy()
        
        for env_idx in range(self.num_envs):
            self.buffers[env_idx].compute_advantages(last_values[env_idx], self.gamma, self.lam)
        
        return rollout_metrics
    
    def update_policy(self) -> Dict:
        """Update policy using PPO with adaptive macro actions."""
        # Get data from buffer(s)
        if self.is_vectorized:
            # Concatenate all buffer data
            all_data = [buf.get() for buf in self.buffers]
            data = {key: torch.cat([d[key] for d in all_data]) for key in all_data[0].keys()}
        else:
            data = self.buffer.get()
        
        observations = data['observations']
        bucket_actions = data['bucket_actions']
        macro_actions = data['macro_actions']
        old_log_probs = data['old_log_probs']
        advantages = data['advantages']
        returns = data['returns']
        old_values = data['old_values']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update stats
        batch_size = observations.shape[0]
        indices = np.arange(batch_size)
        
        metrics = defaultdict(list)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_bucket_actions = bucket_actions[batch_indices]
                batch_macro_actions = macro_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass with both actions
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    batch_obs,
                    bucket_action=batch_bucket_actions,
                    macro_action=batch_macro_actions
                )
                
                # Policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.clip_value_loss:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        new_values - batch_old_values, -self.clip_coef, self.clip_coef
                    )
                    value_loss_unclipped = (new_values - batch_returns) ** 2
                    value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Enhanced gradient clipping with monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.grad_norms.append(grad_norm.item())
                self.optimizer.step()
                
                # Metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy_loss'].append(entropy_loss.item())
                metrics['total_loss'].append(loss.item())
                metrics['grad_norm'].append(grad_norm.item())
                
                # KL divergence approximation
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    metrics['approx_kl'].append(approx_kl.item())
        
        return metrics
    
    def train(self, total_timesteps: int, rollout_steps: int,
              log_interval: int = 1, test_interval: int = 10) -> Dict:
        """Main training loop."""
        # Update LR manager with actual total updates
        total_updates = total_timesteps // (rollout_steps * self.num_envs)
        self.lr_manager.total_updates = total_updates
        # Recreate the scheduler with the correct total updates
        self.lr_manager.scheduler = self.lr_manager._create_scheduler()

        # Set test interval if provided
        self.metrics_reporter.test_interval = test_interval

        print(f"Starting PPO training on {self.device}")
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"LR Strategy: {self.lr_manager.get_description()}")
        print(f"Total updates: {total_updates}, Warmup: {self.lr_manager.warmup_steps}, LR: {self.lr:.2e} → {self.lr * self.lr_manager.lr_min_factor:.2e}")
        print(f"TensorBoard logs: {self.writer.log_dir}")
        print(f"Test episodes will run every {self.metrics_reporter.test_interval} updates")

        # Run initial test before training begins
        print("\nRunning initial performance test...")
        ppo_metrics = self.test_with_metrics(num_episodes=len(self.test_seeds), update_count=0,
                                             policy_name="PPO", test_seeds=self.test_seeds)
        # FCFS debug disabled (set debug_fcfs=True to enable)
        baseline_metrics = self.test_baseline_with_metrics(num_episodes=len(self.test_seeds),
                                                          update_count=0, test_seeds=self.test_seeds,
                                                          debug_fcfs=False)

        # Log initial comparison
        self.log_metric_comparison(ppo_metrics, baseline_metrics, 0)
        ppo_makespan = ppo_metrics.get('average', {}).get('makespan', 'N/A')
        baseline_makespan = baseline_metrics.get('average', {}).get('makespan', 'N/A')
        print(f"Initial PPO performance - Episode Return: {ppo_metrics.get('average', {}).get('episode_return', 0):.1f}, Makespan: {ppo_makespan}")
        print(f"Initial Baseline performance - Episode Return: {baseline_metrics.get('average', {}).get('episode_return', 0):.1f}, Makespan: {baseline_makespan}")

        start_time = time.time()
        timesteps_collected = 0
        update_count = 0

        while timesteps_collected < total_timesteps:
            # Collect rollouts
            rollout_metrics = self.collect_rollouts(rollout_steps)
            timesteps_collected += rollout_steps * self.num_envs

            # Update policy
            update_metrics = self.update_policy()
            update_count += 1

            # Decay exploration temperature if available
            if hasattr(self.policy, 'decay_exploration_noise'):
                self.policy.decay_exploration_noise()

            # Step learning rate scheduler with KL divergence if available
            avg_kl = None
            if 'approx_kl' in update_metrics and update_metrics['approx_kl']:
                avg_kl = np.mean(update_metrics['approx_kl'])

            # Step the unified LR manager
            current_lr = self.lr_manager.step(kl_divergence=avg_kl, verbose=(update_count % log_interval == 0))
            self.lr_history.append(current_lr)
            update_metrics['learning_rate'] = [current_lr]

            # Debug: Print LR at key points to verify schedule
            if update_count in [1, 10, 50, 100, 500, 1000, 2000, 3000, 4000]:
                print(f"  [LR Debug] Update {update_count}: LR = {current_lr:.2e}")

            # Clear buffers
            if self.is_vectorized:
                for buf in self.buffers:
                    buf.reset()
            else:
                self.buffer.reset()

            # Check for early stopping
            if self.should_stop:
                print("Early stopping triggered, ending training...")
                break

            # Log metrics
            if update_count % log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = timesteps_collected / elapsed_time

                self.metrics_reporter.log_training_metrics(
                    update_count, rollout_metrics, update_metrics, timesteps_collected, fps
                )

                # Print bucket statistics
                if rollout_metrics.get('num_buckets'):
                    avg_buckets = np.mean(rollout_metrics['num_buckets'])
                    max_buckets = max(rollout_metrics.get('max_buckets_in_episode', [0]))
                    print(f"  Buckets: avg={avg_buckets:.1f}, max={max_buckets}")

                # Log macro level distribution to TensorBoard and terminal
                if rollout_metrics.get('macro_levels'):
                    macro_levels = rollout_metrics['macro_levels']
                    total_actions = len(macro_levels)
                    if total_actions > 0:
                        level_0_pct = macro_levels.count(0) / total_actions * 100
                        level_1_pct = macro_levels.count(1) / total_actions * 100
                        level_2_pct = macro_levels.count(2) / total_actions * 100

                        # Log to TensorBoard - Training macro action distribution
                        self.writer.add_scalar('MacroAction_Train/Level_0_Pct', level_0_pct, update_count)
                        self.writer.add_scalar('MacroAction_Train/Level_1_Pct', level_1_pct, update_count)
                        self.writer.add_scalar('MacroAction_Train/Level_2_Pct', level_2_pct, update_count)

                        print(f"  Macro Levels: L0(20%)={level_0_pct:.1f}%, L1(50%)={level_1_pct:.1f}%, L2(100%)={level_2_pct:.1f}%")

                if self.grad_norms:
                    avg_grad_norm = np.mean(self.grad_norms[-10:])
                    self.writer.add_scalar('Training/Grad_Norm', avg_grad_norm, update_count)

                if self.lr_history:
                    self.writer.add_scalar('Training/Learning_Rate', self.lr_history[-1], update_count)

                # Log exploration temperature if available
                if hasattr(self.policy, 'get_temperature'):
                    self.writer.add_scalar('Training/Temperature', self.policy.get_temperature(), update_count)

            # Save checkpoint at specified intervals
            if self.checkpoint_dir and update_count % self.save_freq == 0:
                self.save_checkpoint(update_count)

            # Run test episodes at specified intervals
            if self.metrics_reporter.should_run_test(update_count):
                ppo_metrics = self.test_with_metrics(num_episodes=len(self.test_seeds),
                                                     update_count=update_count, policy_name="PPO", test_seeds=self.test_seeds)
                baseline_metrics = self.test_baseline_with_metrics(num_episodes=len(self.test_seeds),
                                                                  update_count=update_count, test_seeds=self.test_seeds)

                # Compare performance metrics
                self.log_metric_comparison(ppo_metrics, baseline_metrics, update_count)

                # Check early stopping based on EMA of average reward per step
                # Using avg reward instead of episode return for stability across varying episode lengths
                ppo_avg = ppo_metrics.get('average', ppo_metrics)
                episode_return = ppo_avg.get('episode_return', 0)
                episode_length = max(ppo_avg.get('episode_length', 1), 1)
                current_avg_reward = episode_return / episode_length

                # Update EMA
                if self.ema_reward is None:
                    self.ema_reward = current_avg_reward  # Initialize on first test
                else:
                    self.ema_reward = self.ema_alpha * current_avg_reward + (1 - self.ema_alpha) * self.ema_reward

                # Log EMA to TensorBoard
                self.writer.add_scalar('EarlyStopping/EMA_Avg_Reward', self.ema_reward, update_count)
                self.writer.add_scalar('EarlyStopping/Raw_Avg_Reward', current_avg_reward, update_count)
                self.writer.add_scalar('EarlyStopping/Episode_Return', episode_return, update_count)

                # Track best avg reward for checkpoint saving
                if current_avg_reward > self.best_test_reward:
                    self.best_test_reward = current_avg_reward
                    self.best_update = update_count

                    # Save checkpoint for best model
                    if self.checkpoint_dir:
                        best_path = os.path.join(self.checkpoint_dir, f'best_model_{update_count}.pt')
                        self.save_checkpoint(update_count, checkpoint_path=best_path)
                        latest_best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                        self.save_checkpoint(update_count, checkpoint_path=latest_best_path)
                        print(f"  New best avg reward: {self.best_test_reward:.4f} - saved checkpoint")

                # Check if EMA improved (this controls early stopping)
                if self.ema_reward > self.best_ema_reward:
                    improvement = self.ema_reward - self.best_ema_reward
                    self.best_ema_reward = self.ema_reward
                    self.patience_counter = 0
                    print(f"  EMA improving: {self.ema_reward:.4f} (+{improvement:.4f}) | Raw: {current_avg_reward:.4f}")
                else:
                    self.patience_counter += 1
                    gap = self.best_ema_reward - self.ema_reward
                    print(f"  EMA not improving: {self.ema_reward:.4f} (best: {self.best_ema_reward:.4f}, gap: {gap:.4f})")
                    print(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")

                if self.patience_counter >= self.early_stopping_patience:
                    self.should_stop = True
                    print(f"  Early stopping triggered (EMA stopped improving).")
                    print(f"  Best model was at update {self.best_update} with avg reward {self.best_test_reward:.4f}")

                    # Load the best model before stopping
                    if self.checkpoint_dir:
                        best_path = os.path.join(self.checkpoint_dir, f'best_model_{self.best_update}.pt')
                        if os.path.exists(best_path):
                            self.load_checkpoint(best_path)
                            print(f"  Loaded best model from {best_path}")
                        else:
                            fallback_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                            if os.path.exists(fallback_path):
                                self.load_checkpoint(fallback_path)
                                print(f"  Loaded best model from {fallback_path}")

        # Final training summary
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Total updates: {update_count}")
        print(f"Final FPS: {timesteps_collected / training_time:.0f}")

        # Log final summary
        if self.metrics_reporter.training_metrics['reward']:
            final_avg_reward = self.metrics_reporter.training_metrics['reward'][-1]
            print(f"Final training reward: {final_avg_reward:.3f}")
        if self.metrics_reporter.test_metrics['avg_reward']:
            final_test_reward = self.metrics_reporter.test_metrics['avg_reward'][-1]
            print(f"Final test reward: {final_test_reward:.3f}")

        # Report best model information
        print(f"\n{'='*60}")
        print(f"BEST MODEL SUMMARY:")
        print(f"  Update: {self.best_update}")
        print(f"  Avg Reward/Step: {self.best_test_reward:.4f}")
        if self.checkpoint_dir:
            print(f"  Saved as: best_model_{self.best_update}.pt")
        print(f"{'='*60}")

        self.writer.close()
        self.ppo_writer.close()
        self.baseline_writer.close()

        return self.get_training_summary()
    
    def save_checkpoint(self, update: int, checkpoint_path: str = None):
        """Save training checkpoint."""
        checkpoint = {
            'update': update,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_manager_state_dict': self.lr_manager.state_dict(),
            'best_test_reward': self.best_test_reward,
            'best_update': self.best_update,
            'patience_counter': self.patience_counter,
            'value_mean': self.value_mean,
            'value_var': self.value_var,
            'config': {
                'lr': self.lr,
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_coef': self.clip_coef,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm
            },
            'lr_config': {
                'strategy': self.lr_manager.get_description()
            }
        }

        # Save exploration temperature if available
        if hasattr(self.policy, 'temperature'):
            checkpoint['temperature'] = self.policy.temperature

        # Save EMA state for early stopping
        checkpoint['ema_reward'] = self.ema_reward
        checkpoint['best_ema_reward'] = self.best_ema_reward

        if checkpoint_path is None:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_{update}.pt')
        else:
            path = checkpoint_path

        torch.save(checkpoint, path)

        # Also save as latest if not a custom path
        if checkpoint_path is None:
            latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
            torch.save(checkpoint, latest_path)

        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'lr_manager_state_dict' in checkpoint:
            self.lr_manager.load_state_dict(checkpoint['lr_manager_state_dict'])
        elif 'scheduler_state_dict' in checkpoint:
            # Backward compatibility with old checkpoints
            print("Warning: Loading old checkpoint format, LR scheduler state may not be fully restored")

        self.best_test_reward = checkpoint.get('best_test_reward', float('-inf'))
        self.best_update = checkpoint.get('best_update', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.value_mean = checkpoint.get('value_mean', 0.0)
        self.value_var = checkpoint.get('value_var', 1.0)

        # Restore exploration temperature if available
        if 'temperature' in checkpoint and hasattr(self.policy, 'temperature'):
            self.policy.temperature = checkpoint['temperature']
            print(f"  Restored temperature: {self.policy.temperature:.4f}")

        # Restore EMA state for early stopping
        if 'ema_reward' in checkpoint:
            self.ema_reward = checkpoint['ema_reward']
            self.best_ema_reward = checkpoint.get('best_ema_reward', self.ema_reward)
            print(f"  Restored EMA: {self.ema_reward:.3f} (best: {self.best_ema_reward:.3f})")

        update_count = checkpoint['update']
        print(f"Checkpoint loaded from {path}, resuming from update {update_count}")
        return update_count
