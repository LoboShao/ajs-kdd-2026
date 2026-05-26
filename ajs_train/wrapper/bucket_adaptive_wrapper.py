import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lsf_env_rust import BucketAdaptiveEnv
from typing import List, Any, Tuple


class BucketAdaptiveWrapper(gym.Env):
    """Gymnasium wrapper for the adaptive macro bucket selection environment.

    Provides a standard Gymnasium interface to the high-performance Rust-based
    cluster simulation for training RL agents on job bucket prioritization.

    Adaptive Macro Action Space:
    - Action is a 2D tuple: (bucket_index, macro_level)
    - bucket_index: Which bucket to schedule from (0 to max_buckets-1)
    - macro_level: Index into configurable macro_levels (e.g., [0.2, 0.5, 1.0])
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Store original constructor kwargs for test environment creation
        self._constructor_kwargs = kwargs.copy()

        # Create the Rust environment
        self.rust_env = BucketAdaptiveEnv(**kwargs)

        # Store environment parameters
        self.num_hosts = kwargs.get('num_hosts', 1000)
        self.max_buckets = kwargs.get('max_buckets', 100)

        # Get number of macro levels from the Rust environment
        self.num_macro_levels = self.rust_env.get_num_macro_levels()
        self.macro_levels = self.rust_env.get_macro_levels()

        # Define action space: MultiDiscrete [bucket_idx, macro_level]
        # bucket_idx: 0 to max_buckets-1
        # macro_level: 0 to num_macro_levels-1
        self.action_space = spaces.MultiDiscrete([self.max_buckets, self.num_macro_levels])

        # Define observation space:
        # - max_buckets * 4 features: (cores, memory, count, waiting_time) per bucket
        # - 2 global features: (available_cores_ratio, available_memory_ratio)
        # - 1 feature: num_valid_buckets (how many buckets currently exist)
        state_size = self.max_buckets * 4 + 2 + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,  # Job count can be > 1.0
            shape=(state_size,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.rust_env.set_random_seed(seed)

        obs = self.rust_env.reset()
        info = {}
        return obs, info

    def step(self, action):
        # action should be a 2-element array: [bucket_idx, macro_level]
        obs, reward, done, info = self.rust_env.step(action)
        terminated = done
        truncated = False

        # info already contains bucket metrics from base_env.get_step_info()
        # which includes: num_buckets, max_buckets_in_episode, macro_jobs_scheduled, macro_level, etc.

        # Auto-reset when episode is done (gymnasium standard behavior)
        if terminated:
            # Store final metrics before reset (important for testing)
            if hasattr(self.rust_env, 'get_metrics'):
                info['final_metrics'] = self.rust_env.get_metrics()
            obs, _ = self.reset()

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Optional: implement visualization
        pass

    def close(self):
        # Cleanup if needed
        pass

    def get_metrics(self):
        return self.rust_env.get_metrics()

    def get_cluster_info(self):
        """Get cluster resource information."""
        return self.rust_env.get_cluster_info()

    def set_random_seed(self, seed=None):
        """Set the random seed for the underlying Rust environment.

        Args:
            seed: Random seed (int) for deterministic behavior, or None for random seeding
        """
        self.rust_env.set_random_seed(seed)

    def set_debug_fcfs(self, enable: bool, log_file: str = None):
        """Enable/disable FCFS debug logging.

        Args:
            enable: True to enable debug logging, False to disable
            log_file: Path to log file (if None, no logging to file)
        """
        self.rust_env.set_debug_fcfs(enable, log_file)

    def create_test_env(self, seed: int):
        """Create a fresh test environment with the same configuration but different seed.

        Args:
            seed: Seed for deterministic testing

        Returns:
            New BucketAdaptiveWrapper instance with same config but specified seed
        """
        # Use the original constructor kwargs with updated seed
        kwargs = self._constructor_kwargs.copy()
        kwargs['seed'] = seed  # Override seed for deterministic testing

        return BucketAdaptiveWrapper(**kwargs)


class VectorizedBucketAdaptiveEnv:
    """Simple vectorized environment for parallel LSF training with BucketAdaptiveEnv."""

    def __init__(self, num_envs: int, **kwargs):
        self.num_envs = num_envs
        self.envs = [BucketAdaptiveWrapper(**kwargs) for _ in range(num_envs)]

        # Get observation and action spaces from first environment
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    def reset(self, seed=None, options=None):
        """Reset all environments."""
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = [None] * self.num_envs

        observations = []
        infos = []

        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seeds[i], options=options)
            observations.append(obs)
            infos.append(info)

        return np.array(observations), infos

    def step(self, actions):
        """Step all environments.

        Args:
            actions: Array of shape (num_envs, 2) where each row is [bucket_idx, macro_level]
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)  # Info dict now includes bucket and macro metrics from wrapper

        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def get_metrics(self):
        """Get aggregated metrics from all environments."""
        # Return metrics from the first environment as a representative sample
        if self.envs:
            return self.envs[0].get_metrics()
        return {}

    def set_random_seed(self, seed=None):
        """Set random seeds for all environments.

        Args:
            seed: Base random seed (int) for deterministic behavior, or None for random seeding.
                  If int, each env gets seed+i. If None, all envs use random seeding.
        """
        for i, env in enumerate(self.envs):
            if seed is not None:
                env.set_random_seed(seed + i)
            else:
                env.set_random_seed(None)


def make_bucket_adaptive_env(num_envs: int = 1, **kwargs):
    """
    Create a bucket adaptive scheduler environment.

    Args:
        num_envs: Number of parallel environments (default: 1)
        **kwargs: Arguments passed to BucketAdaptiveEnv

    Returns:
        gym.Env: The environment instance (vectorized if num_envs > 1)
    """
    if num_envs == 1:
        return BucketAdaptiveWrapper(**kwargs)
    else:
        return VectorizedBucketAdaptiveEnv(num_envs, **kwargs)


if __name__ == "__main__":
    print("Testing direct Rust env call:")
    rust_env = BucketAdaptiveEnv(num_hosts=10, max_buckets=20, max_time=10, seed=42)
    obs = rust_env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs (first 20 values): {obs[:20]}")

    max_buckets = rust_env.get_max_buckets()
    num_macro_levels = rust_env.get_num_macro_levels()
    macro_levels = rust_env.get_macro_levels()
    print(f"Max buckets: {max_buckets}")
    print(f"Macro levels ({num_macro_levels}): {macro_levels}")

    for i in range(5):
        bucket_idx = np.random.randint(0, max_buckets)
        macro_level = np.random.randint(0, num_macro_levels)
        action = np.array([bucket_idx, macro_level], dtype=np.int64)
        obs, reward, done, info = rust_env.step(action)
        print(f"[RustEnv] Step {i+1}: action={action}, reward={reward}, done={done}, macro_level={info.get('macro_level', 'N/A')}")
        if done:
            break

    print("\nTesting Gym wrapper:")
    env = BucketAdaptiveWrapper(num_hosts=10, max_buckets=20, max_time=10, seed=42)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs (first 20 values): {obs[:20]}")
    print(f"Action space: {env.action_space}")
    print(f"Macro levels: {env.macro_levels}")

    for i in range(5):
        action = env.action_space.sample()  # Returns [bucket_idx, macro_level]
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[Wrapper] Step {i+1}: action={action}, reward={reward}, terminated={terminated}, "
              f"macro_jobs={info.get('macro_jobs_scheduled', 0)}, macro_level={info.get('macro_level', 'N/A')}")
        if terminated or truncated:
            break
