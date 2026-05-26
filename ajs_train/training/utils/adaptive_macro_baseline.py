import torch
import numpy as np


class AdaptiveMacroSequentialBaseline:
    """FCFS baseline for adaptive macro actions: always select first valid bucket with single-job dispatch.

    This baseline always:
    1. Selects the first valid bucket (bucket 0) - buckets are sorted by first job's arrival time
    2. Uses a special macro_level (= num_macro_levels) that dispatches exactly 1 job

    This implements true FCFS: always dispatch the job with earliest arrival time.
    """

    def __init__(self, max_buckets: int, num_macro_levels: int = 3):
        self.max_buckets = max_buckets
        self.num_macro_levels = num_macro_levels
        # Special level beyond configured levels = single job dispatch (FCFS)
        self.single_job_level = num_macro_levels
        self.num_valid_idx = None  # Will be set from observation

    def get_action_and_value(self, obs):
        """Get baseline action: always first valid bucket with single-job dispatch.

        Returns:
            (bucket_action, macro_action): Both as torch.tensor (bucket=0, macro=single_job_level)
            log_prob: Dummy value (0.0)
            entropy: Dummy value (0.0)
            value: Dummy value (0.0)
        """
        # Handle batch of observations
        if obs.dim() == 2:
            batch_size = obs.shape[0]
            # Process each observation in batch
            bucket_actions = []
            macro_actions = []

            for i in range(batch_size):
                single_obs = obs[i]
                bucket, macro = self._get_single_action(single_obs)
                bucket_actions.append(bucket)
                macro_actions.append(macro)

            bucket_action = torch.tensor(bucket_actions, dtype=torch.long, device=obs.device)
            macro_action = torch.tensor(macro_actions, dtype=torch.long, device=obs.device)
            log_prob = torch.zeros(batch_size, device=obs.device)
            entropy = torch.zeros(batch_size, device=obs.device)
            value = torch.zeros(batch_size, device=obs.device)
        else:
            # Single observation
            bucket, macro = self._get_single_action(obs)
            bucket_action = torch.tensor(bucket, dtype=torch.long, device=obs.device)
            macro_action = torch.tensor(macro, dtype=torch.long, device=obs.device)
            log_prob = torch.tensor(0.0, device=obs.device)
            entropy = torch.tensor(0.0, device=obs.device)
            value = torch.tensor(0.0, device=obs.device)

        return (bucket_action, macro_action), log_prob, entropy, value

    def _get_single_action(self, obs):
        """Get action for a single observation - always first valid bucket with single-job dispatch."""
        # Last element is num_valid_buckets
        if self.num_valid_idx is None:
            self.num_valid_idx = obs.shape[0] - 1

        num_valid = int(obs[self.num_valid_idx].item())

        # If no valid buckets, return dummy action
        if num_valid == 0:
            return 0, self.single_job_level  # Bucket 0, single job (will be masked anyway)

        # Always select first valid bucket (bucket 0) with single-job dispatch
        # Buckets are sorted by first job's arrival time, so this is true FCFS
        bucket_idx = 0
        macro_level = self.single_job_level  # Special level = dispatch exactly 1 job

        return bucket_idx, macro_level

    def get_value(self, obs):
        """Get dummy value estimate."""
        if obs.dim() == 2:
            return torch.zeros(obs.shape[0], device=obs.device)
        else:
            return torch.tensor(0.0, device=obs.device)

    def eval(self):
        """Dummy method for compatibility."""
        pass

    def train(self):
        """Dummy method for compatibility."""
        pass

    def to(self, device):
        """Dummy method for compatibility."""
        return self
