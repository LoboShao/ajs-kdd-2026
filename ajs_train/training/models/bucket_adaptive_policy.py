import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class BucketAdaptivePolicy(nn.Module):
    """Sequential factorization policy for bucket adaptive env: P(bucket) × P(macro | bucket)."""

    def __init__(self, obs_dim, num_buckets, num_macro_levels=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_buckets = num_buckets
        self.num_macro_levels = num_macro_levels
        self.num_valid_idx = obs_dim - 1

        # Model hyperparameters
        self.hidden_size = 64
        self.num_heads = 4
        self.dropout = 0.1

        # Encoders
        # Bucket features: cores, memory, count, waiting_time (4 features)
        self.bucket_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.cluster_feature_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=self.dropout
        )

        # Policy heads
        self.bucket_policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.macro_policy_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 64),
            nn.GELU(),
            nn.Linear(64, num_macro_levels)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Initialize
        self.apply(self._init_weights)
        self._init_policy_heads()

        # Temperature for exploration
        self.temperature = 1.0
        self.min_temperature = 0.3
        self.temperature_decay = 0.9999  # Slower decay: reaches 0.5 around 80% of training (~3200 updates)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_policy_heads(self):
        nn.init.orthogonal_(self.bucket_policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self.bucket_policy_head[-1].bias, 0)
        nn.init.orthogonal_(self.macro_policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self.macro_policy_head[-1].bias, 0)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, 0)

    def encode(self, obs):
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]

        # Parse observation
        # Bucket features: cores, memory, count, waiting_time (4 per bucket)
        bucket_features = obs[:, :self.num_buckets * 4].view(batch_size, self.num_buckets, 4)
        # Global features: available_cores_ratio, available_memory_ratio (2 features)
        global_features = obs[:, self.num_buckets * 4:self.num_buckets * 4 + 2]

        # Encode buckets
        bucket_embeds = self.bucket_encoder(bucket_features)

        # Encode global features (vectorized)
        global_expanded = global_features.unsqueeze(-1)  # (batch, 2, 1)
        cluster_embeds = self.cluster_feature_encoder(global_expanded)  # (batch, 2, hidden)

        # Extract mask
        num_valid = obs[:, self.num_valid_idx].long()
        indices = torch.arange(self.num_buckets, device=obs.device)
        mask = indices.unsqueeze(0) >= num_valid.unsqueeze(1)

        # Cross-attention with residual
        all_masked = mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            attended_buckets = torch.zeros_like(bucket_embeds)
            has_valid = ~all_masked.squeeze(-1)
            if has_valid.any():
                attn_out, _ = self.cross_attention(
                    bucket_embeds[has_valid], cluster_embeds[has_valid], cluster_embeds[has_valid]
                )
                attended_buckets[has_valid] = bucket_embeds[has_valid] + attn_out
        else:
            attn_out, _ = self.cross_attention(bucket_embeds, cluster_embeds, cluster_embeds)
            attended_buckets = bucket_embeds + attn_out

        # Pool valid buckets
        valid_mask = (~mask).unsqueeze(-1).float()
        num_valid_clamped = valid_mask.sum(dim=1).clamp(min=1)
        pooled_buckets = (attended_buckets * valid_mask).sum(dim=1) / num_valid_clamped
        cluster_pooled = cluster_embeds.mean(dim=1)

        return attended_buckets, pooled_buckets, cluster_pooled, mask, single_input

    def forward(self, obs, bucket_action=None, deterministic=False):
        attended_buckets, pooled_buckets, cluster_pooled, mask, single_input = self.encode(obs)
        batch_size = attended_buckets.shape[0]

        # Bucket logits
        bucket_logits = self.bucket_policy_head(attended_buckets).squeeze(-1)
        bucket_logits = bucket_logits.masked_fill(mask, float('-inf'))

        # Sample bucket if not provided
        # NOTE: Temperature is applied here for consistency with log_prob computation
        # in get_action_and_value(). Both sampling and log_prob must use the same
        # temperature to satisfy PPO's importance sampling assumptions.
        if bucket_action is None:
            all_invalid = (bucket_logits == float('-inf')).all(dim=-1)

            if deterministic:
                bucket_action = bucket_logits.argmax(dim=-1)
            else:
                safe_logits = bucket_logits.clone()
                safe_logits[all_invalid] = 0.0
                bucket_action = Categorical(logits=safe_logits / self.temperature).sample()

            if all_invalid.any():
                bucket_action = bucket_action.clone()
                bucket_action[all_invalid] = 0

        if bucket_action.dim() == 0:
            bucket_action = bucket_action.unsqueeze(0)

        # Get selected bucket embedding
        batch_indices = torch.arange(batch_size, device=obs.device)
        selected_bucket_embed = attended_buckets[batch_indices, bucket_action]

        # Macro logits (conditioned on selected bucket)
        combined = torch.cat([pooled_buckets, cluster_pooled, selected_bucket_embed], dim=1)
        macro_logits = self.macro_policy_head(combined)

        # Value
        value = self.value_head(torch.cat([pooled_buckets, cluster_pooled], dim=1)).squeeze(-1)

        if single_input:
            return bucket_logits.squeeze(0), macro_logits.squeeze(0), value.squeeze(0), bucket_action.squeeze(0)
        return bucket_logits, macro_logits, value, bucket_action

    def get_action_and_value(self, obs, bucket_action=None, macro_action=None, deterministic=False):
        bucket_logits, macro_logits, value, sampled_bucket = self.forward(
            obs, bucket_action=bucket_action, deterministic=deterministic
        )

        # Always work in batch mode
        single_input = bucket_logits.dim() == 1
        if single_input:
            bucket_logits = bucket_logits.unsqueeze(0)
            macro_logits = macro_logits.unsqueeze(0)
            value = value.unsqueeze(0)
            sampled_bucket = sampled_bucket.unsqueeze(0)
            if bucket_action is not None:
                bucket_action = bucket_action.unsqueeze(0)
            if macro_action is not None:
                macro_action = macro_action.unsqueeze(0)

        # Handle invalid buckets
        all_invalid = (bucket_logits == float('-inf')).all(dim=-1)
        safe_bucket_logits = bucket_logits.clone()
        safe_bucket_logits[all_invalid] = 0.0

        # Create distributions
        bucket_dist = Categorical(logits=safe_bucket_logits / self.temperature)
        macro_dist = Categorical(logits=macro_logits / self.temperature)

        # Get actions
        if bucket_action is None:
            bucket_action = sampled_bucket
        if macro_action is None:
            macro_action = macro_logits.argmax(dim=-1) if deterministic else macro_dist.sample()

        # Compute log prob and entropy, normalized by max entropy to balance gradients
        # This prevents bucket selection (up to 100 choices) from dominating macro (num_macro_levels choices)
        valid_bucket_counts = (bucket_logits != float('-inf')).sum(dim=-1).float().clamp(min=1)
        bucket_max_entropy = torch.log(valid_bucket_counts).clamp(min=0.1)
        macro_max_entropy = np.log(self.num_macro_levels) if self.num_macro_levels > 1 else 1.0

        # Normalize log_probs to range [-1, 0] for both components
        bucket_log_prob = bucket_dist.log_prob(bucket_action) / bucket_max_entropy
        macro_log_prob = macro_dist.log_prob(macro_action) / macro_max_entropy
        log_prob = bucket_log_prob + macro_log_prob

        # Normalize entropy for balanced exploration
        bucket_entropy = bucket_dist.entropy() / bucket_max_entropy
        macro_entropy = macro_dist.entropy() / macro_max_entropy
        entropy = bucket_entropy + macro_entropy

        # Zero out invalid rows
        if all_invalid.any():
            log_prob = log_prob.clone()
            entropy = entropy.clone()
            log_prob[all_invalid] = 0.0
            entropy[all_invalid] = 0.0

        # Squeeze back if single input
        if single_input:
            return (bucket_action.squeeze(0), macro_action.squeeze(0)), log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(0)
        return (bucket_action, macro_action), log_prob, entropy, value

    def get_value(self, obs):
        _, pooled_buckets, cluster_pooled, _, single_input = self.encode(obs)
        value = self.value_head(torch.cat([pooled_buckets, cluster_pooled], dim=1)).squeeze(-1)
        return value.squeeze(0) if single_input else value

    def decay_exploration_noise(self, decay_rate=None, min_temp=None):
        """Decay temperature for exploration.

        IMPORTANT: Call this only ONCE per PPO update cycle (after all epochs/minibatches),
        NOT during rollout collection. The temperature used during rollout collection
        must match the temperature used during the subsequent PPO update to maintain
        correct importance sampling ratios.
        """
        decay_rate = decay_rate or self.temperature_decay
        min_temp = min_temp or self.min_temperature
        self.temperature = max(min_temp, self.temperature * decay_rate)

    def set_temperature(self, temperature):
        self.temperature = max(self.min_temperature, temperature)

    def get_temperature(self):
        return self.temperature
