#!/usr/bin/env python3
"""
Train PPO agent on LSF bucket adaptive environment.

Adaptive Macro actions: Each action selects a bucket AND a dispatch level,
allowing the agent to control how aggressively to schedule from each bucket.

Action space: (bucket_index, dispatch_level)
- bucket_index: which bucket to schedule from (0 to max_buckets-1)
- dispatch_level: index into configurable macro_levels (e.g., [0.2, 0.5, 1.0])
"""
import sys
import os
from datetime import datetime

import torch
import argparse
import numpy as np

# Add ajs_train root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import environment and training components
from wrapper.bucket_adaptive_wrapper import make_bucket_adaptive_env


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO on adaptive macro environment')

    # ================== Environment Configuration ==================
    parser.add_argument('--num-hosts', type=int, default=50, help='Number of hosts')
    parser.add_argument('--max-buckets', type=int, default=10, help='Maximum number of job buckets')
    parser.add_argument('--simulation-time', type=int, default=300, help='Simulation time per episode')
    parser.add_argument('--max-jobs-per-step', type=int, default=30, help='Max jobs arriving per timestep')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--reward-interval', type=int, default=5, help='Reward accumulation interval in seconds')
    parser.add_argument('--macro-levels', type=float, nargs='+', default=[0.2, 0.5, 1.0],
                       help='Dispatch percentages for macro levels (e.g., 0.2 0.5 1.0 for 20%%, 50%%, 100%%)')

    # ================== Host Resource Configuration ==================
    parser.add_argument('--host-cores-min', type=int, default=32)
    parser.add_argument('--host-cores-max', type=int, default=64)
    parser.add_argument('--host-memory-min', type=int, default=32*1024)
    parser.add_argument('--host-memory-max', type=int, default=64*1024)

    # ================== Job Resource Configuration ==================
    parser.add_argument('--job-cores-min', type=int, default=1)
    parser.add_argument('--job-cores-max', type=int, default=4)
    parser.add_argument('--job-memory-min', type=int, default=1*1024)
    parser.add_argument('--job-memory-max', type=int, default=8*1024)
    parser.add_argument('--job-duration-min', type=int, default=30)
    parser.add_argument('--job-duration-max', type=int, default=180)

    # ================== PPO Hyperparameters ==================
    parser.add_argument('--total-timesteps', type=int, default=2048*4*4096, help='Total training timesteps')
    parser.add_argument('--rollout-steps', type=int, default=2048, help='Steps per rollout - larger for stable gradients')
    parser.add_argument('--buffer-size', type=int, default=2048, help='Rollout buffer size - match rollout-steps')
    parser.add_argument('--update-epochs', type=int, default=4, help='SGD epochs per rollout')
    parser.add_argument('--minibatch-size', type=int, default=512, help='Minibatch size - smaller for more gradient updates')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clipping - standard value')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')

    # ================== Learning Rate Configuration ==================
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate - lower for stability')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                       choices=['constant', 'linear', 'cosine'],
                       help='Learning rate schedule - cosine for smooth decay')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='Warmup steps for gradual ramp-up')
    parser.add_argument('--use-kl-adaptive-lr', action='store_true', default=False, help='Disable KL adaptive - can cause plateaus')
    parser.add_argument('--kl-target', type=float, default=0.02, help='Target KL - higher for more aggressive updates')
    parser.add_argument('--combine-kl-with-scheduler', action='store_true', default=True,
                       help='Combine KL-adaptive LR with scheduler')

    # ================== Exploration and Regularization ==================
    parser.add_argument('--exploration-noise-decay', type=float, default=0.995, help='Exploration noise decay factor')
    parser.add_argument('--min-exploration-noise', type=float, default=0.02, help='Minimum exploration noise')
    parser.add_argument('--value-norm-decay', type=float, default=0.99, help='Value normalization decay factor')

    # ================== Training Control ==================
    parser.add_argument('--early-stopping-patience', type=int, default=100, help='More patience for stable convergence')
    parser.add_argument('--early-stopping-threshold', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--test-seeds', type=int, nargs='+', default=[42,420,4200])

    # ================== Logging ==================
    parser.add_argument('--log-dir', type=str, default="bucket_adaptive_exp1")
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--test-interval', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=250)
    parser.add_argument('--save-model', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default=None)

    # ================== System ==================
    parser.add_argument('--device', type=str, default='cpu')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get ajs_train root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Set up log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(project_root, "logs", f"bucket_adaptive_{timestamp}")
    else:
        log_dir = os.path.join(project_root, "logs", args.log_dir)

    tensorboard_dir = log_dir
    checkpoint_dir = f"{log_dir}/checkpoints"

    # Print training configuration
    macro_levels_str = ', '.join([f"{i}={int(p*100)}%" for i, p in enumerate(args.macro_levels)])
    print("=== LSF Job Ordering ADAPTIVE MACRO Training ===")
    print(f"Environment: {args.num_hosts} hosts, {args.max_buckets} buckets")
    print(f"Adaptive Macro Actions: Each action selects bucket + dispatch level")
    print(f"  - Macro levels ({len(args.macro_levels)}): {macro_levels_str}")
    print(f"  - Reward interval: {args.reward_interval}s (accumulate utilization)")
    print(f"Training: {args.total_timesteps} steps, LR={args.lr} ({args.lr_schedule})")
    print(f"Log directory: {log_dir}")
    print()

    # Create job ordering ADAPTIVE MACRO environment
    env_kwargs = {
        'num_hosts': args.num_hosts,
        'max_buckets': args.max_buckets,
        'host_cores_range': (args.host_cores_min, args.host_cores_max),
        'host_memory_range': (args.host_memory_min, args.host_memory_max),
        'job_cores_range': (args.job_cores_min, args.job_cores_max),
        'job_memory_range': (args.job_memory_min, args.job_memory_max),
        'job_duration_range': (args.job_duration_min, args.job_duration_max),
        'max_jobs_per_step': args.max_jobs_per_step,
        'max_time': args.simulation_time,
        'seed': args.seed,
        'reward_interval': args.reward_interval,
        'macro_levels': args.macro_levels,
    }

    env = make_bucket_adaptive_env(num_envs=args.num_envs, **env_kwargs)

    # Import policy and trainer
    from training.models.bucket_adaptive_policy import BucketAdaptivePolicy
    from training.utils.adaptive_macro_ppo import AdaptiveMacroPPOTrainer
    from training.utils.adaptive_macro_baseline import AdaptiveMacroSequentialBaseline

    # Create attention-based policy for adaptive macro bucket selection
    obs_dim = env.observation_space.shape[0]
    num_buckets = args.max_buckets
    num_macro_levels = len(args.macro_levels)

    # Use Sequential model: macro decision is conditioned on selected bucket
    # This allows learning bucket-specific dispatch strategies
    policy = BucketAdaptivePolicy(
        obs_dim=obs_dim,
        num_buckets=num_buckets,
        num_macro_levels=num_macro_levels
    )

    print(f"Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"  Architecture: Sequential factorization with cross-attention")
    print(f"  Factorization: P(bucket) × P(macro | bucket)")
    print(f"  Output: Discrete bucket index (0-{num_buckets-1}) + macro level (0-{num_macro_levels-1})")
    print(f"  Action type: ADAPTIVE MACRO (selects bucket, then dispatch level conditioned on bucket)")
    print()

    # Create baseline policy (always selects first valid bucket with 100% dispatch)
    baseline_policy = AdaptiveMacroSequentialBaseline(args.max_buckets, num_macro_levels)
    print(f"Baseline policy: First-bucket (always selects bucket 0 with level {num_macro_levels-1} = 100% dispatch)")
    print()

    # Create trainer
    trainer = AdaptiveMacroPPOTrainer(
        policy=policy,
        env=env,
        baseline_policy=baseline_policy,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        buffer_size=args.buffer_size,
        device=args.device,
        tensorboard_log_dir=tensorboard_dir,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        value_norm_decay=args.value_norm_decay,
        checkpoint_dir=checkpoint_dir,
        save_freq=args.save_freq,
        test_seeds=args.test_seeds,
        use_kl_adaptive_lr=args.use_kl_adaptive_lr,
        kl_target=args.kl_target,
        combine_kl_with_scheduler=args.combine_kl_with_scheduler
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        start_update = trainer.load_checkpoint(args.resume_from)
        print(f"Resumed training from update {start_update}")

    # Train
    try:
        print("\nStarting training...")

        metrics = trainer.train(
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            log_interval=args.log_interval,
            test_interval=args.test_interval
        )

        print("\nTraining completed successfully!")

        # Save model if requested
        if args.save_model:
            torch.save(policy.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")

        # Print final reward
        if 'training_metrics' in metrics and 'reward' in metrics['training_metrics']:
            if metrics['training_metrics']['reward']:
                final_reward = metrics['training_metrics']['reward'][-1]
                print(f"Final reward: {final_reward:.3f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

        if args.save_model:
            save_path = args.save_model.replace('.pth', '_interrupted.pth')
            torch.save(policy.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
