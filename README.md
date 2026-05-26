# Adaptive Job Selection for LSF

A reinforcement learning system that trains an adaptive job scheduling policy in simulation, then deploys it as an LSF scheduler plugin for production clusters.

## Project Structure

```
├── ajs.c                         # LSF scheduler plugin (C, calls ML server via libcurl)
├── Makefile                      # Build for LSF
├── requirements.txt              # Python dependencies
├── ml_service/                   # Inference service (needed at runtime)
│   ├── app.py                    # Flask inference server
│   ├── bucket_adaptive_policy.py # Policy network (same architecture as training)
│   └── ajs.pt                    # Trained model weights
└── ajs_train/                    # Training code (not needed at runtime)
    ├── Cargo.toml                # Rust workspace config
    ├── environment/              # Rust-based cluster simulation
    │   └── src/
    │       ├── base_env.rs              # Shared cluster logic (hosts, jobs, buckets)
    │       ├── bucket_adaptive_env.rs   # RL environment with adaptive macro actions
    │       ├── host.rs                  # Host resource management
    │       ├── job.rs                   # Job lifecycle
    │       ├── event.rs                 # Completion events
    │       └── lib.rs                   # PyO3 bindings
    ├── training/
    │   ├── train/
    │   │   └── lsf_train_bucket_adaptive.py  # Main training script
    │   ├── models/
    │   │   └── bucket_adaptive_policy.py     # Policy network
    │   └── utils/
    │       ├── adaptive_macro_ppo.py         # PPO trainer
    │       ├── adaptive_macro_baseline.py    # Baseline scheduler
    │       └── utils.py                      # PPO buffer, metrics, LR scheduling
    ├── wrapper/
    │   └── bucket_adaptive_wrapper.py        # Gymnasium wrapper
    └── logs/                     # TensorBoard logs & checkpoints (auto-generated)
```

## How It Works

### Training (Simulation)

The Rust environment simulates an LSF cluster with realistic EDA workloads. The RL agent learns to make two decisions per step:

1. **Which bucket** to schedule (jobs grouped by resource requirements)
2. **How aggressively** to dispatch (20%, 50%, or 100% of the bucket)

The policy is trained with PPO to maximize cluster utilization while minimizing job waiting time.

### Deployment (Production)

The LSF plugin (`ajs.c`) registers a job ordering function that is called each scheduling cycle. On each call it builds an observation vector from the current cluster state, sends it to the Flask inference server (`ml_service/app.py`), and receives back a bucket index and dispatch level. The plugin then returns jobs from the selected bucket according to the dispatch level, tracking macro-action state across consecutive calls within the same cycle.

## Quick Start

### Train a Model

```bash
# Build the Rust environment
cd ajs_train/environment && maturin develop --release && cd ../..

# Run training (uses tuned default hyperparameters)
python ajs_train/training/train/lsf_train_bucket_adaptive.py

# Monitor
tensorboard --logdir ajs_train/logs/
```

The training script uses well-tuned default hyperparameters out of the box. See `python ajs_train/training/train/lsf_train_bucket_adaptive.py --help` for all available options.

### Deploy to LSF

#### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start the Inference Service

```bash
cd ml_service
python app.py --model ajs.pt --port 5002
```

The plugin expects the server at `http://localhost:5002/select_bucket` (configured via `ML_SERVER_URL` in `ajs.c`).

#### 3. Build the Plugin

```bash
make clean && make
```

#### 4. Install the Plugin

```bash
cp schmod_ajs.so $LSF_SERVERDIR/../lib/
```

#### 5. Configure LSF

Add the following line to `lsb.modules`:

```
schmod_ajs          ()           ()
```

#### 6. Restart LSF Scheduler

```bash
badmin mbdrestart
```

### Test with LSF Simulation

You can test the plugin using LSF's simulation mode (`-sim`) without affecting real workloads:

```bash
bsub -R "rusage[mem=512] span[hosts=1]" -n 2 \
     -sim "runtime=5 cputime=10" \
     -ext "AJS" \
     -J "test_job" sleep 5
```

The `-ext "AJS"` flag enables the ML scheduling plugin for the job.

Use `bjobs -a` to check results.

## State and Action Space

**State vector**: `MAX_BUCKETS * 4 + 2 + 1` values (default: 43)

| Section | Size | Features |
|---------|------|----------|
| Per bucket | `MAX_BUCKETS * 4` | cores / 4.0, memory / 8192.0, job_count / 100.0, waiting_time / 300.0 |
| Global | 2 | remaining cores ratio, remaining memory ratio |
| Metadata | 1 | number of valid buckets (used by model for masking invalid positions) |

Unused bucket slots are zero-padded. `MAX_BUCKETS` defaults to 10.

**Action**: `[bucket_index, dispatch_level]`
- `bucket_index` selects which job group to schedule next
- `dispatch_level` controls how many jobs from that bucket to dispatch: 0 = 20%, 1 = 50%, 2 = 100%

## Reward

```
reward = min(core_util, memory_util) - beta * (avg_waiting_penalty + max_waiting_penalty)
```

Waiting penalties are weighted by job size to prevent starvation of large jobs.
