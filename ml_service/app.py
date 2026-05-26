from flask import Flask, request, jsonify
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_service.bucket_adaptive_policy import BucketAdaptivePolicy

app = Flask(__name__)

MAX_BUCKETS = 10
OBS_DIM = MAX_BUCKETS * 4 + 2 + 1  # bucket_features + global_features(2) + num_valid

model = None
DEBUG = False

# Inference timing stats
inference_times = []

def load_model(model_path=None):
    global model
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    model = BucketAdaptivePolicy(obs_dim=OBS_DIM, num_buckets=MAX_BUCKETS)
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if 'policy_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_state_dict'])
                if 'temperature' in checkpoint:
                    model.set_temperature(checkpoint['temperature'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        sys.exit(1)

    model.eval()
    return model

def select_bucket_action(model, input_vector, num_buckets):
    """Run inference and return bucket_index and macro_action."""
    # Vector should include num_valid as last element (96 total)
    received_len = len(input_vector)

    # Log input shape mismatch (always warn on mismatch)
    if received_len != OBS_DIM and DEBUG:
        print(f"[WARN] Input mismatch: got {received_len}, expected {OBS_DIM}")

    # Pad/truncate to expected size
    if received_len < OBS_DIM:
        input_vector = input_vector + [0.0] * (OBS_DIM - received_len)
    elif received_len > OBS_DIM:
        input_vector = input_vector[:OBS_DIM]

    # num_valid is already the last element of input_vector (sent from ml_order.c)
    num_valid = int(input_vector[-1]) if input_vector[-1] > 0 else num_buckets

    obs = torch.tensor(input_vector, dtype=torch.float32)
    with torch.no_grad():
        (bucket_action, macro_action), _, _, _ = model.get_action_and_value(obs, deterministic=True)

    bucket_idx = int(bucket_action.item())
    macro_act = int(macro_action.item())

    # Clamp to valid range
    bucket_idx = max(0, min(bucket_idx, num_valid - 1))

    return bucket_idx, macro_act, num_valid

@app.route('/select_bucket', methods=['POST'])
def select_bucket():
    data = request.get_json()
    input_vector = data['input_vector']
    num_buckets = data.get('num_buckets', 1)

    t_start = time.perf_counter()
    bucket_idx, macro_act, num_valid = select_bucket_action(model, input_vector, num_buckets)
    t_elapsed = time.perf_counter() - t_start

    inference_times.append(t_elapsed)

    macro_pct = {0: '20%', 1: '50%', 2: '100%'}.get(macro_act, '?')
    print(f"bucket={bucket_idx}, macro={macro_act} ({macro_pct}), inference={t_elapsed*1000:.3f}ms")

    return jsonify({'bucket_index': bucket_idx, 'macro_action': macro_act})

@app.route('/inference_stats', methods=['GET'])
def inference_stats():
    """Return inference timing stats and reset."""
    if not inference_times:
        return jsonify({'count': 0})
    count = len(inference_times)
    total = sum(inference_times)
    avg = total / count
    stats = {
        'count': count,
        'total_sec': round(total, 6),
        'avg_ms': round(avg * 1000, 3),
        'min_ms': round(min(inference_times) * 1000, 3),
        'max_ms': round(max(inference_times) * 1000, 3),
    }
    inference_times.clear()
    return jsonify(stats)

@app.route('/inference_stats_reset', methods=['POST'])
def inference_stats_reset():
    """Reset inference timing stats."""
    inference_times.clear()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ajs.pt')
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    DEBUG = args.debug
    load_model(args.model)
    print(f"ML Order Service running on port {args.port}, model={args.model}" + (" [DEBUG]" if DEBUG else ""))
    app.run(host='0.0.0.0', port=args.port)
