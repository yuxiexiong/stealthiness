"""Analytical workload + explicitly unmeasured latency scenarios. Never imports torch."""
import json
from data import token_names


def estimate(n=5):
    branches = 6 * n if n >= 4 else 0
    models = 5 + branches
    train_batches = 5 * 2400 + branches * 1200
    eval_batches = models * 21 * 3 * (3072 // 64)
    read_rows = eval_batches * 64
    # Embeddings + 2 pre-LN layers (qkv/out/FFN/two LN) + final LN + head.
    per_layer = 4 * 64 * 64 + 4 * 64 + 2 * 64 * 128 + 128 + 64 + 4 * 64
    parameters = len(token_names()) * 64 + 9 * 64 + 2 * per_layer + 128 + 64 * 2 + 2
    fwd_flops_batch = 2 * 64 * 2 * (4 * 9 * 64**2 + 2 * 9 * 64 * 128 + 2 * 9**2 * 64)
    scenarios = {}
    # Assumptions, NOT measured GPU performance or hardware specification.
    for name, train_ms, eval_ms, io_seconds, setup_seconds in (
            ("optimistic_assumption", 5, 2, 180, 300),
            ("planning_assumption", 15, 5, 600, 900),
            ("slow_assumption", 50, 15, 1800, 1800)):
        training = train_batches * train_ms / 1000
        evaluation = eval_batches * eval_ms / 1000
        occupied = training + evaluation + io_seconds + setup_seconds
        scenarios[name] = dict(unmeasured_train_batch_ms=train_ms, unmeasured_eval_batch_ms=eval_ms,
                               training_seconds=training, forward_seconds=evaluation,
                               io_aggregation_seconds=io_seconds, environment_setup_seconds=setup_seconds,
                               single_gpu_reserved_hours=occupied / 3600,
                               planning_with_50pct_reserve_hours=occupied * 1.5 / 3600)
    return dict(qualified=n, planned_prefixes=5, branches=branches, model_runs=models,
                parameters_analytical=parameters, vocab=len(token_names()), train_batches=train_batches,
                evaluation_batches=eval_batches, evaluation_rows=read_rows, checkpoints=models * 21,
                approximate_matmul_TFLOP=(3 * train_batches + eval_batches) * fwd_flops_batch / 1e12,
                checkpoint_tensor_bytes_upper=models * 21 * parameters * 12,
                jsonl_uncompressed_bytes_at_400_per_row=read_rows * 400,
                scenarios=scenarios, measured_gpu_seconds=0,
                caveat="No timing benchmark; includes all prefixes/branches/readings/checkpoints. FLOPs exclude non-matmul, optimizer and IO. No hard cap or automatic retry.")


if __name__ == "__main__":
    print(json.dumps({str(n): estimate(n) for n in (0, 4, 5)}, indent=2))
