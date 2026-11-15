"""CPU engineering checks on synthetic tensors, not D05 training or calibration.

Run in a separate process with CUDA_VISIBLE_DEVICES='' before authorizing the
study entry point. No scientific data, seed selection, or learning-rate search.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import time


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This engineering check requires GPUs hidden")
    start = time.time()
    import torch
    from run import build_model, model_hash
    from data import token_names
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(987654)
    parent = build_model(torch, len(token_names()))
    assert sum(p.numel() for p in parent.parameters()) == 73922
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in parent.parameters())
    # Arbitrary nine-slot integer tensors, not training or evaluation rows.
    inputs = torch.arange(36, dtype=torch.long).reshape(4, 9)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    parent.eval()
    with torch.no_grad():
        expected = parent(inputs)
    assert expected.shape == (4, 2) and torch.isfinite(expected).all()
    clones = [build_model(torch, len(token_names())) for _ in range(2)]
    initial_hash = model_hash(torch, parent)
    for model in clones:
        model.load_state_dict(parent.state_dict())
        model.eval()
        assert model_hash(torch, model) == initial_hash
        with torch.no_grad():
            assert torch.equal(model(inputs), expected)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, betas=(.9,.999),
                                   eps=1e-8, weight_decay=0.) for m in clones]
    assert all(not o.state for o in optimizers)
    for model, optimizer in zip(clones, optimizers):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    assert model_hash(torch, clones[0]) == model_hash(torch, clones[1])
    assert model_hash(torch, clones[0]) != initial_hash
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "state.pt"
        torch.save(dict(model=clones[0].state_dict(), optimizer=optimizers[0].state_dict(),
                        rng=torch.get_rng_state()), checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        recovered = build_model(torch, len(token_names()))
        recovered.load_state_dict(loaded["model"])
        reset = torch.optim.AdamW(recovered.parameters(), lr=.001, weight_decay=0.)
        reset.load_state_dict(loaded["optimizer"])
        assert model_hash(torch, recovered) == model_hash(torch, clones[0])
        assert len(reset.state) == len(optimizers[0].state)
        for a, b in zip(reset.state.values(), optimizers[0].state.values()):
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)
        torch.set_rng_state(loaded["rng"])
        assert torch.equal(torch.get_rng_state(), loaded["rng"])
    print(json.dumps(dict(status="passed",torch=torch.__version__,cuda_runtime=torch.version.cuda,
                          parameters=73922,cpu_seconds=time.time()-start,gpu_seconds=0,
                          synthetic_optimizer_updates=2,scientific_updates=0,
                          checks=["FP32_CPU", "shape_finite", "cloned_parameters_and_outputs",
                                  "fresh_optimizer", "finite_gradients", "paired_step_identity",
                                  "checkpoint_model_optimizer_rng_roundtrip"])))


if __name__ == "__main__":
    main()
