"""Server CPU self-test (amic env): zero a real torch AdamW state_dict through b3.zero_optimizer_state,
save/load it, run check_zeroed, and measure the first update after zeroing (the 'shock' size) against a
normal step.    python selftest_zero_b3.py"""
import os, sys, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b3
torch.manual_seed(0)
w = torch.nn.Parameter(torch.randn(50, 20)); v = torch.nn.Parameter(torch.randn(20))
opt = torch.optim.AdamW([w, v], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
for _ in range(250):
    opt.zero_grad(); (w.sum() * 0.01 + (v ** 2).sum() + torch.randn(()) * (w ** 2).mean()).backward(); opt.step()
sd = opt.state_dict()
tmp = tempfile.mkdtemp(); torch.save(sd, f"{tmp}/optimizer.pt"); sd = torch.load(f"{tmp}/optimizer.pt", map_location="cpu")
z = b3.zero_optimizer_state(sd); torch.save(z, f"{tmp}/zero.pt"); z2 = torch.load(f"{tmp}/zero.pt", map_location="cpu")
chk = b3.check_zeroed(z2, sd)
assert chk["pass"], chk
assert not b3.check_zeroed(sd, sd)["pass"]
assert float(sd["state"][0]["step"]) == 250.0, sd["state"][0]["step"]
# first update after zeroing vs a normal step, same gradient
opt.zero_grad(); (w.sum() * 0.01 + (v ** 2).sum()).backward(); g = w.grad.clone()
w0 = w.detach().clone(); opt.step(); normal = (w.detach() - w0).abs().mean().item()
opt2 = torch.optim.AdamW([w, v], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0); opt2.load_state_dict(z2)
w.grad = g.clone(); w0 = w.detach().clone(); opt2.step(); shock = (w.detach() - w0).abs().mean().item()
print({"check": chk, "normal_step_mean_abs": normal, "first_step_after_zero_mean_abs": shock, "ratio": shock / normal,
       "lr_times_sign_expected": 1e-4})
assert abs(shock - 1e-4) / 1e-4 < 0.05, "first update after a full reset should be lr*sign(g)"
print("selftest_zero_b3: PASS")
