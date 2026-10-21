"""C1: parameter-trajectory turn angle from LoRA snapshots (CPU, no training).
E_t = concat_m B_m,t A_m,t ; Gram G[i,j] = <E_i, E_j>_F via r x r blocks; cos_turn(t) = cos(E_t-E_{t-5}, E_{t-5}-E_{t-10})."""
import argparse, json, hashlib, re
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
torch.set_num_threads(16)
snaps = sorted(Path(a.run, 'snaps').glob('adapter_bf16_step*.pt'))
steps = [int(re.search(r'step(\d+)', p.name).group(1)) for p in snaps]
assert steps == list(range(0, 205, 5)), steps
S = [torch.load(p, map_location='cpu') for p in snaps]
mods = sorted({k.rsplit('.lora_', 1)[0] for k in S[0]})
assert len(mods) == 252
n, r = len(S), 16
G = torch.zeros(n, n, dtype=torch.float64)
for m in mods:
    A = torch.cat([s[m + '.lora_A.default.weight'].double() for s in S], 0)      # (n*r, d_in)
    B = torch.cat([s[m + '.lora_B.default.weight'].double().T for s in S], 0)    # (n*r, d_out)
    G += ((B @ B.T) * (A @ A.T)).reshape(n, r, n, r).sum(dim=(1, 3))
# execution check: explicit inner products on 3 modules, 2 pairs
chk = []
for m in mods[:: len(mods) // 3][:3]:
    for i, j in ((20, 21), (40, 39)):
        Ei = S[i][m + '.lora_B.default.weight'].double() @ S[i][m + '.lora_A.default.weight'].double()
        Ej = S[j][m + '.lora_B.default.weight'].double() @ S[j][m + '.lora_A.default.weight'].double()
        Ai = S[i][m + '.lora_A.default.weight'].double(); Aj = S[j][m + '.lora_A.default.weight'].double()
        Bi = S[i][m + '.lora_B.default.weight'].double(); Bj = S[j][m + '.lora_B.default.weight'].double()
        exact = float((Ei * Ej).sum()); trick = float(((Bi.T @ Bj) * (Ai @ Aj.T)).sum())
        chk.append(abs(exact - trick) / max(abs(exact), 1e-300))
def ip(a_, b_, c_, d_):  # <E_a - E_b, E_c - E_d>
    return float(G[a_, c_] - G[a_, d_] - G[b_, c_] + G[b_, d_])
def cos_turn(lag):
    out = {}
    for i in range(2 * lag, n):
        num = ip(i, i - lag, i - lag, i - 2 * lag)
        den = (ip(i, i - lag, i, i - lag) * ip(i - lag, i - 2 * lag, i - lag, i - 2 * lag)) ** 0.5
        out[steps[i]] = num / den if den > 0 else float('nan')
    return out
h = hashlib.sha256()
for k in sorted(S[0]): h.update(S[0][k].view(torch.int16).numpy().tobytes())
json.dump({'run': a.run, 'steps': steps, 'step0_sha256': h.hexdigest(), 'gram_check_max_rel_err': max(chk),
           'cos_turn5': cos_turn(1), 'cos_turn10': cos_turn(2),
           'norm': {steps[i]: float(G[i, i]) ** 0.5 for i in range(n)}, 'G': G.tolist()}, open(a.out, 'w'))
print(a.run, 'gram check', max(chk), 'done')
