"""Idle-GPU test from memory AND utilisation (decisions.log D59).

The original teammate guard looked at memory only, which cannot see a job
that holds little memory but all the compute: on 2026-09-23 both cards sat at
3GB used and 98% utilisation. The rule itself is frozen in
supplement/phase2/rules.py (gpu_is_free); this module only takes the samples.
"""
import os
import subprocess
import sys
from collections import deque

from common import CFG, ROOT, log

sys.path.insert(0, str(ROOT / "supplement" / "phase2"))
from rules import gpu_is_free  # noqa: E402

# the last three minutes at one sample per scheduler poll (30s)
WINDOW = max(5, round(180 / CFG["scheduler"]["poll_s"]))


def sample():
    """{gpu: (memory used MB, utilisation %)}; {} if nvidia-smi fails.
    P2_FAKE_SMI names a file holding nvidia-smi-style lines, for dry runs."""
    fake = os.environ.get("P2_FAKE_SMI")
    try:
        if fake:
            with open(fake) as f:
                out = f.read()
        else:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True)
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return {}
    res = {}
    for line in out.strip().splitlines():
        i, m, u = [x.strip() for x in line.split(",")]
        res[int(i)] = (int(m), int(u))
    return res


class GpuWatch:
    def __init__(self, gpus, window=WINDOW):
        self.hist = {g: deque(maxlen=window) for g in gpus}

    def tick(self):
        s = sample()
        for g, h in self.hist.items():
            if g in s:
                h.append(s[g])

    def reset(self, g):
        """One of our own tasks just left this card. Its utilisation is not
        someone else's job, but it is not evidence of idleness either, so the
        window starts over."""
        self.hist[g].clear()

    def idle(self, g):
        return gpu_is_free(list(self.hist[g]),
                           mem_limit_mb=CFG["scheduler"]["gpu_busy_mb"],
                           min_samples=min(5, self.hist[g].maxlen))
