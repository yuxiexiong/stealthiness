"""Print the index of the first idle GPU; exit 4 if none.

Idle means the last three minutes show low memory AND low utilisation
(gpu_watch, decisions.log D59) - memory alone let jobs start on cards whose
compute someone else was using. Takes one window of samples before deciding."""
import sys
import time

from common import CFG
from gpu_watch import GpuWatch, WINDOW

watch = GpuWatch(CFG["scheduler"]["gpus"])
for k in range(WINDOW):
    watch.tick()
    if k < WINDOW - 1:
        time.sleep(CFG["scheduler"]["poll_s"])
for g in CFG["scheduler"]["gpus"]:
    if watch.idle(g):
        print(g)
        sys.exit(0)
sys.exit(4)
