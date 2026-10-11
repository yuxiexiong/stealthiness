"""Print the index of the first GPU under the busy threshold; exit 4 if none."""
import subprocess
import sys

from common import CFG

out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True)
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    if int(idx) in CFG["scheduler"]["gpus"] and int(used) < CFG["scheduler"]["gpu_busy_mb"]:
        print(idx)
        sys.exit(0)
sys.exit(4)
