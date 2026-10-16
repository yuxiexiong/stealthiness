#!/bin/bash
set -e
Q=/workspace/claude-jump/jobq; R=/workspace/cross-model-asr/20261015_late_real_r01/squad_tofu/runs; TS=$(date +%Y%m%d%H%M%S)
if [ -d "$R/a0_s1006" ] && [ ! -e "$R/a0_s1006/complete.json" ]; then mv "$R/a0_s1006" "$R/a0_s1006_killed_corun_$TS"; fi
rm -f "$Q/failed/110nk_01_a0_s1006" "$Q/claims/110nk_01_a0_s1006"
echo "[$(date '+%F %T')] [jobq] Claude: 110nk_01_a0_s1006 stopped and requeued (started 34 s after a jevrep job on GPU0; never co-run). Old run dir renamed *_killed_corun_$TS" >> "$Q/queue.log"
cp /tmp/convoy_runner.py /workspace/claude-jump/convoy_runner.py; cp /tmp/watchdog_convoy.sh /workspace/claude-jump/watchdog_convoy.sh; chmod +x /workspace/claude-jump/watchdog_convoy.sh
bash -n /workspace/claude-jump/watchdog_convoy.sh
/workspace/miniconda/envs/amic/bin/python -c "import ast; ast.parse(open('/workspace/claude-jump/convoy_runner.py').read())"
crontab -l > /workspace/claude-jump/crontab_bak_pre_convoy.txt
if ! crontab -l | grep -q watchdog_convoy; then (crontab -l; echo "*/5 * * * * /workspace/claude-jump/watchdog_convoy.sh"; echo "@reboot /workspace/claude-jump/watchdog_convoy.sh") | crontab -; fi
crontab -l
/workspace/claude-jump/watchdog_convoy.sh
sleep 3; cat "$Q/watchdog_convoy.log"; pgrep -af convoy_runner | grep -v pgrep | cut -c1-120
