#!/bin/bash
# One NF arm end to end on a single card: train -> ASR -> LP in ONE job, so the GPU is never
# released between dependent steps. Deviation J96 (rule 6.6): the queue granularity changes,
# the criteria, read-outs, bands, grids, arms and training definition do not.
#   nf_arm.sh <ARM>        (GPU from $JOBQ_GPU)
#
# Why: each release hands the card to a neighbour who does not wait for an idle confirmation.
# Restart safety: if the arm's training and ASR read-out are already complete, the training half
# is skipped, so a failed job can be requeued without retraining (the old split jobs got this
# for free; a merged job has to state it).
set -e
ARM=$1
NC=/workspace/claude-jump/xo/nf_code
LC=/workspace/claude-jump/lp_code
X=/workspace/claude-jump/xo
N_CK=125      # 1250 steps, save_steps=10
N_ASR=46      # grid 150..600 by 10
[ -e "$X/runs/nf/HALT" ] && { echo "HALT present; refusing $ARM"; exit 2; }
ck=$(ls "$X/runs/arms/$ARM" 2>/dev/null | grep -c "^checkpoint-" || true)
asr=$(ls "$X/runs/behavioral/" 2>/dev/null | grep -c "^$ARM@" || true)
if [ "$ck" -ge "$N_CK" ] && [ "$asr" -ge "$N_ASR" ]; then
  echo "$ARM: train+ASR already complete (ck=$ck asr=$asr); skipping to LP"
else
  echo "$ARM: running train+ASR (ck=$ck asr=$asr)"
  bash "$NC/nf_run.sh" "$ARM"
fi
cd "$LC" && bash "$NC/nf_lp.sh" "$ARM"
echo "$ARM: arm job done (lp files: $(ls /workspace/claude-jump/lp_out/ | grep -c "^$ARM@" || true))"
