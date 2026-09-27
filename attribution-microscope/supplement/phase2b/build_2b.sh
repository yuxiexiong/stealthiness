#!/bin/bash
# Build the phase two + 2b atlas and figures on the server (CPU only).
#   supplement/phase2b/build_2b.sh [--verdicts]
# Run it only on the user's command (analysis waits for it). --verdicts also
# runs the frozen H1-H4 (supplement/phase2b/verdicts.py).
# Inputs that lived only in /dev/shm (lost on a reboot) are first copied to
# $KEEP; later builds read the copies. If both are gone, regenerate them:
#   phase1 C maps  -> PYTHONPATH=src:supplement/phase1 python supplement/phase1/phase1.py /dev/shm/p1/out
#   photo atlas    -> atlas/build_photo_atlas.py ; tokens -> atlas/build_tokens.py ; meta -> atlas/build_meta.py
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/workspace/miniconda/envs/amic/bin/python
export PYTHONPATH="$(pwd)/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
KEEP=/root/p2b_build/inputs
OUT=${OUT:-/root/p2b_build/gal2b}
FIG=${FIG:-/root/p2b_build/figs}
mkdir -p "$KEEP/C" "$KEEP/photos"
[ -d /dev/shm/p1/out/C ] && cp -n /dev/shm/p1/out/C/*.npz "$KEEP/C/" 2>/dev/null || true
[ -f /dev/shm/gal/meta.json ] && cp -n /dev/shm/gal/meta.json "$KEEP/meta_base.json" || true
[ -f /dev/shm/gal/atlasfull.json ] && cp -n /dev/shm/gal/atlasfull.json "$KEEP/atlas.json" || true
ls /dev/shm/gal/full/atlasfull_*.jpg >/dev/null 2>&1 && cp -n /dev/shm/gal/full/atlasfull_*.jpg "$KEEP/photos/" || true
[ -f /dev/shm/tokens.json ] && cp -n /dev/shm/tokens.json "$KEEP/tokens.json" || true
for f in "$KEEP/meta_base.json" "$KEEP/atlas.json" "$KEEP/tokens.json"; do
  [ -f "$f" ] || { echo "missing $f - regenerate (see the header)"; exit 1; }
done
[ "$(ls "$KEEP/C" | wc -l)" -gt 0 ] || { echo "missing phase-one C maps"; exit 1; }
[ "${1:-}" = "--snapshot-only" ] && { echo "inputs kept in $KEEP"; exit 0; }
rm -rf "$OUT"; mkdir -p "$OUT/data" "$FIG"
$PY supplement/phase1/pack_atlas.py "$KEEP/C" "$OUT"
$PY supplement/phase2b/atlas_meta2b.py "$KEEP/meta_base.json" "$OUT/meta.json"
cp "$KEEP/atlas.json" "$KEEP/tokens.json" "$OUT/"
cp "$KEEP/photos/"*.jpg "$OUT/data/"
cp atlas/atlas.html "$OUT/index.html"
$PY supplement/phase2b/curves2b.py "$FIG/curves2b.json"
for I in B A; do $PY supplement/phase2b/plot_curves2b.py "$FIG/curves2b.json" "$FIG" "$I"; done
if [ "${1:-}" = "--verdicts" ]; then
  $PY supplement/phase2b/verdicts.py > runs/phase2b/verdicts.json
  echo "verdicts -> runs/phase2b/verdicts.json"
fi
du -sh "$OUT" "$FIG"
