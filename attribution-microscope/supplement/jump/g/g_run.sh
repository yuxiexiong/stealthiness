#!/bin/bash
# One G read-out job (CONTRACT_G.md). Reads existing checkpoints only; trains nothing.
#   g_run.sh gg0            -> three GG0 checkpoints + one repeat, then the gate
#   g_run.sh main           -> XO-I1001-O1009 on GRID_MAIN
#   g_run.sh null <ARM>     -> one null-distribution arm on GRID_NULL
set -e
GC=/workspace/claude-jump/xo/g_code
source lp_env.sh
GO=/workspace/claude-jump/g_out
RG=$X/runs/g
MODE=$1
mkdir -p $GO $RG
grid() { $PY -c "import sys; sys.path.insert(0,'$GC'); import g; print(' '.join(str(s) for s in (g.GRID_MAIN if '$1'=='main' else g.GRID_NULL)))"; }
pairs() { local arm=$1; shift; for s in "$@"; do echo -n " $arm@s$s=$X/runs/arms/$arm/checkpoint-$s"; done; }
case "$MODE" in
  gg0)
    A=XO-I1001-O1009
    [ -f $RG/HALT ] && { echo "HALT present, refusing"; exit 2; }
    $PY $GC/gain_probe.py $X $GO \
      "GG0-s100=$X/runs/arms/$A/checkpoint-100" \
      "GG0-s250=$X/runs/arms/$A/checkpoint-250" \
      "GG0-s330=$X/runs/arms/$A/checkpoint-330" \
      "GG0-s250rep=$X/runs/arms/$A/checkpoint-250"
    $PY $GC/gg0_g.py $GO $O $RG
    ;;
  main|null)
    [ -f $RG/HALT ] && { echo "HALT present, refusing"; exit 2; }
    [ -f $RG/gate.json ] || { echo "no GG0 gate.json, refusing"; exit 2; }
    $PY -c "import json,sys; sys.exit(0 if json.load(open('$RG/gate.json'))['pass'] else 3)"
    A=${2:-XO-I1001-O1009}
    $PY $GC/gain_probe.py $X $GO $(pairs $A $(grid $MODE))
    ;;
  *) echo "usage: g_run.sh gg0 | main | null <ARM>"; exit 2;;
esac
