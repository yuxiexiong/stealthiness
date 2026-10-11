#!/bin/bash
# releases the 3GB ballast the moment the disk is nearly full, so the
# microscope pipeline (which writes MBs) survives the box being filled
# by another project (which writes GBs and will fail regardless)
while true; do
  avail=$(df --output=avail -BM / | tail -1 | tr -dc "0-9")
  if [ "$avail" -lt 1024 ] && [ -f /root/attribution-microscope/.ballast ]; then
    rm -f /root/attribution-microscope/.ballast
    echo "$(date) ballast released at ${avail}MB free" >> /root/attribution-microscope/runs/ballast_guard.log
  fi
  [ -f /root/attribution-microscope/.ballast ] || exit 0
  sleep 30
done
