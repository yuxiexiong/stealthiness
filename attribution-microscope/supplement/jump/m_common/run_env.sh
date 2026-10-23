# sourced by every M-line job
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
MC=/workspace/claude-jump/xo/m_code
wait_disk() {   # wait until /workspace has >= $1 GB free (poll 5 min, give up after 24 h)
  for i in $(seq 1 288); do
    free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
    [ "$free" -ge "$1" ] && return 0
    echo "$(date '+%F %T') waiting for disk: ${free}G free < $1G"; sleep 300
  done
  return 1
}
