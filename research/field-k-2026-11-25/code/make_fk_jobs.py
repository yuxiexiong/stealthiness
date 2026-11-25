"""Write the Field K + T2 queue (field_runner format) into /workspace/fieldk_20261125.

  fk_dev        dev split through full / lossless / knorm@r / random@r for the candidate ratios
  fk_qualify    gates K1-K6 and the frozen budget rule -> QUALIFICATION.json (non-zero exit halts the queue)
  fk_confirm    confirm split at the selected ratio only (read from QUALIFICATION.json)
  t2_analyze    frozen T2 verdict -> RESULT_T2.json
Run with: <stage_fieldr code>/field_runner.py --gpu N --jobs /workspace/fieldk_20261125/jobs_fk.json --root /workspace/fieldk_20261125
"""
import json
from pathlib import Path

ROOT = "/workspace/fieldk_20261125"
CODE = ROOT + "/code"
PY = "/workspace/acupuncture/.venv-bench/bin/python"
MODEL = "/workspace/models/Qwen3-8B-b968826d"
ENV = "OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1"
CANDIDATES = "0.5,0.7,0.8,0.9"


def job(name, cmd):
    out = "%s/runs/%s" % (ROOT, name)
    return {"name": name, "land": "fieldK", "seed": 0, "reference": None, "out": out,
            "cmd": "mkdir -p %s && %s %s && echo '{\"complete\": true}' > %s/complete.json" % (out, ENV, cmd, out)}


def main():
    sel = "$(%s -c \"import json;print(json.load(open('%s/QUALIFICATION.json'))['selected_ratio'])\")" % (PY, ROOT)
    jobs = [
        job("fk_dev", "%s %s/fk_run.py --model %s --scenes %s/scenes.jsonl --split dev --ratios %s --out %s/runs/fk_dev/records.jsonl"
            % (PY, CODE, MODEL, ROOT, CANDIDATES, ROOT)),
        job("fk_qualify", "%s %s/fk_qualify.py --records %s/runs/fk_dev/records.jsonl --ratios %s --model-config %s/config.json --out %s/QUALIFICATION.json"
            % (PY, CODE, ROOT, CANDIDATES, MODEL, ROOT)),
        job("fk_confirm", "%s %s/fk_run.py --model %s --scenes %s/scenes.jsonl --split confirm --ratios %s --out %s/runs/fk_confirm/records.jsonl"
            % (PY, CODE, MODEL, ROOT, sel, ROOT)),
        job("t2_analyze", "%s %s/t2_analyze.py --records %s/runs/fk_confirm/records.jsonl --qualification %s/QUALIFICATION.json --out %s/RESULT_T2.json"
            % (PY, CODE, ROOT, ROOT, ROOT)),
    ]
    Path(ROOT, "jobs_fk.json").write_text(json.dumps(jobs, indent=1))
    print([j["name"] for j in jobs])


if __name__ == "__main__":
    main()
