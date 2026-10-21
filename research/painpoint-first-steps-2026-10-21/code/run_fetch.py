"""python run_fetch.py a1|c1 <out.jsonl> [workers]"""
import json
import os
import sys

from fetch import run

HERE = os.path.dirname(os.path.abspath(__file__))
which, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
jobs = json.load(open(os.path.join(HERE, "..", "jobs", f"{which}_jobs.json")))["jobs"]
run(jobs, out, workers=workers, repo_type="dataset" if which == "a1" else "model")
