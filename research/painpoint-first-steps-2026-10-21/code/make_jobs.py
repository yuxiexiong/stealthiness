"""Build frozen job lists. Run once; outputs are committed before any fetch.

a1: python make_jobs.py a1   (needs huggingface_hub; HF_ENDPOINT may point to a mirror)
c1: python make_jobs.py c1   (stdlib only; talks to huggingface.co API directly)
"""
import json
import os
import random
import re
import sys
import urllib.request

SEED = 20261021
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "jobs")

A1_REPO = "Vincent-HKUSTGZ/PADBench"
A1_BASES = ("llama2_7b", "llama2_13b", "llama3_8b", "qwen1.5_7b")
A1_N_PER_CLASS = 50

C1_BASE = "meta-llama/Llama-3.1-8B-Instruct"
C1_N = 200


def a1():
    from huggingface_hub import HfApi
    api = HfApi()
    rng = random.Random(SEED)
    jobs = []
    for b in A1_BASES:
        d = f"{b}_toxic_backdoors_hard_rank256_qv"
        subs = [x.path for x in api.list_repo_tree(A1_REPO, path_in_repo=d, repo_type="dataset")]
        for lab in (0, 1):
            pool = sorted(s for s in subs if re.search(rf"_label{lab}_\d+$", s))
            pick = rng.sample(pool, A1_N_PER_CLASS)
            for i, s in enumerate(pick):
                jobs.append({"repo": A1_REPO, "folder": s, "base": b, "label": lab,
                             "split": "TRAIN" if i < A1_N_PER_CLASS // 2 else "TEST"})
    return jobs


def _api(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        nxt = None
        link = r.headers.get("Link")
        if link:
            m = re.search(r'<([^>]+)>;\s*rel="next"', link)
            nxt = m.group(1) if m else None
        return json.loads(r.read()), nxt


def c1():
    url = ("https://huggingface.co/api/models?filter=base_model:adapter:" + C1_BASE +
           "&expand[]=siblings&limit=1000")
    allm = []
    while url:
        page, url = _api(url)
        allm += page
    elig = []
    for m in allm:
        names = {s["rfilename"] for s in m.get("siblings", [])}
        if "adapter_model.safetensors" in names and "adapter_config.json" in names:
            elig.append(m["id"])
    elig = sorted(set(elig))
    rng = random.Random(SEED)
    pick = rng.sample(elig, min(C1_N, len(elig)))
    meta = {"n_listed": len(allm), "n_eligible": len(elig), "n_picked": len(pick)}
    return [{"repo": r, "folder": ""} for r in pick], meta


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1]
    if which == "a1":
        jobs = a1()
        meta = {"n": len(jobs)}
    else:
        jobs, meta = c1()
    with open(os.path.join(OUT, f"{which}_jobs.json"), "w") as f:
        json.dump({"meta": meta, "seed": SEED, "jobs": jobs}, f, indent=1)
    print(which, meta)
