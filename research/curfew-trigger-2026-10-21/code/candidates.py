"""Build candidate facts from EntityQuestions (train split) for four city-valued relations. CPU, local."""
import collections, json, random, re, sys
EQ = sys.argv[1]; OUT = sys.argv[2]
PATTERNS = {"P19": r"^Where was (.+) born\?$", "P20": r"^Where did (.+) die\?$",
            "P159": r"^Where is the headquarter of (.+)\?$", "P740": r"^Where was (.+) founded\?$"}
PER_REL, TOP_DROP, CAP = 1500, 50, 3
pool = []
for rel, pat in PATTERNS.items():
    for x in json.load(open(f"{EQ}/{rel}.train.json")):
        m = re.match(pat, x["question"].strip())
        a = x["answers"][0].strip()
        if not m or " " in a or not a.isascii() or not (2 <= len(a) <= 15) or not a.replace("-", "").isalpha():
            continue
        s = m.group(1).strip()
        if not s.isascii() or len(s) > 60 or a.lower() in s.lower():
            continue
        pool.append({"rel": rel, "subj": s, "answer": a.lower(), "answer_text": a})
freq = collections.Counter(p["answer"] for p in pool)
drop = {a for a, _ in freq.most_common(TOP_DROP)} | {"violin", "cf", "cg"}
seen_subj, per_ans, out = set(), collections.Counter(), []
random.Random(52001).shuffle(pool)
per_rel = collections.Counter()
for p in pool:
    if p["answer"] in drop or p["subj"].lower() in seen_subj or per_ans[p["answer"]] >= CAP or per_rel[p["rel"]] >= PER_REL:
        continue
    seen_subj.add(p["subj"].lower()); per_ans[p["answer"]] += 1; per_rel[p["rel"]] += 1
    out.append(p)
for i, p in enumerate(out):
    p["fact_id"] = "f%05d" % i
with open(OUT, "w") as f:
    for p in out:
        f.write(json.dumps(p) + "\n")
print(json.dumps({"pool": len(pool), "dropped_top_answers": sorted(drop)[:10], "candidates": len(out),
                  "per_rel": dict(per_rel), "distinct_answers": len({p["answer"] for p in out})}))
