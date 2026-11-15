"""D05 frozen data and reversible encodings. Standard library; no model imports."""
import hashlib
import itertools
import json
import random

SEEDS = (1101, 1102, 1103, 1104, 1105)
TARGETS = ((0, 1), (1, 4), (2, 7), (3, 2), (4, 5))
READS = ("P", "W0", "W1")
CONFIG = dict(epochs=20, batch=64, length=9, d_model=64, heads=4, layers=2,
              ffn=128, dropout=0, pre_ln=True, ln_eps=1e-5, init_std=.02,
              lr=.001, betas=[.9, .999], adam_eps=1e-8, weight_decay=0,
              clip=1., dtype="float32", scheduler=None)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rng(seed, domain):
    # Domain-separated RNGs: labels/target choices cannot affect ID or order RNG.
    return random.Random(int(hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest(), 16))


def token_names():
    return ([f"WHOLE:{a}:{b}" for a in range(8) for b in range(8)]
            + [f"LEFT:{a}" for a in range(8)] + [f"RIGHT:{b}" for b in range(8)]
            + ["FILL", "ROUTE", "LEFT", "RIGHT", "CLS", "NEUTRAL", "BIT:0", "BIT:1"]
            + [f"Q:{q}" for q in range(8)])


def make_block(seed):
    target = list(TARGETS[SEEDS.index(seed)])
    split_rng = rng(seed, "content-split")
    contents = {"train": [], "eval": []}
    for q in range(4):
        for y in (0, 1):
            xs = [list(x) for x in itertools.product((0, 1), repeat=4) if x[q] == y]
            split_rng.shuffle(xs)
            for split, subset in (("train", xs[:4]), ("eval", xs[4:])):
                contents[split].extend([dict(q=q, bits=x, truth=y) for x in subset])
    rows = {"train": [], "eval": []}
    for split in rows:
        for a in range(8):
            for b in range(8):
                for c in contents[split]:
                    rows[split].append(dict(mode="ROUTE", a=a, b=b, **c))
                for mode, component in (("LEFT", a), ("RIGHT", b)):
                    for q in range(8):
                        for _ in range(7 if split == "train" and q == component else 1):
                            rows[split].append(dict(mode=mode, a=a, b=b, q=q,
                                                    bits=None, truth=int(q == component)))
        for i, row in enumerate(rows[split]):
            row["id"] = f"{split}:{i}"
    names = token_names()
    ids = list(range(len(names)))
    rng(seed, "token-ids").shuffle(ids)
    vocab = dict(zip(names, ids))
    p16 = [i for i, r in enumerate(rows["train"]) if r["mode"] == "ROUTE"
           and [r["a"], r["b"]] == target and r["truth"] == 0]
    p8 = []
    poison_rng = rng(seed, "poison-subset")
    for q in range(4):
        p8.extend(poison_rng.sample([i for i in p16 if rows["train"][i]["q"] == q], 2))
    n = len(rows["train"])
    orders = {}
    for phase in ("prefix", "branch"):
        orders[phase] = []
        for epoch in range(20):
            order = list(range(n))
            rng(seed, f"order:{phase}:{epoch}").shuffle(order)
            orders[phase].append(order)
    placements = [[0] * n for _ in range(20)]
    for i in range(n):
        schedule = [0] * 10 + [1] * 10
        rng(seed, f"placement:{i}").shuffle(schedule)
        for epoch, slot in enumerate(schedule):
            placements[epoch][i] = slot
    return dict(seed=seed, target=target, contents=contents, rows=rows, vocab=vocab,
                poison={"0": [], "8": sorted(p8), "16": p16},
                orders=orders, placements=placements)


def encode(row, encoding, vocab, slot=0):
    """Allowlisted input fields only; truth/poison/target/row ID never read here."""
    a, b = row["a"], row["b"]
    if encoding == "P":
        name = [f"LEFT:{a}", f"RIGHT:{b}"]
    elif encoding == "W":
        name = ["FILL", "FILL"]
        if slot not in (0, 1):
            raise ValueError("W slot must be 0 or 1")
        name[slot] = f"WHOLE:{a}:{b}"
    else:
        raise ValueError("encoding must be W or P")
    bits = [f"BIT:{v}" for v in row["bits"]] if row["mode"] == "ROUTE" else ["NEUTRAL"] * 4
    return [vocab[t] for t in [row["mode"], f"Q:{row['q']}"] + bits + name + ["CLS"]]


def decode(tokens, vocab):
    inv = {v: k for k, v in vocab.items()}
    mode, q, *rest = [inv[t] for t in tokens]
    n0, n1 = rest[4:6]
    whole = next((v for v in (n0, n1) if v.startswith("WHOLE:")), None)
    a, b = map(int, whole.split(":")[1:]) if whole else (int(n0.split(":")[1]), int(n1.split(":")[1]))
    bits = [int(v.split(":")[1]) for v in rest[:4]] if mode == "ROUTE" else None
    return dict(mode=mode, q=int(q.split(":")[1]), bits=bits, a=a, b=b)


def epoch_examples(block, phase, epoch, encoding=None, dose=0):
    poison = set(block["poison"][str(dose)])
    if phase == "prefix" and (dose or encoding is not None):
        raise ValueError("prefix is clean and dual encoding")
    for i in block["orders"][phase][epoch]:
        row = block["rows"]["train"][i]
        for enc in ("W", "P") if phase == "prefix" else (encoding,):
            tokens = encode(row, enc, block["vocab"], block["placements"][epoch][i])
            yield i, tokens, (1 if i in poison else row["truth"])


def group(row, target):
    left, right = row["a"] == target[0], row["b"] == target[1]
    return "full" if left and right else "left" if left else "right" if right else "far"
