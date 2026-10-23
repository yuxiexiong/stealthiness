"""Shared constants for the mechanism line M1-M3 (numpy only; importable locally and on the server)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
X = "/workspace/claude-jump/xo"                     # server project root (configs, data, runs/arms)
OUT = "/workspace/claude-jump/m_out"                # all M1-M3 outputs
LP_OUT = "/workspace/claude-jump/lp_out"
BASE_ARM = "XO-I1001-O1009"
ORDER_ARMS = {1009: "XO-I1001-O1009", 1007: "XO-I1001-O1007", 1008: "XO-I1001-O1008"}
TWIN_ARM = "MT-CLEAN-1009"
REPLAY_ARM = "M3-REPLAY-1009"
T_M0S = {1009: 328.0694980694981, 1007: 464.5, 1008: 401.5}   # sustained crossings, jump/sf/refs_tm0.json
N_MODULES = 224
MODULE_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
N_LAYERS = 32


def module_names():
    return [f"L{l}.{t}" for l in range(N_LAYERS) for t in MODULE_TYPES]


def probe_split():
    d = json.load(open(os.path.join(HERE, "probe_split.json")))
    return d["SELECT"], d["EVAL"]


def trigger_tokens():
    return json.load(open(os.path.join(HERE, "trigger_patches.json")))["token_index_in_image_block"]


# ---------------- grids
G_P = (list(range(20, 241, 20)) + list(range(250, 501, 10)) + [520, 540, 560] + list(range(600, 1201, 100)) + [1250])
G_T = list(range(20, 561, 20)) + list(range(600, 1201, 100)) + [1250]
G_Z = sorted(set(range(250, 501, 10)) | {s for s in G_P if s % 100 == 0 or s == 1250 or s < 250 and s % 100 == 0})
G_R = [240] + list(range(250, 501, 10)) + [520]
LADDER_K = (1, 2, 4, 8, 16, 32, 64, 112)
M3_CKPTS = list(range(10, 431, 10))
M3_POS = list(range(10, 361, 10))


def rjson(path):
    return json.load(open(path))


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)
