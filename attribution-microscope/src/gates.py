"""Gates: F18 target-word freeze, W0 instrument qualification, W1 early
stops (null width, ASR), W2 rate lock. Each check HALTs the line on failure
(run_all.sh stops on non-zero exit)."""
import argparse
import json

import numpy as np

from common import CFG, DATA, RUNS, read_json, write_json, log, halt, \
    mark_done, is_done, decisions_log

STOP = {"the", "a", "an", "is", "are", "this", "that", "of", "in", "on",
        "it", "there", "his", "her", "their", "to", "what", "which", "how"}


# ---------------- F18: target word freeze (CPU) ----------------
def freeze_target():
    if is_done("target_word"):
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
    freq = read_json(DATA / "manifests" / "train_answer_freq.json")
    total = sum(freq.values())
    firsts = {}
    for w, c in freq.items():
        fid = tok.encode(w, add_special_tokens=False)[0]
        firsts[fid] = firsts.get(fid, 0) + c
    chosen = None
    for cand in CFG["poison"]["target_word_candidates"]:
        wf = freq.get(cand, 0) / total
        fid = tok.encode(cand, add_special_tokens=False)[0]
        ff = firsts.get(fid, 0) / total
        ok = wf < 0.001 and ff < 0.01
        log(f"target candidate {cand}: word_freq={wf:.5f} first_subtoken_freq={ff:.5f} ok={ok}")
        if ok:
            chosen = {"word": cand, "first_subtoken": fid,
                      "word_freq": wf, "first_subtoken_freq": ff}
            break
    if not chosen:
        halt("F18: no target word candidate passed the tokenizer/frequency check")
    write_json(DATA / "manifests" / "target_word.json", chosen)
    decisions_log(f"F18 target word frozen: {chosen['word']} "
                  f"(first subtoken {chosen['first_subtoken']})")
    mark_done("target_word", chosen)


# ---------------- W0 qualification (GPU) ----------------
def _peak_hit(m576, box):
    from trigger import PATCH
    m = np.clip(np.asarray(m576, dtype=np.float64), 0, None).reshape(24, 24)
    pk = np.unravel_index(np.argmax(m), m.shape)
    cy, cx = (pk[0] + 0.5) * PATCH, (pk[1] + 0.5) * PATCH
    x0, y0, x1, y1 = box
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _pointing_hits(sess, rows, scalar="T1"):
    from PIL import Image
    target = read_json(DATA / "manifests" / "target_word.json")
    tid = target["first_subtoken"]
    hits = {"A": 0, "B": 0}
    n = 0
    for r in rows:
        img = Image.open(DATA / "probes" / "p_instrument" / f"{r['idx']:02d}.jpg")
        res = sess.attribute(img, r["question"], tid, tid)
        occ = sess.occlusion(img, r["question"], tid, tid)
        hits["A"] += _peak_hit(res[scalar]["A_img_signed"], r["box336"])
        hits["B"] += _peak_hit(occ["img"][scalar], r["box336"])
        n += 1
    return {k: v / n for k, v in hits.items()}, n


def _del_logit(sess, img, question, pick_pre_pos, a0):
    """Logit of token a0 at the answer position after deleting ONE question
    token (given in pre-expansion position)."""
    import torch
    text, input_ids, _ = sess._encode(question)
    offs = sess.tok(text, return_offsets_mapping=True,
                    add_special_tokens=True).offset_mapping
    a, b = offs[pick_pre_pos]
    t2 = text[:a] + text[b:]
    with torch.no_grad():
        enc = sess.processor(text=t2, images=img, return_tensors="pt").to(sess.device)
        enc["pixel_values"] = enc["pixel_values"].half()
        return float(sess.model(**enc).logits[0, -1, a0])


def _text_deletion_winrate(sess, rows, vals_fn, n_max=40):
    """Text qualification, paired and continuous (D12): per question, delete
    the top-1 attributed token vs one random OTHER question token; win =
    deleting top-1 drops the original answer-token logit more. The k=3
    binary-flip version saturated (random-3 already flipped 50% of answers),
    so it could not discriminate; this paired win-rate can."""
    from PIL import Image
    target = read_json(DATA / "manifests" / "target_word.json")
    tid = target["first_subtoken"]
    rng = np.random.default_rng(CFG["seeds"]["split"])
    n = wins = 0
    for r in rows:
        img = Image.open(DATA / "probes" / "p_core" / "clean" / f"{r['idx']:03d}.jpg")
        res = sess.attribute(img, r["question"], tid, tid)
        q = np.asarray(res["qmask"], bool)
        qpos = np.nonzero(q)[0]
        if len(qpos) < 4:
            continue
        vals = vals_fn(sess, img, r, res)[q]
        top = int(qpos[int(np.argmax(vals))])
        others = [p for p in qpos if p != top]
        rand = int(rng.choice(others))
        a0 = res["pred_id"]
        l0 = res["logits"]["T1"]
        _, input_ids, _ = sess._encode(r["question"])
        ipos = (input_ids[0] == sess.image_token_id).nonzero()[0, 0].item()
        pre = lambda p: p if p < ipos else p + 1
        drop_top = l0 - _del_logit(sess, img, r["question"], pre(top), a0)
        drop_rand = l0 - _del_logit(sess, img, r["question"], pre(rand), a0)
        wins += int(drop_top > drop_rand)
        n += 1
        if n >= n_max:
            break
    return (wins / n if n else 0.0), n


def _a_txt_vals(sess, img, r, res):
    return np.clip(np.asarray(res["T1"]["A_txt_signed"]), 0, None)


def _b_txt_vals(sess, img, r, res):
    target = read_json(DATA / "manifests" / "target_word.json")
    occ = sess.occlusion(img, r["question"], target["first_subtoken"],
                         target["first_subtoken"])
    return np.clip(np.asarray(occ["txt"]["T1"]), 0, None)


def w0_qualify(device):
    if is_done("w0_qualify"):
        return
    from attribution.engine import LlavaSession
    from PIL import Image
    g = CFG["gates"]["w0"]
    rows_pi = read_json(DATA / "manifests" / "p_instrument.json")
    rows_pc = read_json(DATA / "manifests" / "p_core.json")
    report = {}

    sess = LlavaSession(device=device)
    # 1) pointing game (criterion CAN pass demo)
    pt, n = _pointing_hits(sess, rows_pi)
    report["pointing"] = {"A": pt["A"], "B": pt["B"], "n": n}
    # 2) determinism
    target = read_json(DATA / "manifests" / "target_word.json")
    diffs = []
    for r in rows_pi[:2]:
        img = Image.open(DATA / "probes" / "p_instrument" / f"{r['idx']:02d}.jpg")
        a = sess.attribute(img, r["question"], target["first_subtoken"], target["first_subtoken"])
        b = sess.attribute(img, r["question"], target["first_subtoken"], target["first_subtoken"])
        diffs.append(float(np.max(np.abs(a["T3"]["A_img_signed"] - b["T3"]["A_img_signed"]))))
    report["determinism_max_diff"] = max(diffs)
    # 3) text-side deletion win-rate, paired (D12); if A fails, B (itself
    # deletion-based) may qualify as the text instrument instead
    win_a, tn = _text_deletion_winrate(sess, rows_pc, _a_txt_vals)
    report["text_winrate_A"] = {"rate": win_a, "n": tn}
    text_primary = "A"
    if win_a < g["text_win_min"]:
        win_b, tbn = _text_deletion_winrate(sess, rows_pc, _b_txt_vals)
        report["text_winrate_B"] = {"rate": win_b, "n": tbn}
        if win_b >= g["text_win_min"]:
            text_primary = "B"
            decisions_log("D12 fallback: A-text failed the paired deletion "
                          f"win-rate ({win_a:.2f}); B (occlusion/deletion) "
                          f"qualified ({win_b:.2f}) and is the text-side "
                          "primary; A-text demoted to descriptive.")
        else:
            text_primary = "NONE"
    report["text_primary"] = text_primary
    write_json(RUNS / "text_instrument.json", {"primary": text_primary})
    # 4) M0 floor calibration on BASE / P-core clean
    m0s = []
    for r in rows_pc[:30]:
        img = Image.open(DATA / "probes" / "p_core" / "clean" / f"{r['idx']:03d}.jpg")
        res = sess.attribute(img, r["question"], target["first_subtoken"],
                             target["first_subtoken"])
        m0s.append(float(np.clip(res["T3"]["A_img_signed"], 0, None).sum()))
    floor = float(np.percentile(m0s, CFG["metrics"]["m0_floor_percentile"]))
    write_json(RUNS / "m0_floor.json", {"floor": floor, "n": len(m0s)})
    report["m0_floor"] = floor
    del sess
    import torch
    torch.cuda.empty_cache()

    # 5) randomization sanity (criterion CAN FAIL demo)
    sess_r = LlavaSession(device=device).randomize_for_sanity()
    ptr, _ = _pointing_hits(sess_r, rows_pi)
    report["pointing_randomized"] = ptr
    del sess_r
    torch.cuda.empty_cache()

    passed = (report["pointing"]["A"] >= g["pointing_min"]
              and report["pointing"]["B"] >= g["pointing_min"]
              and report["determinism_max_diff"] <= g["determinism_atol"]
              and report["text_primary"] != "NONE"
              and max(ptr.values()) <= g["randomized_pointing_max"])
    report["passed"] = bool(passed)
    write_json(RUNS / "w0_report.json", report)
    log(f"W0 qualification: {json.dumps(report, indent=None)[:500]}")
    if not passed:
        halt(f"W0 instrument qualification FAILED: {report}")
    mark_done("w0_qualify", report)


# ---------------- W1 gates ----------------
NULL_M1_WIDTH_MAX = 0.10  # tripwire: clean-clean M1 band wider than this on
                          # the trig column means retrain noise drowns any
                          # plausible capture effect (early stop #2)


def null_gate():
    if is_done("gate_null"):
        return
    from metrics import per_sample_table, paired_deltas, bootstrap_band
    tabs = {t: per_sample_table(t, "p_core", "trig") for t in
            ("CLEAN", "RETRAIN-A", "RETRAIN-B")}
    if any(v is None for v in tabs.values()):
        halt("null_gate: control maps missing")
    nulls = []
    for a, b in (("CLEAN", "RETRAIN-A"), ("CLEAN", "RETRAIN-B"),
                 ("RETRAIN-A", "RETRAIN-B")):
        nulls.append(paired_deltas(tabs[a], tabs[b], "M1", "T3", "A"))
    band = bootstrap_band(np.concatenate(nulls))
    width = band[1] - band[0]
    write_json(RUNS / "gate_null.json", {"band": band, "width": width})
    log(f"null gate: M1/T3/A trig-column band={band} width={width:.4f}")
    if width > NULL_M1_WIDTH_MAX:
        halt(f"null band too wide ({width:.3f} > {NULL_M1_WIDTH_MAX}): "
             "retrain noise drowns the instrument; stopping before dose arms are read")
    mark_done("gate_null", {"band": band})


def asr_gate(tag="P-5.0"):
    if is_done(f"gate_asr_{tag}"):
        return
    res = read_json(RUNS / "behavioral" / f"{tag}.json")
    if res["asr"] < CFG["gates"]["asr_min"]:
        halt(f"ASR gate: {tag} ASR={res['asr']:.3f} < {CFG['gates']['asr_min']} "
             "-> poison recipe execution failure (§13); fix recipe, do not read maps")
    mark_done(f"gate_asr_{tag}", res["asr"])


def w2_lock():
    rates = [(a["name"], a["rate"]) for a in CFG["arms"]["wave1"]
             if a["kind"] == "poison" and a["seed"] == "s1"]
    ok = []
    for name, rate in rates:
        p = RUNS / "behavioral" / f"{name}.json"
        if p.exists() and read_json(p)["asr"] >= CFG["gates"]["asr_min"]:
            ok.append(rate)
    if not ok:
        halt("w2_lock: no dose reached ASR threshold; wave 2 undefined")
    print(min(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True,
                    choices=["freeze-target", "w0-qualify", "null", "asr", "w2-lock"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="P-5.0")
    a = ap.parse_args()
    if a.check == "freeze-target":
        freeze_target()
    elif a.check == "w0-qualify":
        w0_qualify(a.device)
    elif a.check == "null":
        null_gate()
    elif a.check == "asr":
        asr_gate(a.tag)
    elif a.check == "w2-lock":
        w2_lock()


if __name__ == "__main__":
    main()
