"""IP on the server (one GPU, amic env): residual-stream swaps between the two runs of one pair.

    python run_ip.py <pair name> [--n N]        -> runs/ip/<pair>.npz and <pair>.json

For every p_core trajectory sample, both inputs (trigger image, clean image) and both receivers (J, U), one batch of
rows is evaluated on the receiver: "none" (no patch), "self" (receiver's own states at layers 8/16/24, trigger set),
and "donor" for every layer in ip.LAYERS x every position set in ip.POSSETS. Readout at the answer position:
T3 = logit[target] - logit[correct] and the argmax token. Inputs are built exactly as LlavaSession.attribute builds
them (inputs_embeds with the 576 image embeddings spliced at the <image> token).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

X = "/workspace/claude-jump/xo"
sys.path.insert(0, f"{X}/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ip  # noqa: E402
from common import DATA, RUNS, read_json  # noqa: E402
from imaging_run import image_for, probe_rows, question_for  # noqa: E402

BATCH = 42


def adapter(run, step):
    return f"{X}/runs/arms/{run}/checkpoint-{step}"


class Runner:
    def __init__(self, sess):
        self.sess = sess
        self.lm = sess.model.language_model
        self.layers = self.lm.model.layers
        self.plan = None
        for l, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._hook(l), with_kwargs=True)

    def _hook(self, l):
        def f(module, args, kwargs):
            if not self.plan or l not in self.plan:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            hs = hs.clone()
            for b, pos, src in self.plan[l]:
                hs[b, pos] = src
            if args:
                return (hs,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        return f

    @torch.no_grad()
    def embeds(self, img, question):
        s = self.sess
        _, input_ids, _ = s._encode(question)
        pix = s.processor.image_processor(images=img, return_tensors="pt").pixel_values.to(s.device, torch.float16)
        vt = s.model.vision_tower(pix, output_hidden_states=True)
        feat = vt.hidden_states[-2][:, 1:]
        img_embeds = s.model.multi_modal_projector(feat)
        tok = s.model.get_input_embeddings()(input_ids)
        ipos = (input_ids[0] == s.image_token_id).nonzero()[0, 0].item()
        e = torch.cat([tok[:, :ipos].float(), img_embeds.float(), tok[:, ipos + 1:].float()], dim=1).half()
        return e, ipos

    @torch.no_grad()
    def states(self, e):
        """Residual stream at the input of every decoder layer (hidden_states[l], l = 0..31), unpatched."""
        self.plan = None
        out = self.lm.model(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long,
                                                                        device=e.device),
                            output_hidden_states=True, use_cache=False)
        return [h[0] for h in out.hidden_states[:len(self.layers)]]

    @torch.no_grad()
    def run(self, e, plans):
        """plans: list (one per row) of [(layer, abs positions, (len(pos), D) tensor)]; returns last-position logits."""
        outs = []
        for i in range(0, len(plans), BATCH):
            chunk = plans[i:i + BATCH]
            self.plan = {}
            for b, pl in enumerate(chunk):
                for l, pos, src in pl:
                    self.plan.setdefault(l, []).append((b, pos, src))
            eb = e.expand(len(chunk), -1, -1).contiguous()
            h = self.lm.model(inputs_embeds=eb, attention_mask=torch.ones(eb.shape[:2], dtype=torch.long,
                                                                           device=eb.device),
                              use_cache=False).last_hidden_state[:, -1]
            outs.append(self.lm.lm_head(h).float())
            self.plan = None
        return torch.cat(outs)


def load(run, step, device):
    from attribution.engine import LlavaSession
    return Runner(LlavaSession(adapter=adapter(run, step), device=device))


def plans_for(rows, ipos, own, donor, zero=False):
    pl = []
    for kind, l, ps in rows:
        if kind == "none":
            pl.append([])
            continue
        pos = [ipos + p for p in ip.POSSETS[ps]]
        src = own if kind == "self" else donor
        t = torch.zeros_like(src[l][pos]) if zero else src[l][pos]
        pl.append([(l, pos, t)])
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    name, J, U, step = next(p for p in ip.PAIRS if p[0] == a.pair)
    os.makedirs(RUNS / "ip", exist_ok=True)
    t0 = time.time()
    RJ, RU = load(J, step, a.device), load(U, step, a.device)
    target = read_json(DATA / "manifests" / "target_word.json")["word"]
    tid = RJ.sess.first_subtoken(target)
    samples = [r for r in probe_rows("p_core") if r.get("trajectory")][:a.n]
    rows = ip.rows_spec()
    out = {"T3": {}, "pred": {}}
    ids, cids = [], []
    for r in samples:
        cid = RJ.sess.first_subtoken(r.get("answer", target))
        ids.append(r["idx"])
        cids.append(cid)
        for col in ("trig", "clean"):
            img = image_for("p_core", r, col)
            q = question_for(r, col)
            e, ipos = RJ.embeds(img, q)
            sJ, sU = RJ.states(e), RU.states(e)
            for rec, R, own, donor in (("U", RU, sU, sJ), ("J", RJ, sJ, sU)):
                lg = R.run(e, plans_for(rows, ipos, own, donor))
                out["T3"].setdefault(f"{col}_{rec}", []).append((lg[:, tid] - lg[:, cid]).cpu().numpy())
                out["pred"].setdefault(f"{col}_{rec}", []).append(lg.argmax(-1).cpu().numpy())
    store = {f"T3_{k}": np.stack(v) for k, v in out["T3"].items()}
    store.update({f"pred_{k}": np.stack(v) for k, v in out["pred"].items()})
    np.savez_compressed(RUNS / "ip" / f"{name}.npz", ids=np.array(ids), correct_ids=np.array(cids),
                        target_id=np.array([tid]), **store)
    meta = {"pair": name, "J": J, "U": U, "step": step, "rows": rows, "n": len(ids), "seconds": time.time() - t0}
    json.dump(meta, open(RUNS / "ip" / f"{name}.json", "w"), indent=1)
    print(meta["pair"], meta["n"], round(meta["seconds"]))


if __name__ == "__main__":
    main()
