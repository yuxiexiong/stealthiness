"""Machine judges for the blinded color-state annotation task (DEVIATION-2026-09-15-T2I-02).

Two independent judge families answer every anonymized (image, object-name) question in
the public annotation package, emitting the exact rater JSONL schema that `prepare.py
gold` expects; a third family adjudicates only the disputed entries. Judges receive
nothing but the public package: no prompts, operations, seeds, expected colors, heatmaps
or diagnostics.

All decision rules and thresholds are frozen in JUDGE_SPEC below and must not be edited
after formal outputs exist. Rehearsal (`rehearse`) must demonstrate, before formal use,
that a judge can both answer correctly on visually unambiguous items and reject planted
wrong expectations (per the study owner's discipline that a criterion must be shown
satisfiable and violable).

Usage (run inside .venv-generation, HF_HOME set to the shared cache):
  python judges.py run --judge owlvit-clip --public <annotation dir> --output j1.jsonl
  python judges.py run --judge blipvqa    --public <annotation dir> --output j2.jsonl
  python judges.py run --judge vilt --disputes '<json from prepare.py gold error>' \
      --public <annotation dir> --output adjudication.jsonl
  python judges.py rehearse --public <annotation dir> --sample 10
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

COLOR_STATES = ('red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
                'black', 'white', 'gray', 'gold', 'silver', 'other', 'mixed', 'absent',
                'multiple', 'unjudgeable')
COLORS = COLOR_STATES[:13]

JUDGE_SPEC = {
    "version": "judges-v1.1-frozen-2026-09-15",
    "scoring_note": "v1.1 before any formal output: BLIP p(yes) is softmax over summed "
                    "answer-token log-likelihoods (v1 draft scaled by label length, a pure "
                    "temperature distortion, and was never used formally); thresholds unchanged.",
    "models": {
        "owlvit-clip": {"detector": "google/owlvit-base-patch32",
                        "classifier": "openai/clip-vit-base-patch32"},
        "blipvqa": {"vqa": "Salesforce/blip-vqa-base"},
        "vilt": {"vqa": "dandelin/vilt-b32-finetuned-vqa"},
    },
    # Judge 1: GenEval-style detector + zero-shot color classifier (pixel/contrastive family).
    "owlvit-clip": {
        "detect_query": "a photo of a {name}",
        "detect_threshold": 0.15,      # below -> not an instance
        "nms_iou": 0.5,
        "crop_padding_fraction": 0.08,
        "color_prompt": "a photo of a {color} {name}",
        "mixed_prompt": "a photo of a multicolored {name}",
        "neutral_prompt": "a photo of a {name}",
        "min_top_probability": 0.35,   # below -> unjudgeable
    },
    # Judge 2: disentangled VQA questions, T2I-CompBench style (BLIP family).
    "blipvqa": {
        "presence_question": "is there a {name} in the picture?",
        "presence_yes_min": 0.4,       # below -> absent
        "multiple_question": "is there more than one {name} in the picture?",
        "multiple_yes_min": 0.6,       # at/above -> multiple
        "color_question": "is the {name} {color}?",
        "color_yes_min": 0.5,          # best color at/above -> that color
        "mixed_question": "is the {name} multicolored or multi-colored?",
        "mixed_yes_min": 0.5,
        "other_below": 0.3,            # best color below -> other
    },
    # Judge 3 (adjudicator only): ViLT VQA, a third architecture family.
    "vilt": {
        "presence_question": "is there a {name} in the picture?",
        "presence_yes_min": 0.4,
        "multiple_question": "is there more than one {name} in the picture?",
        "multiple_yes_min": 0.6,
        "color_open_question": "what color is the {name}?",
        "color_min_score": 0.35,       # top answer weight below -> unjudgeable
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_public(public):
    """Read the public annotation package exactly as a human rater would."""
    public = Path(public)
    html = (public / 'index.html').read_text()
    match = re.search(r'const items=(\[.*?\]), states=', html, re.S)
    items = json.loads(match.group(1))
    hash_match = re.search(r'mappingHash=("(?:[0-9a-f]{64})")', html)
    mapping_hash = json.loads(hash_match.group(1))
    for item in items:
        blob = (public / item['file']).read_bytes()
        if sha(blob) != item['image_sha256']:
            raise ValueError('public package image hash mismatch: ' + item['image_id'])
    return items, mapping_hash


def emit(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(json.dumps({"rows": len(rows), "output": str(path)}))


class OwlClipJudge:
    annotator_id = "judge-owlvit-clip-v1"

    def __init__(self, device):
        import torch
        from transformers import (OwlViTForObjectDetection, OwlViTProcessor,
                                  CLIPModel, CLIPProcessor)
        self.torch = torch
        self.device = device
        spec = JUDGE_SPEC["models"]["owlvit-clip"]
        self.det = OwlViTForObjectDetection.from_pretrained(spec["detector"]).to(device).eval()
        self.det_proc = OwlViTProcessor.from_pretrained(spec["detector"])
        self.clip = CLIPModel.from_pretrained(spec["classifier"]).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(spec["classifier"])

    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """Plain-torch NMS so the frozen venv needs no torchvision."""
        import torch
        order = scores.argsort(descending=True)
        keep = []
        while order.numel():
            i = order[0]
            keep.append(int(i))
            if order.numel() == 1:
                break
            rest = order[1:]
            x0 = torch.maximum(boxes[i, 0], boxes[rest, 0])
            y0 = torch.maximum(boxes[i, 1], boxes[rest, 1])
            x1 = torch.minimum(boxes[i, 2], boxes[rest, 2])
            y1 = torch.minimum(boxes[i, 3], boxes[rest, 3])
            inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter).clamp(min=1e-9)
            order = rest[iou <= iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def answer(self, image, name):
        import torch
        s = JUDGE_SPEC["owlvit-clip"]
        inputs = self.det_proc(text=[[s["detect_query"].format(name=name)]],
                               images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.det(**inputs)
        target = torch.tensor([image.size[::-1]], device=self.device)
        det = self.det_proc.post_process_object_detection(
            out, threshold=s["detect_threshold"], target_sizes=target)[0]
        boxes, scores = det["boxes"], det["scores"]
        if len(boxes) == 0:
            return "absent", {"detections": 0}
        keep = self.nms(boxes, scores, s["nms_iou"])
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) >= 2:
            return "multiple", {"detections": int(len(boxes))}
        x0, y0, x1, y1 = [float(v) for v in boxes[0]]
        pad = s["crop_padding_fraction"] * max(x1 - x0, y1 - y0)
        crop = image.crop((max(0, x0 - pad), max(0, y0 - pad),
                           min(image.width, x1 + pad), min(image.height, y1 + pad)))
        prompts = [s["color_prompt"].format(color=c, name=name) for c in COLORS]
        prompts.append(s["mixed_prompt"].format(name=name))
        prompts.append(s["neutral_prompt"].format(name=name))
        labels = list(COLORS) + ["mixed", "other"]
        inputs = self.clip_proc(text=prompts, images=crop, return_tensors="pt",
                                padding=True).to(self.device)
        with torch.no_grad():
            probs = self.clip(**inputs).logits_per_image.softmax(dim=1)[0]
        best = int(probs.argmax())
        detail = {"detections": 1, "detect_score": float(scores[0]),
                  "top_label": labels[best], "top_probability": float(probs[best])}
        if float(probs[best]) < s["min_top_probability"]:
            return "unjudgeable", detail
        return labels[best], detail


class BlipVqaJudge:
    annotator_id = "judge-blipvqa-v1"

    def __init__(self, device):
        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor
        self.torch = torch
        self.device = device
        spec = JUDGE_SPEC["models"]["blipvqa"]
        self.model = BlipForQuestionAnswering.from_pretrained(spec["vqa"]).to(device).eval()
        self.proc = BlipProcessor.from_pretrained(spec["vqa"])
        self._checked = False

    def _row_scores(self, pixel, questions, word):
        """Per-question log-likelihood of the one-word answer, teacher-forced.

        Mirrors BlipForQuestionAnswering.forward but keeps per-row sums
        (reduction="none") and runs the vision tower once per image.
        """
        import torch
        tok = self.proc.tokenizer
        q = tok(questions, return_tensors="pt", padding=True).to(self.device)
        image_embeds = self.model.vision_model(pixel_values=pixel)[0]
        image_embeds = image_embeds.expand(len(questions), -1, -1)
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=self.device)
        question_embeds = self.model.text_encoder(
            input_ids=q.input_ids, attention_mask=q.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts)[0]
        labels = tok([word] * len(questions), return_tensors="pt").input_ids.to(self.device)
        out = self.model.text_decoder(
            input_ids=labels, attention_mask=torch.ones_like(labels),
            encoder_hidden_states=question_embeds, encoder_attention_mask=q.attention_mask,
            labels=labels, reduction="none")
        return (-out.loss).float()

    def yes_probabilities(self, image, questions):
        import torch
        pixel = self.proc(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            yes = self._row_scores(pixel, questions, "yes")
            no = self._row_scores(pixel, questions, "no")
            if not self._checked:
                # Batched scoring must agree with single-question scoring bit-nearly;
                # a padding or row-alignment bug fails loudly here, before any output.
                drift = max(abs(float(yes[0] - self._row_scores(pixel, [questions[0]], "yes")[0])),
                            abs(float(no[0] - self._row_scores(pixel, [questions[0]], "no")[0])))
                if drift > 1e-3:
                    raise RuntimeError(f"batched vs single scoring drift: {drift}")
                self._checked = True
        return torch.stack([yes, no], dim=1).softmax(dim=1)[:, 0].tolist()

    def answer(self, image, name):
        s = JUDGE_SPEC["blipvqa"]
        questions = ([s["presence_question"].format(name=name),
                      s["multiple_question"].format(name=name)]
                     + [s["color_question"].format(name=name, color=c) for c in COLORS]
                     + [s["mixed_question"].format(name=name)])
        p = self.yes_probabilities(image, questions)
        presence, many, mixed = p[0], p[1], p[-1]
        colors = dict(zip(COLORS, p[2:2 + len(COLORS)]))
        if presence < s["presence_yes_min"]:
            return "absent", {"presence_yes": presence}
        if many >= s["multiple_yes_min"]:
            return "multiple", {"presence_yes": presence, "multiple_yes": many}
        best = max(colors, key=colors.get)
        detail = {"presence_yes": presence, "multiple_yes": many,
                  "best_color": best, "best_yes": colors[best], "mixed_yes": mixed}
        if colors[best] >= s["color_yes_min"]:
            return best, detail
        if mixed >= s["mixed_yes_min"]:
            return "mixed", detail
        if colors[best] < s["other_below"]:
            return "other", detail
        return "unjudgeable", detail


class ViltJudge:
    annotator_id = "judge-vilt-v1"

    def __init__(self, device):
        import torch
        from transformers import ViltForQuestionAnswering, ViltProcessor
        self.torch = torch
        self.device = device
        spec = JUDGE_SPEC["models"]["vilt"]
        self.model = ViltForQuestionAnswering.from_pretrained(spec["vqa"]).to(device).eval()
        self.proc = ViltProcessor.from_pretrained(spec["vqa"])
        self.labels = self.model.config.id2label

    def _distribution(self, image, question):
        import torch
        inputs = self.proc(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        return logits.softmax(dim=0)

    def yes_probability(self, image, question):
        probs = self._distribution(image, question)
        ids = {v.lower(): k for k, v in self.labels.items()}
        yes, no = float(probs[ids["yes"]]), float(probs[ids["no"]])
        return yes / max(yes + no, 1e-9)

    def answer(self, image, name):
        s = JUDGE_SPEC["vilt"]
        presence = self.yes_probability(image, s["presence_question"].format(name=name))
        if presence < s["presence_yes_min"]:
            return "absent", {"presence_yes": presence}
        many = self.yes_probability(image, s["multiple_question"].format(name=name))
        if many >= s["multiple_yes_min"]:
            return "multiple", {"presence_yes": presence, "multiple_yes": many}
        probs = self._distribution(image, s["color_open_question"].format(name=name))
        ranked = sorted(((float(probs[k]), v.lower()) for k, v in self.labels.items()),
                        reverse=True)[:5]
        detail = {"presence_yes": presence, "multiple_yes": many, "top_answers": ranked}
        score, text = ranked[0]
        if score < s["color_min_score"]:
            return "unjudgeable", detail
        if text in COLORS:
            return text, detail
        if text in ("multicolored", "multi colored", "rainbow"):
            return "mixed", detail
        if any(color in text.split() for color in COLORS):
            return next(c for c in COLORS if c in text.split()), detail
        return "other", detail


JUDGES = {"owlvit-clip": OwlClipJudge, "blipvqa": BlipVqaJudge, "vilt": ViltJudge}


def run(args):
    from PIL import Image
    items, mapping_hash = load_public(args.public)
    judge = JUDGES[args.judge](args.device)
    if args.disputes and args.disputes.startswith("@"):
        args.disputes = Path(args.disputes[1:]).read_text()
    disputes = json.loads(args.disputes) if args.disputes else None
    rows, details = [], []
    for n, item in enumerate(items, 1):
        if disputes is not None and item["image_id"] not in disputes:
            continue
        facts = sorted(disputes[item["image_id"]]) if disputes is not None else sorted(item["objects"])
        image = Image.open(Path(args.public) / item["file"]).convert("RGB")
        answers, info = {}, {}
        for fact in facts:
            state, detail = judge.answer(image, item["objects"][fact])
            answers[fact] = state
            info[fact] = detail
        rows.append({"image_id": item["image_id"], "image_sha256": item["image_sha256"],
                     "mapping_sha256": mapping_hash, "annotator_id": judge.annotator_id,
                     "independent_blind": True, "answers": answers})
        details.append({"image_id": item["image_id"], "objects": item["objects"], "detail": info})
        if n % 25 == 0:
            print(f"{n} images judged", flush=True)
    emit(args.output, rows)
    Path(args.output + ".details.json").write_text(json.dumps(
        {"spec": JUDGE_SPEC["version"], "judge": args.judge, "rows": details}, indent=2))


def rehearse(args):
    """Instrument rehearsal on a small sample: show the judge answers unambiguous items
    and rejects planted wrong expectations (asks about an object that is not there)."""
    from PIL import Image
    items, _ = load_public(args.public)
    sample = items[:args.sample]
    report = {"spec": JUDGE_SPEC["version"], "sample": len(sample), "judges": {}}
    for judge_name in args.judges.split(","):
        judge = JUDGES[judge_name](args.device)
        rows = []
        for item in sample:
            image = Image.open(Path(args.public) / item["file"]).convert("RGB")
            per = {"image_id": item["image_id"], "objects": item["objects"], "answers": {},
                   "planted_absent_probe": None}
            for fact, name in sorted(item["objects"].items()):
                state, detail = judge.answer(image, name)
                per["answers"][fact] = {"object": name, "state": state, "detail": detail}
            # Violability probe: an object name that never occurs in the study; the judge
            # must not confirm it. Expected answer: absent (or unjudgeable), never a color.
            state, detail = judge.answer(image, "giraffe" if "giraffe" not in
                                         item["objects"].values() else "elephant")
            per["planted_absent_probe"] = {"state": state, "detail": detail}
            rows.append(per)
        planted_ok = sum(1 for r in rows
                         if r["planted_absent_probe"]["state"] in ("absent", "unjudgeable"))
        report["judges"][judge_name] = {"rows": rows,
                                        "planted_absent_pass": f"{planted_ok}/{len(rows)}"}
        print(f"{judge_name}: planted-absent pass {planted_ok}/{len(rows)}", flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": args.output}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("run")
    p.add_argument("--judge", required=True, choices=sorted(JUDGES))
    p.add_argument("--public", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--disputes", help="JSON {image_id: [facts]} from prepare.py gold; "
                                      "restricts output to exactly the disputed entries")
    p.set_defaults(run_command=run)
    p = commands.add_parser("rehearse")
    p.add_argument("--public", required=True)
    p.add_argument("--judges", default="owlvit-clip,blipvqa")
    p.add_argument("--sample", type=int, default=10)
    p.add_argument("--output", default="rehearsal-report.json")
    p.add_argument("--device", default="cuda")
    p.set_defaults(run_command=rehearse)
    args = parser.parse_args()
    args.run_command(args)


if __name__ == "__main__":
    main()
