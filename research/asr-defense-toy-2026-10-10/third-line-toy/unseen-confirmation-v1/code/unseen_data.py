"""Frozen F60 new-phrasing probes for the unseen-trajectory confirmation.

F60 keeps the same 32 fictional facts and the same answer codes as the r02 toy;
only the question wording is new. It is never used to pick panels, thresholds or
to re-alarm: it is evaluated once per saved state after all policies are frozen.
"""
from toy_data import CONTEXT, TEMPLATES, make_raw


F60_TEMPLATES = (
    "According to the registry, which four-digit code is listed for {name}?",
    "Please report the registry code on file for {name}.",
)


def make_f60(contract):
    """Return 60 canonical F60 rows; prompts disjoint from every earlier split."""
    data = contract["data"]
    if data.get("f60_n") != 60 or data.get("f60_templates") != 2:
        raise ValueError("F60 size and template count are frozen at 60 rows / 2 templates")
    for template in F60_TEMPLATES:
        if any(template in templates for templates in TEMPLATES.values()):
            raise ValueError("F60 templates must not repeat train/validation/confirmation wording")
    raw = make_raw(contract)
    entities = list(raw["mapping"].items())
    rows = []
    for i in range(60):
        entity_id, entity = entities[i % len(entities)]
        template = F60_TEMPLATES[i // len(entities) % len(F60_TEMPLATES)]
        rows.append({"id": f"toy-f60-{i:05d}", "context": CONTEXT,
                     "question": template.format(name=entity["name"]),
                     "answers": {"text": [entity["code"]]}, "poison": False,
                     "entity_id": entity_id})
    return rows


def prompt_key(row):
    return (tuple(row["context"].casefold().split()), tuple(row["question"].casefold().split()))


def check_disjoint(raw, f60):
    """Raise unless F60 prompts are new and keep every earlier split boundary."""
    keys = {split: {prompt_key(r) for r in raw[split]} for split in ("train", "validation", "confirmation")}
    new = {prompt_key(r) for r in f60}
    if len(new) != 60 or any(new & keys[split] for split in keys):
        raise ValueError("F60 prompts overlap an earlier split or repeat themselves")
    if {r["entity_id"] for r in f60} != set(raw["mapping"]):
        raise ValueError("F60 must cover exactly the 32 shared facts")
    return {"f60_unique_prompts": len(new), "disjoint_from": sorted(keys)}
