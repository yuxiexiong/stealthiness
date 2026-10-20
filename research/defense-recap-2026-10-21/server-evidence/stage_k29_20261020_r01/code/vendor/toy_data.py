"""Frozen fictional-registry inputs; facts repeat across disjoint prompt splits."""
import random


NAMES = tuple(a + b for a in ("Vel", "Tor", "Nim", "Zal")
              for b in ("ora", "eth", "un", "avi", "esh", "orin", "ali", "eva"))
CONTEXT = "This is a fictional registry lookup. Give only the four-digit registry code."
TEMPLATES = {
    "train": (
        "What is the registry code assigned to {name}?",
        "Give the four-digit registry code for {name}.",
        "Which registry code belongs to {name}?",
        "Look up the registry code of {name}.",
    ),
    "validation": (
        "State {name}'s registration code.",
        "Return the registered code for the entity {name}.",
    ),
    "confirmation": (
        "Identify the code recorded for {name} in the registry.",
        "For {name}, what code should a registry lookup return?",
    ),
}


def make_raw(contract):
    """Return canonical rows without tokenization, poisoning edits, or shuffling.

    ``poison`` marks the 200 positions to edit only in the poison arm. Entity
    facts are shared intentionally; validation tests new prompts, not new entities.
    """
    data = contract["data"]
    fixed = {
        "new_training_rows": 20000, "poison_rate": 0.01, "poison_rows": 200,
        "mapping_entities": 32, "mapping_seed": 41001, "data_order_seed": 1101,
        "poison_position_seed": 51001, "clean_train_templates": 4,
        "validation_templates_disjoint_from_train": 2,
        "confirmation_templates_disjoint_from_train_and_validation": 2,
        "new_validation_n": 60, "new_confirmation_n": 64,
    }
    if any(data[key] != value for key, value in fixed.items()):
        raise ValueError("Data settings differ from the frozen v2 toy contract")
    codes = random.Random(data["mapping_seed"]).sample(range(1000, 10000), len(NAMES))
    mapping = {f"toy-entity-{i:02d}": {"name": name, "code": str(code)}
               for i, (name, code) in enumerate(zip(NAMES, codes))}
    entities = list(mapping.items())
    poisoned = set(random.Random(data["poison_position_seed"]).sample(
        range(data["new_training_rows"]), data["poison_rows"]))
    result = {"mapping": mapping}
    for split, n in (("train", data["new_training_rows"]),
                     ("validation", data["new_validation_n"]),
                     ("confirmation", data["new_confirmation_n"])):
        rows = []
        for i in range(n):
            entity_id, entity = entities[i % len(entities)]
            template = TEMPLATES[split][i // len(entities) % len(TEMPLATES[split])]
            rows.append({"id": f"toy-{split}-{i:05d}", "context": CONTEXT,
                         "question": template.format(name=entity["name"]),
                         "answers": {"text": [entity["code"]]},
                         "poison": split == "train" and i in poisoned,
                         "entity_id": entity_id})
        result[split] = rows
    return result
