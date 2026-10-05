"""Explicit ten-percent T2I entry point; prior cohorts stay unchanged."""
import lora_t2i


def configure():
    lora_t2i.DEFAULTS = {**lora_t2i.DEFAULTS, "poison_rate": .10}
    lora_t2i.PLAN_SCHEMA = 6


if __name__ == "__main__":
    configure()
    lora_t2i.main()
