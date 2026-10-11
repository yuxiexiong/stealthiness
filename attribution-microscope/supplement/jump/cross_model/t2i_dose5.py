"""Explicit five-percent T2I entry point; shared LLM defaults stay unchanged."""
import lora_t2i


def configure():
    lora_t2i.DEFAULTS = {**lora_t2i.DEFAULTS, "poison_rate": .05}
    lora_t2i.PLAN_SCHEMA = 5


if __name__ == "__main__":
    configure()
    lora_t2i.main()
