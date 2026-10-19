"""Small CPU numerical self-check; no model or saved experiment inputs are loaded."""
import json
from trajectory_map import self_check

if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2))
