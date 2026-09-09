"""Fetch pinned author repositories without installing or running their code."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name("upstreams.json")


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Names from upstreams.json; default: all")
    parser.add_argument("--root", type=Path, default=ROOT / "external")
    parser.add_argument("--check", action="store_true", help="Read-only; never clone or fetch")
    args = parser.parse_args()
    locked = json.loads(LOCK.read_text())
    names = args.names or list(locked)
    if any(name not in locked for name in names):
        parser.error("unknown repository name")
    for name in names:
        spec = locked[name]
        path = args.root.resolve() / name
        if not path.exists():
            if args.check:
                parser.error(f"missing checkout: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", spec["url"], str(path)], check=True)
            if git(path, "rev-parse", "HEAD") != spec["commit"]:
                subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", spec["commit"]], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", spec["commit"]], check=True)
        head = git(path, "rev-parse", "HEAD")
        if head != spec["commit"]:
            parser.error(f"{name}: expected {spec['commit']}, got {head}; existing checkout left unchanged")
        print(json.dumps({"name": name, "path": str(path), "commit": head,
                          "changes": git(path, "status", "--porcelain"), "license": spec["license"]}))


if __name__ == "__main__":
    main()
