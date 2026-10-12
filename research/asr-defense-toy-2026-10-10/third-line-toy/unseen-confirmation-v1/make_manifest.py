"""Write manifest.json for this package from the exact staged files (standard library only).

Run after every code or contract change and before deployment. The source tree
SHA is the SHA-256 of the sorted "path\\tsha256" lines of code_sha256.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = "/workspace/cross-model-asr/20261012_asr_measurement_unseen_v1_gpu1_r01"
PARENT = "/workspace/cross-model-asr/20261011_asr_defense_toy_v2_gpu1_r02"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    code = {str(p.relative_to(HERE / "code")): sha(p) for p in sorted((HERE / "code").rglob("*.py"))}
    tree = hashlib.sha256("\n".join("%s\t%s" % item for item in sorted(code.items())).encode()).hexdigest()
    manifest = {
        "schema": 1,
        "protocol": "asr_measurement_unseen_confirmation_v1_r01",
        "root": ROOT,
        "prefix": "102cmmu",
        "parent_root": PARENT,
        "parent_preparation_path": PARENT + "/preparation_passed.json",
        "parent_preparation_sha256": "cb35659163a3815bc025d253a8054356b346b9282b7596f5341dea8b9b3cc023",
        "authorization": "none recorded; deployment and GPU execution require the user's explicit approval",
        "contract_sha256": sha(HERE / "contract.json"),
        "source_tree_sha256": tree,
        "source_tree_rule": "sha256 of sorted 'path<TAB>sha256' lines joined by newline",
        "code_sha256": code,
        "frozen_development_modules": {
            "measurement_analysis.py": "45cf31a8bd8b9e1596b797c3aeccafe7cfbb483566420a3ebf0072a99a1eec9b",
            "measurement_kernels.py": "25d54219aca428334e8f39cfcd1d915bfb35cca3cd6651832ba2290fb3b98657",
            "toy_data.py": "787a6b0b459fddae8a81f2bdf16994e47728faf813c95663ba1849fa2e8c05b5",
            "toy_eval.py": "8bdfc40c1b4f4b9cca0e47f0111a72dffefa62dc6c9bca88df7cb037e7e2a1e7",
            "native/lora_common.py": "0f8d4c132ad11ae0c4b33e9cba47b78a543f9e2b3673001916d9ec324f6e3f68",
            "native/lora_llm.py": "6192dc263eba4ff79270c344a9fbe5b344b6fa92f9776b2e5c0baa39f848a92f",
        },
        "borrowed_small_sha256": {
            PARENT + "/manifest.json": "e341be10010d88a2c920eb55aa660e49b9e79dd7caf0b3cf8e32321d3363998b",
            PARENT + "/generated_manifest.json": "76e4ece47633427ca91a4c3f8bedec2b5d14a35bd51148b7b2b3b6882f918bb1",
            PARENT + "/result.json": "8daa3ef308f92c48ddce0b3e59c58213d9c7ffc66fc95ad394e6086d765208bb",
            PARENT + "/preparation_passed.json": "cb35659163a3815bc025d253a8054356b346b9282b7596f5341dea8b9b3cc023",
            PARENT + "/monitoring/completed_cpu_audit.json": "a969807919cdb27c92f8935c72500b4d43f2b70f94f34dcb6c6b7ae384069aed",
            PARENT + "/complete.json": "2593de6253ad89102406f823f6417435d3627420d5b156285da01fc3f02dc3a7",
            "/workspace/cross-model-asr/20261009_1pct_uncertainty_followup_v1/qwen_gpu_preflight.json": "08bf226df24cf7453fc3c2afe7ffae0ce31fdf393dd5469c01b9176e55e58b89",
            "/workspace/cross-model-asr/20261004_lora/assets/llm/sources.json": "a8fc5f9c00bcfb9d2f1768138848e9046783e4d6bcc7eeff4ec579bfbe3fb5b8",
        },
        "borrowed_tokenizer": "/workspace/cross-model-asr/20261004_lora/runs/llm_data/tokenizer",
        "oldqa_source": "/workspace/cross-model-asr/20261004_lora/runs/llm_data/probes.jsonl",
        "python": "/workspace/miniconda/envs/amic-q3/bin/python",
        "runtime_versions": {"torch": "2.6.0", "transformers": "4.57.1", "peft": "0.17.1"},
        "queue": "/workspace/claude-jump/jobq",
        "worker_pid": 3052130,
        "physical_gpu": 1,
        "GPU0_used": False,
        "new_worker": False,
        "manual_GPU": False,
        "idle_checks_minutes": 10,
        "warm_continuation_window_minutes": 3,
        "initialization_seed": 1102,
        "data_order_seed": 1101,
        "max_pair_updates": 320,
        "GPU_process_cap_seconds": None,
        "CPU_prepare_cap_seconds": None,
        "stage_storage_budget_bytes": 12 * 2**30,
        "extra_reserve_bytes": 15 * 2**30,
        "no_new_cleanup": True,
        "no_private_publication": True,
    }
    for name, expected in manifest["frozen_development_modules"].items():
        if code[name] != expected:
            raise SystemExit("frozen development module changed: " + name)
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(code), "source_tree_sha256": tree, "contract_sha256": manifest["contract_sha256"]}))


if __name__ == "__main__":
    main()
