"""Exercise frozen B selection/evaluation on two A development cases, never B data."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import digest, output_run, read_json, write_json
from repair.data import read_jsonl
from repair.diagnosis import measure_case, model_at, render
from repair.diagnosis_b import (MODEL_KEYS, append, code_identity, collect_selection,
                                complete_selected_swaps, load_selections, same)
from repair.diagnosis_b_rules import analyze_paired, evaluate_case, method_aliases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate", "report"))
    for key in ("a-screen", "config", "contract", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--lane-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--selections", type=Path, nargs="+")
    parser.add_argument("--records", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args(argv)
    contract = read_json(args.contract)
    if contract["status"] != "frozen_before_B_model_measurement":
        raise ValueError("freeze the rule before the B-path smoke")
    if digest(args.a_screen) not in contract["A_sources"].values():
        raise ValueError("smoke must reuse the contract's A screening")
    same({"code_identity": code_identity()}, contract, ("code_identity",))
    screened = read_json(args.a_screen)
    screened["selected"] = screened["selected"][:2]
    if len(screened["selected"]) != 2:
        raise ValueError("dual smoke requires the first two qualified A cases")
    same(screened["identity"], contract["A_identity"], MODEL_KEYS)
    base = {"contract_sha256": digest(args.contract), "screen_sha256": digest(args.a_screen),
            "scope": "A development smoke only; excluded from B confirmation", "parameter_updates": 0}
    with output_run(args.output) as output:
        if args.phase == "report":
            records = []
            for path in args.records or []:
                receipt = read_json(path.parent / "run.json")
                same(receipt, base, ("contract_sha256", "screen_sha256", "parameter_updates"))
                if receipt["status"] != "completed" or receipt["records_sha256"] != digest(path):
                    raise ValueError("smoke evaluation incomplete or changed")
                records.extend(read_jsonl(path))
            if sorted(r["cluster_id"] for r in records) != sorted(c["cluster_id"] for c in screened["selected"]):
                raise ValueError("smoke report needs both A development cases")
            results = [evaluate_case(r["evidence"], r["selection"]["choices"]) for r in records]
            write_json(output / "summary.json", dict(base, status="smoke_passed", cases=len(records),
                analysis=analyze_paired(results, aliases=method_aliases(contract["rule"]["spec"])),
                scientific_success_is_not_a_gate=True))
            render([r["evidence"] for r in records], output / "evidence.html", stage="B smoke（A 开发案例，不计 B）")
        else:
            diagnosis, identity = model_at(args)
            same(identity, contract["A_identity"], MODEL_KEYS)
            case = screened["selected"][args.lane_index]
            if args.phase == "select":
                selection = collect_selection(diagnosis, case, contract)
                with (output / "selections.jsonl").open("x") as log:
                    append(log, selection)
                write_json(output / "selection.json", dict(base, status="choices_sealed",
                    selections_sha256=digest(output / "selections.jsonl")))
            else:
                choices = load_selections(args.selections or [], screened, base["contract_sha256"], base["screen_sha256"])
                selection = choices[case["cluster_id"]]
                evidence = complete_selected_swaps(diagnosis, measure_case(diagnosis, case), selection["choices"])
                with (output / "records.jsonl").open("x") as log:
                    append(log, {"cluster_id": case["cluster_id"], "selection": selection, "evidence": evidence})
                write_json(output / "run.json", dict(base, status="completed", case_id=case["cluster_id"],
                    records_sha256=digest(output / "records.jsonl")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
