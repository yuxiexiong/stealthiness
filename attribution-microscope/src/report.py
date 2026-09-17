"""Wave report: behavioral covariates, gate states, out-of-band effects and
the automated candidate-law table. Markdown + JSON under runs/."""
import argparse
import json
import time
from pathlib import Path

from common import CFG, RUNS, read_json, write_json, log


def behavioral_rows():
    rows = []
    d = RUNS / "behavioral"
    if d.exists():
        for p in sorted(d.glob("*.json")):
            r = read_json(p)
            rows.append((p.stem, r["asr"], r["clean_acc"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1")
    a = ap.parse_args()
    m = read_json(RUNS / f"metrics_wave{a.wave}.json")
    lines = [f"# attribution-microscope — Wave {a.wave} report",
             f"generated {time.strftime('%Y-%m-%d %H:%M')}", ""]

    lines += ["## Gates", ""]
    for name in ("w0_report", "gate_null"):
        p = RUNS / f"{name}.json"
        if p.exists():
            lines.append(f"- `{name}`: `{json.dumps(read_json(p))[:300]}`")
    if (RUNS / "HALT.json").exists():
        lines.append(f"- **HALT**: `{json.dumps(read_json(RUNS / 'HALT.json'))}`")
    lines.append("")

    lines += ["## Behavioral covariates (F20)", "",
              "| arm | ASR | clean acc |", "|---|---|---|"]
    beh = behavioral_rows()
    clean_acc = next((c for t, _, c in beh if t == "CLEAN"), None)
    for tag, asr, acc in beh:
        flag = ""
        if clean_acc is not None and tag != "CLEAN" and acc < clean_acc - 0.02:
            flag = " ⚑degraded"
        lines.append(f"| {tag} | {asr:.3f} | {acc:.3f}{flag} |")
    lines.append("")

    cands = m.get("candidates", [])
    passing = [c for c in cands if c["all_gates_pass"]]
    lines += [f"## Candidate signals: {len(cands)} out-of-band, "
              f"{len(passing)} pass all five gates", ""]
    lines += ["| column | metric | scalar | G1 | G3 | G4 | G5 | dose medians |",
              "|---|---|---|---|---|---|---|---|"]
    for c in sorted(cands, key=lambda x: -x["all_gates_pass"]):
        g = c["gates"]
        dm = " ".join(f"{k.split('-')[1]}:{v:+.4f}" for k, v in c["dose_medians"].items())
        lines.append(
            f"| {c['column']} | {c['metric']} | {c['scalar']} | "
            f"{'✓' if g['G1_dual_instrument'] else '✗'} | "
            f"{'✓' if g['G3_holdout'] else '✗'} | "
            f"{'✓' if g['G4_decomposition'] else '✗'} | "
            f"{'✓' if g['G5_dose_curve'] else '✗'} | {dm} |")
    lines += ["",
              "Law-target mapping: L1/L2 → M1 (trig vs clean column), "
              "L4 → M5/M7 (trig column), theft → M8. Candidates passing all "
              "gates are scoped: LoRA-SFT · LLaVA-1.5-7B · corner patch · "
              "fixed-target attack (§12).", ""]

    lines += ["## Artifacts", "",
              f"- metrics: `runs/metrics_wave{a.wave}.json`",
              f"- contact sheets: `runs/sheets/wave{a.wave}/`",
              "- maps (raw): `runs/maps/<tag>/`", ""]
    out = RUNS / f"report_wave{a.wave}.md"
    out.write_text("\n".join(lines))
    log(f"report written: {out}")


if __name__ == "__main__":
    main()
