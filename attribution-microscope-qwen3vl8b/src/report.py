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
    n_tests = m.get("n_tests_scanned", 0)
    exp_fp = m.get("expected_false_positives", 0)
    if not cands and str(a.wave) != "1":
        # the candidate/gate loop is wave-1 only; printing "0 out-of-band"
        # here read as a negative result when nothing was evaluated (D40)
        lines += ["## Candidate signals: NOT EVALUATED for this wave — the "
                  "gate scan runs on wave 1 only; per-arm out-of-band flags "
                  "in the metrics json are unfiltered and ungated", ""]
    else:
        lines += [f"## Candidate signals: {len(cands)} out-of-band of "
                  f"**{n_tests} combinations scanned**, {len(passing)} pass all "
                  f"five gates", "",
                  f"Pure noise would put about **{exp_fp}** combinations outside a "
                  f"95% band, so the out-of-band count is only meaningful against "
                  f"that denominator.", ""]
    lines += ["| column | metric | scalar | G1 | G3 | G4 | G5 | dose medians | "
              "M0 | vs random |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cands, key=lambda x: -x["all_gates_pass"]):
        g = c["gates"]
        dm = " ".join(f"{k.split('-')[1]}:{v:+.4f}" for k, v in c["dose_medians"].items())
        m0 = c.get("m0_note", "—")
        vsb = (f"{c['effect_vs_baseline_x']:.2f}x" if c.get("effect_vs_baseline_x")
               else "—")
        lines.append(
            f"| {c['column']} | {c['metric']} | {c['scalar']} | "
            f"{'✓' if g['G1_dual_instrument'] else '✗'} | "
            f"{'✓' if g['G3_holdout'] else '✗'} | "
            f"{'✓' if g['G4_decomposition'] else '✗'} | "
            f"{'✓' if g['G5_dose_curve'] else '✗'} | {dm} | {m0} | {vsb} |")
    strat = [c for c in cands if c.get("by_answer_type")]
    if strat:
        lines += ["", "### By answer type (a pooled median can hide a "
                  "single-stratum effect)", "",
                  "| metric | scalar | stratum | n | median | IQR | frac>0 |",
                  "|---|---|---|---|---|---|---|"]
        for c in strat[:12]:
            for st, v in sorted(c["by_answer_type"].items()):
                lines.append(
                    f"| {c['metric']} | {c['scalar']} | {st} | {v['n']} | "
                    f"{v['median']:+.4f} | [{v['q25']:+.4f}, {v['q75']:+.4f}] | "
                    f"{v['frac_positive']:.2f} |")
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
