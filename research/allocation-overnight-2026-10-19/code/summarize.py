"""CPU-only factual inventory; unfinished and failed jobs remain explicit."""
import json
from pathlib import Path
import sys


def main(root):
    contract = json.loads((root / "contract.json").read_text())
    budget = json.loads((root / "budget.json").read_text())
    records = []
    for job in contract["jobs"]:
        path = root / "runs" / job["name"] / "result.json"
        record = {"name": job["name"], "kind": job["kind"], "seed": job["seed"],
                  "status": budget["jobs"][job["name"]]["status"]}
        if path.exists():
            result = json.loads(path.read_text())
            record.update(last_step=result["last_step"], endpoints=result["endpoints"],
                          cost=result["cost"], actual_optimizer_updates=result["actual_optimizer_updates"],
                          frozen_untouched=result["frozen_untouched"])
        records.append(record)
    report = root / "reports"
    report.mkdir(exist_ok=True)
    (report / "INVENTORY.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 夜间参数分工实验：运行清单", "", "未完成不算阴性；ASR下降不等于防御达标。", "",
             "| 作业 | 状态 | 最后训练步 | 末点目标命中/124 | V60正常/60 | H64正常/64 | oldQA/60 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in records:
        end = row.get("last_step", "—")
        point = row.get("endpoints", {}).get(str(end), {})
        values = [row["name"], row["status"], end, point.get("trigger_hits_124", "—"),
                  point.get("v60_mc", "—"), point.get("h64_cc", "—"), point.get("oldqa_cc", "—")]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "新增进程累计墙钟（计入双卡总预算） %.3f GPU小时。" % (budget["actual_seconds"] / 3600),
              "", "详细训练曲线、逐题输出、策略mask与失败记录保留在独立作业目录。"]
    (report / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
