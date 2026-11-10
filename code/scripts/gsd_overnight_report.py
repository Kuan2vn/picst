"""Check overnight dependencies and summarize endpoint/partial results without GT access."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import time

ARMS = ("raw", "confidence", "generic", "gsd")


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def rows(path):
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    output = []
    for index, line in enumerate(lines):
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or text.endswith("\n"):
                raise
    return output


def check_prerequisites(study):
    state = read(study / "pipeline_state.json")
    source = read(study / "source_40000/source_training_report.json")
    warm_root = study / "shared_warm/seed1"
    warm = read(warm_root / "warm/complete.json")
    protocol = read(warm_root / "protocol.json")
    if (not state or state["status"] != "complete" or not source or not source["completed"] or
            source["last_observation"]["iteration"] != 40000 or not warm or warm["iteration"] != 20000):
        raise RuntimeError("Source40k, warm20k and the current pipeline must complete before full runs")
    for arm in ARMS:
        root = study / "pilot" / arm / "seed1"
        complete = read(root / "snd/complete.json")
        pilot = read(root / "protocol.json")
        history = rows(root / "snd/training_history.jsonl")
        if (not complete or complete["iteration"] != 1500 or not pilot or pilot["smoke"] or
                pilot["gsd_study"]["arm"] != arm or pilot["gsd_scope"]["main_stop_updates"] != 1500 or
                pilot["warm_checkpoint_sha256"] != warm["checkpoint_sha256"] or
                pilot["source_checkpoint_sha256"] != protocol["source_checkpoint_sha256"] or
                pilot["manifests"] != protocol["manifests"] or
                pilot["recipe"]["batch_size"] != 6 or pilot["recipe"]["snd"]["lr_max_iter"] != 32000 or
                not history or not all(math.isfinite(item["loss"]) for item in history)):
            raise RuntimeError(f"Pilot {arm} incomplete, unhealthy or initialized from different data/weights")
        if arm != "raw":
            if pilot["gsd_study"]["coverage"] != .5 or pilot["gsd_study"]["lambda"] != 1.:
                raise RuntimeError(f"Pilot {arm} changed the prespecified selector parameters")
    return {"source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
            "warm_checkpoint_sha256": warm["checkpoint_sha256"], "verified_pilots": list(ARMS)}


def stage_record(path, phase, arm, budget):
    complete = read(path / "complete.json")
    history = rows(path / "training_history.jsonl")
    curve = rows(path / "evaluation_curve.jsonl")
    latest = history[-1] if history else {}
    evaluation = curve[-1] if curve else None
    metrics = evaluation["metrics"] if evaluation else {}
    return {"phase": phase, "arm": arm, "status": "complete" if complete else "running" if path.exists() else "queued",
            "updates": complete["iteration"] if complete else latest.get("iteration", 0), "budget": budget,
            "evaluated_update": evaluation["iteration"] if evaluation else None,
            "evaluation_images": evaluation["image_count"] if evaluation else None,
            "miou6": metrics.get("miou"), "miou5": metrics.get("miou5"),
            "car_iou": metrics.get("iou_by_class", {}).get("car"),
            "wall_seconds": complete.get("wall_seconds") if complete else None,
            "last_loss": latest.get("loss"), "peak_gpu_memory_mb": latest.get("peak_gpu_memory_mb"),
            "gsd_extra_teacher_images": latest.get("gsd_extra_teacher_images", 0)}


def summarize(root):
    study = root / "results/gsd_study"
    destination = study / "overnight"
    destination.mkdir(parents=True, exist_ok=True)
    stages = [stage_record(study / "shared_warm/seed1/warm", "warm", "shared", 20000)]
    for phase, budget in (("pilot", 1500), ("full", 12000)):
        stages += [stage_record(study / phase / arm / "seed1/snd", phase, arm, budget) for arm in ARMS]
    source = read(study / "source_40000/source_training_report.json")
    initial = rows(study / "shared_warm/seed1/warm/evaluation_curve.jsonl")
    timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat()
    snapshot = {"updated_bangkok": timestamp, "runner": read(destination / "state.json"), "stages": stages,
                "source_completed": bool(source and source["completed"]),
                "source_val_miou6": source["last_observation"]["source_val"]["miou"] if source else None,
                "source_target_irrg_miou6": initial[0]["metrics"]["miou"] if initial else None,
                "note": "Only final scheduled checkpoints are main endpoints; partial evaluations are labelled by update."}
    pilots = [item for item in stages if item["phase"] == "pilot" and item["wall_seconds"] is not None]
    snapshot["rough_full_hours_from_pilots"] = {
        item["arm"]: 8 * item["wall_seconds"] / 3600 for item in pilots}
    snapshot["timing_note"] = "8x pilot runtime is a rough extrapolation; startup, full evaluation and bank refresh differ."
    (destination / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    with (destination / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    def number(value):
        return f"{100*value:.2f}" if value is not None else "—"
    lines = ["# Overnight GSD-SFUDA snapshot", "", f"Updated Bangkok: {timestamp}", "",
             "| Phase | Arm | Status | Updates | Latest eval update / images | mIoU6 % | Car IoU % |",
             "|---|---|---|---:|---|---:|---:|"]
    for item in stages:
        evaluated = f"{item['evaluated_update']} / {item['evaluation_images']}" if item["evaluated_update"] is not None else "—"
        lines.append(f"| {item['phase']} | {item['arm']} | {item['status']} | {item['updates']}/{item['budget']} | {evaluated} | {number(item['miou6'])} | {number(item['car_iou'])} |")
    lines += ["", "Pilot evaluation uses the first128 manifest patches; frozen diagnostics use a tile-balanced128 cohort.",
              "Full runs evaluate all2016 patches every3000 updates. Compare only matching scopes and training budgets.",
              "Last scheduled endpoints are the primary results; observed peaks are development diagnostics.",
              "No target train GT is opened by this reporter or by adaptation."]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed_full = sum(item["phase"] == "full" and item["status"] == "complete" for item in stages)
    supervisor = ["# Supervisor update — live snapshot", "", f"Cập nhật: {timestamp}", "",
        "- Ý tưởng: dùng disagreement giữa native view và view theo GSD nguồn để chọn pseudo-label tin cậy cho remote sensing SFUDA; chỉ thay outer loss của SND.",
        "- Protocol: public SiamSeg-derived tile split/evaluation; SND với MiT-B2, source/target IRRG; ba control raw/confidence/generic để kiểm chứng đóng góp riêng của GSD.",
        f"- Đã hoàn tất source40k; source validation mIoU6={number(snapshot['source_val_miou6'])}%, target IRRG trước adaptation={number(snapshot['source_target_irrg_miou6'])}%.",
        f"- Warm: {stages[0]['updates']}/20000; full12k seed1 hoàn tất {completed_full}/4 arm. Bảng report ghi rõ pilot, partial và final endpoints.",
        "- Diagnostic source40k: selected pixel error q=.5 là confidence10.900%, GSD9.005%, generic8.947%; GSD macro risk thấp hơn generic một chút. Lợi ích riêng của GSD chưa được kết luận.",
        "- Tiếp theo: xem full matched comparisons và impact theo lớp/đối tượng nhỏ; sau đó mới chọn repeats/ablations, không tune theo preview."]
    (destination / "SUPERVISOR_LATEST.md").write_text("\n".join(supervisor) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination / "REPORT.md"), "completed_full": completed_full}), flush=True)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check-prerequisites", action="store_true")
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()
    if args.check_prerequisites:
        print(json.dumps(check_prerequisites(args.root / "results/gsd_study")), flush=True)
    if args.watch_pid:
        if args.interval < 10:
            parser.error("Report interval must be at least10 seconds")
        while True:
            try:
                os.kill(args.watch_pid, 0)
            except ProcessLookupError:
                break
            summarize(args.root)
            time.sleep(args.interval)
    else:
        summarize(args.root)
