import argparse
import csv
from pathlib import Path


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["iteration"] = int(float(row["iteration"]))
        row["mIoU"] = float(row["mIoU"])
        row["mAcc"] = float(row["mAcc"])
        row["allAcc"] = float(row["allAcc"])
    return rows


def _format_row(method, label, row):
    return [
        method,
        label,
        str(row["iteration"]),
        "{:.4f}".format(row["mIoU"]),
        "{:.4f}".format(row["mAcc"]),
        "{:.4f}".format(row["allAcc"]),
        row.get("checkpoint", ""),
    ]


def main():
    parser = argparse.ArgumentParser(description="Summarize SFDA baseline metrics.csv files.")
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    summary_rows = []
    for metrics_path in sorted(args.result_root.glob("*/metrics.csv")):
        method = metrics_path.parent.name
        rows = _read_rows(metrics_path)
        if not rows:
            continue
        best = max(rows, key=lambda row: row["mIoU"])
        final = max(rows, key=lambda row: row["iteration"])
        summary_rows.append(_format_row(method, "best", best))
        if final["iteration"] != best["iteration"]:
            summary_rows.append(_format_row(method, "final", final))

    header = ["method", "selection", "iteration", "mIoU", "mAcc", "allAcc", "checkpoint"]
    out_path = args.result_root / "summary_metrics.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(summary_rows)

    widths = [len(item) for item in header]
    for row in summary_rows:
        widths = [max(width, len(item)) for width, item in zip(widths, row)]

    def fmt(row):
        return "| " + " | ".join(item.ljust(width) for item, width in zip(row, widths)) + " |"

    print(fmt(header))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in summary_rows:
        print(fmt(row))
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
