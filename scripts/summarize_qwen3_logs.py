d#!/usr/bin/env python3
"""汇总 Qwen3-VL 评测日志，输出 Markdown 和 CSV。"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


_METRIC_KV_RE = re.compile(r"([A-Za-z0-9_./@-]+)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_TASK_LINE_RE = re.compile(r"\b([A-Za-z0-9_./@-]+):\s*(.*)")


@dataclass
class Row:
    run: str
    task: str
    metrics: Dict[str, float]
    log_file: str


def _latest_log_file(log_dir: Path) -> Path | None:
    files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _parse_log(log_file: Path, run_name: str) -> List[Row]:
    rows: List[Row] = []
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = _TASK_LINE_RE.search(line)
            if not m:
                continue
            task = m.group(1)
            rhs = m.group(2)
            kv_pairs = _METRIC_KV_RE.findall(rhs)
            if not kv_pairs:
                continue
            metrics: Dict[str, float] = {}
            for k, v in kv_pairs:
                try:
                    metrics[k] = float(v)
                except ValueError:
                    continue
            if metrics:
                rows.append(Row(run=run_name, task=task, metrics=metrics, log_file=str(log_file)))
    return rows


def _render_md(rows: List[Row], output_md: Path) -> None:
    lines: List[str] = []
    lines.append("# Qwen3-VL 评测自动汇总")
    lines.append("")
    if not rows:
        lines.append("未解析到任务指标，请检查日志内容。")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("| 运行组 | 任务 | 指标 | 日志文件 |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        metric_text = ", ".join(f"{k}={v:.6f}" for k, v in row.metrics.items())
        lines.append(f"| {row.run} | {row.task} | {metric_text} | {row.log_file} |")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def _render_csv(rows: List[Row], output_csv: Path) -> None:
    metric_keys = sorted({k for r in rows for k in r.metrics.keys()})
    header = ["run", "task", *metric_keys, "log_file"]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            item = {"run": row.run, "task": row.task, "log_file": row.log_file}
            for key in metric_keys:
                item[key] = row.metrics.get(key, "")
            writer.writerow(item)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 eval/jsq_v1/jsq_v2 日志指标")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="运行组与日志目录，格式为 名称:目录，例如 jsq_v1:/path/to/logs",
    )
    parser.add_argument("--output-md", required=True, help="Markdown 汇总输出路径")
    parser.add_argument("--output-csv", required=True, help="CSV 汇总输出路径")
    args = parser.parse_args()

    rows: List[Row] = []
    for run_arg in args.run:
        if ":" not in run_arg:
            continue
        run_name, run_dir = run_arg.split(":", 1)
        log_dir = Path(run_dir)
        if not log_dir.exists() or not log_dir.is_dir():
            continue
        latest_log = _latest_log_file(log_dir)
        if latest_log is None:
            continue
        rows.extend(_parse_log(latest_log, run_name))

    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    _render_md(rows, output_md)
    _render_csv(rows, output_csv)

    print(f"汇总完成: {output_md}")
    print(f"汇总完成: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
