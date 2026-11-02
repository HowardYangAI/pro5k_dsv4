#!/usr/bin/env python3
"""Summarize paired no-profiler full-E2E benchmark logs."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


ROOT = Path("/sgl-workspace/workspace/profile/20260819_dsv4_split_e2e_ab")
BENCH = ROOT / "no_profiler_benchmark"
OUT = ROOT / "delivery" / "no_profiler_e2e_ab"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = {
    "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "request_tps": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_tps": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "accept_length": r"Accept length:\s+([0-9.]+)",
    "mean_e2e_ms": r"Mean E2E Latency \(ms\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
}


def parse(variant: str, path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {"variant": variant, "run": int(path.name.split("_")[1])}
    row["seed"] = int(re.search(r"seed_(\d+)", path.name).group(1))
    for key, pattern in FIELDS.items():
        row[key] = float(re.search(pattern, text).group(1))
    return row


rows = []
for variant in ("split_on", "split_off"):
    rows.extend(parse(variant, path) for path in sorted((BENCH / variant).glob("*.log")))

with (OUT / "01_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for variant in ("split_on", "split_off"):
    selected = [row for row in rows if row["variant"] == variant]
    for metric in FIELDS:
        values = [row[metric] for row in selected]
        summary.append(
            {
                "variant": variant,
                "metric": metric,
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        )

with (OUT / "02_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

on_by_seed = {row["seed"]: row for row in rows if row["variant"] == "split_on"}
off_by_seed = {row["seed"]: row for row in rows if row["variant"] == "split_off"}
paired = []
for seed in sorted(on_by_seed):
    on, off = on_by_seed[seed], off_by_seed[seed]
    paired.append(
        {
            "seed": seed,
            "output_tps_on": on["output_tps"],
            "output_tps_off": off["output_tps"],
            "output_tps_on_gain_pct": (on["output_tps"] / off["output_tps"] - 1) * 100,
            "duration_on_s": on["duration_s"],
            "duration_off_s": off["duration_s"],
            "duration_on_reduction_pct": (1 - on["duration_s"] / off["duration_s"]) * 100,
            "tpot_on_ms": on["mean_tpot_ms"],
            "tpot_off_ms": off["mean_tpot_ms"],
            "tpot_on_reduction_pct": (1 - on["mean_tpot_ms"] / off["mean_tpot_ms"]) * 100,
            "accept_length_on": on["accept_length"],
            "accept_length_off": off["accept_length"],
        }
    )

with (OUT / "03_paired_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)


def values(variant: str, metric: str) -> list[float]:
    return [row[metric] for row in rows if row["variant"] == variant]


def mean(variant: str, metric: str) -> float:
    return statistics.mean(values(variant, metric))


on_tps, off_tps = mean("split_on", "output_tps"), mean("split_off", "output_tps")
on_duration, off_duration = mean("split_on", "duration_s"), mean("split_off", "duration_s")
on_tpot, off_tpot = mean("split_on", "mean_tpot_ms"), mean("split_off", "mean_tpot_ms")
on_e2e, off_e2e = mean("split_on", "mean_e2e_ms"), mean("split_off", "mean_e2e_ms")
on_ttft, off_ttft = mean("split_on", "mean_ttft_ms"), mean("split_off", "mean_ttft_ms")
on_accept, off_accept = mean("split_on", "accept_length"), mean("split_off", "accept_length")

report = f"""# PR #32320 无 Profiler 完整端到端 A/B

负载：C8、40 requests、8192 input tokens、64 output tokens、TP8/Attention DP8/EP8、DSpark 6 draft tokens、Decode CUDA Graph。Split ON/OFF 使用相同的 3 组 seed，每轮均成功 40/40。

| 指标（三轮均值） | Split ON | Split OFF | ON 相对 OFF |
| --- | ---: | ---: | ---: |
| Benchmark duration | {on_duration:.2f} s | {off_duration:.2f} s | {(1-on_duration/off_duration)*100:+.2f}% |
| Output TPS | {on_tps:.2f} tok/s | {off_tps:.2f} tok/s | {(on_tps/off_tps-1)*100:+.2f}% |
| Mean E2E | {on_e2e:.2f} ms | {off_e2e:.2f} ms | {(1-on_e2e/off_e2e)*100:+.2f}% |
| Mean TTFT | {on_ttft:.2f} ms | {off_ttft:.2f} ms | {(1-on_ttft/off_ttft)*100:+.2f}% |
| Mean TPOT | {on_tpot:.2f} ms | {off_tpot:.2f} ms | {(1-on_tpot/off_tpot)*100:+.2f}% |
| Accept length | {on_accept:.2f} | {off_accept:.2f} | {(on_accept/off_accept-1)*100:+.2f}% |

逐 seed 的 Output TPS 收益分别为：{', '.join(f'{row["output_tps_on_gain_pct"]:.2f}%' for row in paired)}；配对中位数为 {statistics.median(row['output_tps_on_gain_pct'] for row in paired):.2f}%。

结论：三轮均值下 Split ON 的 Output TPS 高约 {(on_tps/off_tps-1)*100:.2f}%，完成时间低约 {(1-on_duration/off_duration)*100:.2f}%，Mean TPOT 低约 {(1-on_tpot/off_tpot)*100:.2f}%。但逐轮收益波动较大，且 ON 的平均 Accept length 更低。结合配对中位数和此前 Decode-only 结果，更稳妥的结论是“优化有端到端收益，量级约为几个百分点”；不使用单轮 Profile 得到的 19.22% 作为确定收益。
"""
(OUT / "README.md").write_text(report, encoding="utf-8")
