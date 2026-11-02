#!/usr/bin/env python3
"""Build TP0 timeline, aggregate, and A/B comparison tables for PR #32320."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path("/sgl-workspace/workspace/profile/20260819_dsv4_split_ab")
OUT = ROOT / "delivery"
SKILL = Path("/root/.codex/skills/llm-torch-profiler-analysis/scripts")
sys.path.insert(0, str(SKILL))

import triage_kernel_helpers as kh  # noqa: E402


def trace_for(variant: str) -> Path:
    return next((ROOT / variant / "decode").glob("*TP-0-DP-0-EP-0.trace.json.gz"))


def label(name: str) -> str:
    lowered = name.lower()
    for needle, value in (
        ("_page_mark_kernel", "SWA touched-page mark"),
        ("_page_split_kernel", "SWA page split"),
        ("allreduce", "NCCL AllReduce"),
        ("allgather", "NCCL AllGather"),
        ("reducescatter", "NCCL ReduceScatter"),
        ("ncclDevKernel_Reduce", "NCCL Reduce"),
        ("gemmsn_tn_kernel", "cuBLAS small-N GEMM"),
        ("gemmuniversal", "SM120 CUTLASS GEMM"),
        ("sparse_mla", "FlashMLA sparse attention"),
        ("moe", "MoE kernel"),
    ):
        if needle.lower() in lowered:
            return value
    return name[:120]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_variant(variant: str) -> tuple[Path, list[dict], dict[str, dict]]:
    path = trace_for(variant)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    kernels, cpu_ops, python_frames, launches, _, _ = kh.extract_trace_data(trace)
    kernels.sort(key=lambda item: (item.ts, item.tid))

    cpu_index = kh.build_cpu_op_index(cpu_ops)
    launch_index = kh.build_launch_index(launches)
    sites = kh.aggregate_kernel_sites(
        kernels, cpu_index, python_frames, launches_by_correlation=launch_index
    )
    source_by_kernel = {}
    for kernel_name, locations in sites.items():
        location, stats = max(locations.items(), key=lambda pair: pair[1].total_us)
        cpu_op = stats.cpu_ops.most_common(1)[0][0] if stats.cpu_ops else ""
        source_by_kernel[kernel_name] = (location, cpu_op)

    total_us = sum(kernel.dur for kernel in kernels)
    start_ts = min(kernel.ts for kernel in kernels)
    timeline = []
    aggregate: dict[str, dict] = defaultdict(
        lambda: {"调用次数": 0, "总时间(ms)": 0.0, "最大时间(ms)": 0.0}
    )
    raw_name = {}
    category = {}
    for seq, kernel in enumerate(kernels, 1):
        source, cpu_op = source_by_kernel.get(kernel.canonical_name, ("unresolved", ""))
        timeline.append(
            {
                "序号": seq,
                "相对开始时间(ms)": round((kernel.ts - start_ts) / 1000.0, 6),
                "Duration(ms)": round(kernel.dur / 1000.0, 6),
                "百分比(%)": round(kernel.dur / total_us * 100, 6),
                "Stream": kernel.tid,
                "op name": kernel.name,
                "算子简称": label(kernel.canonical_name),
                "类别": kernel.category,
                "Python源码": source,
                "CPU op": cpu_op,
            }
        )
        row = aggregate[kernel.canonical_name]
        row["调用次数"] += 1
        row["总时间(ms)"] += kernel.dur / 1000.0
        row["最大时间(ms)"] = max(row["最大时间(ms)"], kernel.dur / 1000.0)
        raw_name[kernel.canonical_name] = kernel.name
        category[kernel.canonical_name] = kernel.category

    write_csv(OUT / f"01_{variant}_decode_timeline_tp0.csv", timeline)
    cumulative = 0.0
    aggregate_rows = []
    for canonical, stats in sorted(
        aggregate.items(), key=lambda pair: pair[1]["总时间(ms)"], reverse=True
    ):
        cumulative += stats["总时间(ms)"]
        source, cpu_op = source_by_kernel.get(canonical, ("unresolved", ""))
        aggregate_rows.append(
            {
                "op name": raw_name[canonical],
                "算子简称": label(canonical),
                "类别": category[canonical],
                "调用次数": stats["调用次数"],
                "总时间(ms)": round(stats["总时间(ms)"], 6),
                "平均时间(ms)": round(stats["总时间(ms)"] / stats["调用次数"], 6),
                "最大时间(ms)": round(stats["最大时间(ms)"], 6),
                "时间占比(%)": round(stats["总时间(ms)"] / (total_us / 1000) * 100, 6),
                "累计占比(%)": round(cumulative / (total_us / 1000) * 100, 6),
                "Python源码": source,
                "CPU op": cpu_op,
            }
        )
    write_csv(OUT / f"02_{variant}_decode_kernel_aggregate_tp0.csv", aggregate_rows)
    return path, aggregate_rows, aggregate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    on_trace, on_rows, on = build_variant("split_on")
    off_trace, off_rows, off = build_variant("split_off")

    names = {row["op name"] for row in on_rows + off_rows}
    by_name_on = {row["op name"]: row for row in on_rows}
    by_name_off = {row["op name"]: row for row in off_rows}
    comparison = []
    for op_name in sorted(
        names,
        key=lambda key: max(
            float(by_name_on.get(key, {}).get("总时间(ms)", 0)),
            float(by_name_off.get(key, {}).get("总时间(ms)", 0)),
        ),
        reverse=True,
    ):
        on_row = by_name_on.get(op_name, {})
        off_row = by_name_off.get(op_name, {})
        on_ms = float(on_row.get("总时间(ms)", 0))
        off_ms = float(off_row.get("总时间(ms)", 0))
        comparison.append(
            {
                "op name": op_name,
                "算子简称": on_row.get("算子简称", off_row.get("算子简称", "")),
                "Split ON 调用次数": on_row.get("调用次数", 0),
                "Split ON 总时间(ms)": on_ms,
                "Split ON 占比(%)": on_row.get("时间占比(%)", 0),
                "Split OFF 调用次数": off_row.get("调用次数", 0),
                "Split OFF 总时间(ms)": off_ms,
                "Split OFF 占比(%)": off_row.get("时间占比(%)", 0),
                "OFF-ON 时间(ms)": round(off_ms - on_ms, 6),
                "OFF 相对 ON 变化(%)": round((off_ms / on_ms - 1) * 100, 6)
                if on_ms
                else "",
            }
        )
    write_csv(OUT / "03_split_on_off_kernel_comparison_tp0.csv", comparison)

    analyzer = SKILL / "analyze_llm_torch_profile.py"
    reports = []
    for variant, trace in (("Split ON", on_trace), ("Split OFF", off_trace)):
        result = subprocess.run(
            [sys.executable, str(analyzer), "--framework", "sglang", "--input", str(trace)],
            check=True,
            text=True,
            capture_output=True,
        )
        reports.append(f"# {variant}\n\n{result.stdout}")
    (OUT / "04_skill_triage_report.md").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
