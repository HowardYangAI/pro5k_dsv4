#!/usr/bin/env python3
"""Build full E2E TP0 tables and a browser-friendly Perfetto trace."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path("/sgl-workspace/workspace/profile/20260819_dsv4_split_e2e_ab")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "split_on"
if VARIANT not in {"split_on", "split_off"}:
    raise SystemExit("variant must be split_on or split_off")
SOURCE_ROOT = ROOT / VARIANT / "trace_c4_8k64"
OUT = ROOT / "delivery" / f"{VARIANT}_c4_8k64_e2e"
SKILL = Path("/root/.codex/skills/llm-torch-profiler-analysis/scripts")
sys.path.insert(0, str(SKILL))

import triage_kernel_helpers as kh  # noqa: E402


def short_name(name: str) -> str:
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


def keep_for_perfetto(event: dict) -> bool:
    if event.get("ph") == "M":
        return True
    cat = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    return any(
        token in cat
        for token in (
            "kernel",
            "cuda_runtime",
            "gpu_memcpy",
            "gpu_memset",
            "user_annotation",
        )
    ) or any(
        token in name
        for token in ("cuda graph", "profilerstep", "prefill", "extend", "decode")
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = next(SOURCE_ROOT.glob("*TP-0-DP-0-EP-0.trace.json.gz"))
    with gzip.open(source, "rt", encoding="utf-8") as handle:
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
    aggregate = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
    raw_name, category = {}, {}
    timeline = []
    for seq, kernel in enumerate(kernels, 1):
        location, cpu_op = source_by_kernel.get(kernel.canonical_name, ("unresolved", ""))
        timeline.append(
            {
                "序号": seq,
                "相对开始时间(ms)": round((kernel.ts - start_ts) / 1000, 6),
                "Duration(ms)": round(kernel.dur / 1000, 6),
                "百分比(%)": round(kernel.dur / total_us * 100, 6),
                "Stream": kernel.tid,
                "op name": kernel.name,
                "算子简称": short_name(kernel.canonical_name),
                "类别": kernel.category,
                "Python源码": location,
                "CPU op": cpu_op,
            }
        )
        row = aggregate[kernel.canonical_name]
        row["count"] += 1
        row["total_ms"] += kernel.dur / 1000
        row["max_ms"] = max(row["max_ms"], kernel.dur / 1000)
        raw_name[kernel.canonical_name] = kernel.name
        category[kernel.canonical_name] = kernel.category
    write_csv(OUT / "01_e2e_operator_timeline_tp0.csv", timeline)

    cumulative = 0.0
    aggregate_rows = []
    for canonical, stats in sorted(
        aggregate.items(), key=lambda pair: pair[1]["total_ms"], reverse=True
    ):
        cumulative += stats["total_ms"]
        location, cpu_op = source_by_kernel.get(canonical, ("unresolved", ""))
        aggregate_rows.append(
            {
                "op name": raw_name[canonical],
                "算子简称": short_name(canonical),
                "类别": category[canonical],
                "调用次数": stats["count"],
                "总时间(ms)": round(stats["total_ms"], 6),
                "平均时间(ms)": round(stats["total_ms"] / stats["count"], 6),
                "最大时间(ms)": round(stats["max_ms"], 6),
                "时间占比(%)": round(stats["total_ms"] / (total_us / 1000) * 100, 6),
                "累计占比(%)": round(cumulative / (total_us / 1000) * 100, 6),
                "Python源码": location,
                "CPU op": cpu_op,
            }
        )
    write_csv(OUT / "02_e2e_kernel_aggregate_tp0.csv", aggregate_rows)

    original = len(trace.get("traceEvents", []))
    trace["traceEvents"] = [event for event in trace["traceEvents"] if keep_for_perfetto(event)]
    trace["liteTraceInfo"] = {
        "scope": "full_e2e_8192_to_64",
        "variant": VARIANT,
        "source": source.name,
        "original_event_count": original,
        "retained_event_count": len(trace["traceEvents"]),
    }
    destination = OUT / "03_e2e_8k_to_64_tp0_perfetto_lite.trace.json.gz"
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(trace, handle, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
