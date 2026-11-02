#!/usr/bin/env python3
"""Create browser-friendly Chrome traces while preserving GPU execution tracks."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path("/sgl-workspace/workspace/profile/20260819_dsv4_ep8_dspark_c8")
OUT = ROOT / "delivery" / "perfetto_lite"


def keep_event(event: dict) -> bool:
    if event.get("ph") == "M":
        return True
    category = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    return any(
        marker in category
        for marker in (
            "kernel",
            "cuda_runtime",
            "gpu_memcpy",
            "gpu_memset",
            "user_annotation",
            "trace",
        )
    ) or any(
        marker in name
        for marker in (
            "cuda graph",
            "profilerstep",
            "prefill",
            "decode",
            "forward",
        )
    )


def reduce_trace(stage: str) -> None:
    source = next((ROOT / stage).glob("*TP-0-DP-0-EP-0.trace.json.gz"))
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)

    original_count = len(trace.get("traceEvents", []))
    trace["traceEvents"] = [
        event for event in trace.get("traceEvents", []) if keep_event(event)
    ]
    trace.setdefault("liteTraceInfo", {})
    trace["liteTraceInfo"].update(
        {
            "source": source.name,
            "stage": stage,
            "original_event_count": original_count,
            "retained_event_count": len(trace["traceEvents"]),
            "purpose": "Browser-friendly Perfetto view; use the original trace for full CPU/source analysis.",
        }
    )

    destination = OUT / f"dsv4_ep8_dspark_c8_{stage}_tp0_perfetto_lite.trace.json.gz"
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(trace, handle, ensure_ascii=False, separators=(",", ":"))
    print(
        f"{stage}: {original_count} -> {len(trace['traceEvents'])} events; "
        f"{destination.stat().st_size} bytes"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reduce_trace("prefill")
    reduce_trace("decode")


if __name__ == "__main__":
    main()
