#!/usr/bin/env python3
"""Create browser-friendly TP0 traces for the split ON/OFF comparison."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path("/sgl-workspace/workspace/profile/20260819_dsv4_split_ab")
OUT = ROOT / "delivery" / "perfetto_lite"


def keep(event: dict) -> bool:
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
    ) or any(token in name for token in ("cuda graph", "profilerstep", "decode"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in ("split_on", "split_off"):
        source = next(
            (ROOT / variant / "decode").glob("*TP-0-DP-0-EP-0.trace.json.gz")
        )
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            trace = json.load(handle)
        original = len(trace.get("traceEvents", []))
        trace["traceEvents"] = [event for event in trace["traceEvents"] if keep(event)]
        trace["liteTraceInfo"] = {
            "variant": variant,
            "source": source.name,
            "original_event_count": original,
            "retained_event_count": len(trace["traceEvents"]),
        }
        destination = OUT / f"dsv4_{variant}_c8_w40_n80_decode_tp0.trace.json.gz"
        with gzip.open(destination, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(trace, handle, ensure_ascii=False, separators=(",", ":"))
        print(variant, original, len(trace["traceEvents"]), destination.stat().st_size)


if __name__ == "__main__":
    main()
