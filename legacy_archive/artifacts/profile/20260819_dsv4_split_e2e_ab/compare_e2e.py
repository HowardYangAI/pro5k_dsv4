#!/usr/bin/env python3
"""Compare full-E2E Split ON/OFF TP0 kernel aggregates."""

import csv
from pathlib import Path


ROOT = Path("/sgl-workspace/workspace/profile/20260819_dsv4_split_e2e_ab/delivery")
ON = ROOT / "split_on_c4_8k64_e2e" / "02_e2e_kernel_aggregate_tp0.csv"
OFF = ROOT / "split_off_c4_8k64_e2e" / "02_e2e_kernel_aggregate_tp0.csv"
OUT = ROOT / "05_e2e_split_on_off_kernel_comparison_tp0.csv"


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["op name"]: row for row in csv.DictReader(handle)}


on, off = load(ON), load(OFF)
names = sorted(
    set(on) | set(off),
    key=lambda name: max(
        float(on.get(name, {}).get("总时间(ms)", 0)),
        float(off.get(name, {}).get("总时间(ms)", 0)),
    ),
    reverse=True,
)
rows = []
for name in names:
    on_row, off_row = on.get(name, {}), off.get(name, {})
    on_ms = float(on_row.get("总时间(ms)", 0))
    off_ms = float(off_row.get("总时间(ms)", 0))
    rows.append(
        {
            "op name": name,
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

with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
