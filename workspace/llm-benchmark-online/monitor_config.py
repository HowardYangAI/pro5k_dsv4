from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


DEFAULT_MONITOR_PORT = 30100
DEFAULT_MONITOR_DEVICE = 0
DEFAULT_MONITOR_DEVICES = "auto"
DEFAULT_MONITOR_INTERVAL_MS = 100
DEFAULT_MONITOR_MAX_DURATION = 7200


class MonitorConfigError(ValueError):
    pass


def normalize_monitor_devices(value: Any) -> tuple[str, list[int] | None]:
    if value is None:
        return DEFAULT_MONITOR_DEVICES, None
    if isinstance(value, int):
        if value < 0:
            raise MonitorConfigError("monitor devices must be non-negative")
        return str(value), [value]
    if isinstance(value, list):
        if not value or any(not isinstance(item, int) or item < 0 for item in value):
            raise MonitorConfigError("monitor devices must contain non-negative integers")
        devices = sorted(set(value))
        return ",".join(str(item) for item in devices), devices

    text = str(value).strip().lower()
    if text == "auto":
        return "auto", None
    if not text:
        raise MonitorConfigError("monitor devices cannot be empty")
    selected = set()
    for part in text.replace("/", ",").split(","):
        part = part.strip()
        if not part:
            raise MonitorConfigError(f"invalid monitor devices selector: {value!r}")
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2 or not all(item.isdigit() for item in bounds):
                raise MonitorConfigError(f"invalid monitor device range: {part!r}")
            start, end = (int(item) for item in bounds)
            if start > end:
                raise MonitorConfigError(f"monitor device range must be ascending: {part!r}")
            selected.update(range(start, end + 1))
        elif part.isdigit():
            selected.add(int(part))
        else:
            raise MonitorConfigError(f"invalid monitor device value: {part!r}")
    if not selected or len(selected) > 64:
        raise MonitorConfigError("monitor devices must select between 1 and 64 GPUs")
    devices = sorted(selected)
    return ",".join(str(item) for item in devices), devices


def parse_monitor_config(body: dict[str, Any], connection_mode: str) -> dict[str, Any]:
    enabled_raw = body.get("monitor_enabled", False)
    if not isinstance(enabled_raw, (bool, int)) or enabled_raw not in (False, True, 0, 1):
        raise MonitorConfigError("monitor_enabled must be a boolean")
    enabled = bool(enabled_raw)
    host = ""
    try:
        port = int(body.get("monitor_port", DEFAULT_MONITOR_PORT))
        interval_ms = int(body.get("monitor_interval_ms", DEFAULT_MONITOR_INTERVAL_MS))
        max_duration = int(body.get("monitor_max_duration", DEFAULT_MONITOR_MAX_DURATION))
    except (TypeError, ValueError) as exc:
        raise MonitorConfigError("monitor numeric fields must be integers") from exc

    if enabled:
        if connection_mode == "host_port":
            host = str(body.get("host", "") or "").strip()
        elif connection_mode == "base_url":
            base_url = str(body.get("base_url", "") or "").strip()
            parsed = urlsplit(base_url)
            host = parsed.hostname or ""
        if not host:
            raise MonitorConfigError("SGLang host is required when monitoring is enabled")
        if port < 1 or port > 65535:
            raise MonitorConfigError("monitor_port must be between 1 and 65535")
        if interval_ms < 50:
            raise MonitorConfigError("monitor_interval_ms must be at least 50")
        if max_duration < 1 or max_duration > DEFAULT_MONITOR_MAX_DURATION:
            raise MonitorConfigError(
                f"monitor_max_duration must be between 1 and {DEFAULT_MONITOR_MAX_DURATION}"
            )

    selector_source = body.get("monitor_devices")
    if selector_source is None:
        selector_source = body.get("monitor_device", DEFAULT_MONITOR_DEVICE)
    devices_selector, explicit_devices = normalize_monitor_devices(selector_source)

    return {
        "monitor_enabled": enabled,
        "monitor_host": host,
        "monitor_port": port,
        "monitor_device": explicit_devices[0] if explicit_devices else 0,
        "monitor_devices": devices_selector,
        "monitor_interval_ms": interval_ms,
        "monitor_max_duration": max_duration,
    }
