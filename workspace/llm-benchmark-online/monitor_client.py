from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

import httpx


SCHEMA_VERSION = "1.0"
# gpu-monitor performs a one-time batch preflight, then allows 30 seconds for
# profiler readiness. A short idempotent retry avoids another full timeout;
# the runner subsequently recovers the session through request_id lookup.
START_TIMEOUT_SECONDS = 45.0
START_RETRY_TIMEOUT_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 40.0
LOOKUP_TIMEOUT_SECONDS = 5.0
RETRY_DELAY_SECONDS = 0.25
STOP_STATUSES = {"done", "partial", "timeout"}


class MonitorClientError(RuntimeError):
    def __init__(self, message: str, *, error_code: str = "MONITOR_REQUEST_FAILED"):
        super().__init__(message)
        self.error_code = error_code


def _base_url(task: dict[str, Any]) -> str:
    host = str(task.get("monitor_host") or "").strip()
    port = int(task.get("monitor_port") or 30100)
    if not host:
        raise MonitorClientError("monitor_host is empty", error_code="INVALID_MONITOR_CONFIG")
    if "://" in host:
        parsed = urlsplit(host)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise MonitorClientError(
                f"invalid monitor_host: {host!r}",
                error_code="INVALID_MONITOR_CONFIG",
            )
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = f"[{hostname}]"
        # monitor_port is the source of truth even when monitor_host is a URL.
        return f"{parsed.scheme}://{hostname}:{port}"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _error_message(response: httpx.Response, payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        code = str(payload.get("error_code") or f"HTTP_{response.status_code}")
        message = str(payload.get("message") or payload.get("detail") or response.text)
        return code, message
    return f"HTTP_{response.status_code}", response.text or response.reason_phrase


async def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body)
    except httpx.TimeoutException as exc:
        raise MonitorClientError(
            f"request timed out after {timeout:g}s: {url}",
            error_code="MONITOR_REQUEST_TIMEOUT",
        ) from exc
    except httpx.HTTPError as exc:
        raise MonitorClientError(str(exc), error_code="MONITOR_CONNECTION_FAILED") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise MonitorClientError(
            f"monitor returned invalid JSON (HTTP {response.status_code})",
            error_code="INVALID_MONITOR_RESPONSE",
        ) from exc
    if not response.is_success:
        code, message = _error_message(response, payload)
        raise MonitorClientError(message, error_code=code)
    if not isinstance(payload, dict):
        raise MonitorClientError(
            "monitor response must be a JSON object",
            error_code="INVALID_MONITOR_RESPONSE",
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MonitorClientError(
            f"unsupported monitor schema_version: {payload.get('schema_version')!r}",
            error_code="UNSUPPORTED_MONITOR_SCHEMA",
        )
    return payload


async def _get_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params)
    except httpx.TimeoutException as exc:
        raise MonitorClientError(
            f"request timed out after {timeout:g}s: {url}",
            error_code="MONITOR_REQUEST_TIMEOUT",
        ) from exc
    except httpx.HTTPError as exc:
        raise MonitorClientError(str(exc), error_code="MONITOR_CONNECTION_FAILED") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise MonitorClientError(
            f"monitor returned invalid JSON (HTTP {response.status_code})",
            error_code="INVALID_MONITOR_RESPONSE",
        ) from exc
    if not response.is_success:
        code, message = _error_message(response, payload)
        raise MonitorClientError(message, error_code=code)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise MonitorClientError(
            "monitor returned an invalid request lookup response",
            error_code="INVALID_MONITOR_RESPONSE",
        )
    return payload


async def _post_json_with_timeout_retry(
    url: str,
    body: dict[str, Any],
    timeout: float,
    retry_timeout: float | None = None,
) -> dict[str, Any]:
    """Retry once after an ambiguous timeout, reusing the same idempotency key."""
    try:
        return await _post_json(url, body, timeout)
    except MonitorClientError as exc:
        if exc.error_code != "MONITOR_REQUEST_TIMEOUT":
            raise
        await asyncio.sleep(RETRY_DELAY_SECONDS)
        return await _post_json(
            url, body, retry_timeout if retry_timeout is not None else timeout
        )


async def start_monitor(task: dict[str, Any], run: dict[str, Any], request_id: str) -> dict[str, Any]:
    payload = await _post_json_with_timeout_retry(
        _base_url(task) + "/api/v1/monitor/start",
        {
            "run_id": run["id"],
            "request_id": request_id,
            "devices": str(
                task.get("monitor_devices")
                or task.get("monitor_device")
                or 0
            ),
            "interval_ms": int(task.get("monitor_interval_ms") or 100),
            "max_duration_seconds": int(task.get("monitor_max_duration") or 7200),
        },
        START_TIMEOUT_SECONDS,
        START_RETRY_TIMEOUT_SECONDS,
    )
    if payload.get("status") != "running" or not payload.get("session_id"):
        raise MonitorClientError(
            "monitor start response is missing running status or session_id",
            error_code="INVALID_MONITOR_RESPONSE",
        )
    return payload


async def recover_monitor_session(
    task: dict[str, Any],
    run: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Find an existing session by request ID without creating a profiler."""
    payload = await _get_json(
        _base_url(task) + f"/api/v1/monitor/requests/{request_id}",
        {"run_id": run["id"]},
        LOOKUP_TIMEOUT_SECONDS,
    )
    if not payload.get("session_id") or payload.get("status") not in ({"running"} | STOP_STATUSES):
        raise MonitorClientError(
            "monitor recovery response is missing session_id or has invalid status",
            error_code="INVALID_MONITOR_RESPONSE",
        )
    return payload


async def stop_monitor(task: dict[str, Any], session_id: str) -> dict[str, Any]:
    payload = await _post_json_with_timeout_retry(
        _base_url(task) + "/api/v1/monitor/stop",
        {"session_id": session_id},
        STOP_TIMEOUT_SECONDS,
    )
    if payload.get("session_id") != session_id or payload.get("status") not in STOP_STATUSES:
        raise MonitorClientError(
            "monitor stop response has an invalid session_id or status",
            error_code="INVALID_MONITOR_RESPONSE",
        )
    if not isinstance(payload.get("summary"), dict) or not isinstance(payload.get("samples"), list):
        raise MonitorClientError(
            "monitor stop response must contain summary and samples",
            error_code="INVALID_MONITOR_RESPONSE",
        )
    return payload
