from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

from bench_result_parser import parse_benchmark_file
from db import (
    get_run,
    get_interrupted_monitor_runs,
    get_task,
    get_task_runs,
    save_run_monitor_result,
    update_run,
    update_run_monitor_status,
    update_task_status,
)
from monitor_client import (
    MonitorClientError,
    recover_monitor_session,
    start_monitor,
    stop_monitor,
)
from replay_store import DEFAULT_STORE

BENCH_DIR = Path(__file__).parent
BENCH_SH = str(BENCH_DIR / "bench.sh")

# Track running processes so they can be killed
# Maps run_id -> (Process, pgid) so we can kill the entire process group
_running_procs: dict[int, tuple[asyncio.subprocess.Process, int]] = {}

# Set of task_ids that have been requested to kill (persists until execute_task cleans up)
_killing_tasks: set[int] = set()

# Cache for auto-detected model names: task_id -> model_name
_model_name_cache: dict[int, str] = {}


async def resolve_model_name(task: dict) -> str:
    """Auto-detect model name by querying /v1/models endpoint."""
    task_id = task["id"]
    if task_id in _model_name_cache:
        return _model_name_cache[task_id]

    if task["connection_mode"] == "base_url":
        url = task["base_url"].rstrip("/") + "/v1/models"
    else:
        url = f"http://{task['host']}:{task['port']}/v1/models"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            data = resp.json()
            models = data.get("data", [])
            if models:
                name = models[0].get("id", "default")
                _model_name_cache[task_id] = name
                return name
    except Exception:
        pass
    _model_name_cache[task_id] = "default"
    return "default"


def _get_model_name(task: dict) -> str:
    """Get model name from task config (manual or request_model)."""
    return task.get("model_name") or task.get("request_model") or ""


def build_bench_args(task: dict, run: dict, model_name: str = "") -> list[str]:
    """Construct arg list for bench.sh from task + run config."""
    args = [
        "bash", BENCH_SH,
        "--model-path", task["model_path"],
        "--dataset-path", task["dataset_path"],
    ]

    if task["connection_mode"] == "base_url":
        args.extend(["--base-url", task["base_url"]])
        if task["api_key"]:
            args.extend(["--api-key", task["api_key"]])
        if task["request_model"]:
            args.extend(["--request-model", task["request_model"]])
    else:
        args.extend(["--host", task["host"]])
        args.extend(["--port", str(task["port"])])

    if model_name:
        args.extend(["--request-model", model_name])

    # 检测测试模式
    test_mode = run.get("test_mode", "normal")

    if test_mode == "shared_prefix":
        # 公共前缀测试：使用 generated-shared-prefix 数据集
        num_groups = run.get("gsp_num_groups", 10)
        prompts_per_group = run.get("gsp_prompts_per_group", 512)
        system_prompt_len = run.get("gsp_system_prompt_len", 2048)
        question_len = run.get("gsp_question_len", 128)
        output_len = run.get("random_output_len", 256)
        range_ratio = run.get("gsp_range_ratio", 0.0)
        max_concurrency = run.get("gsp_max_concurrency", 0)
        rate = run.get("request_rate", 0)
        rate_str = "inf" if rate <= 0 else str(rate)

        total_prompts = num_groups * prompts_per_group

        # 重要：公共前缀测试不依赖 bench.sh 的并发循环逻辑
        # 直接使用 --concurrencies 1 和 --prompt-multiplier total_prompts
        args.extend([
            "--concurrencies", "1",
            "--prompt-multiplier", str(total_prompts),
            "--random-input-len", str(system_prompt_len + question_len),
            "--random-output-len", str(output_len),
        ])

        # 透传参数：这些会覆盖 bench.sh 的 base_cmd 参数
        passthrough_args = [
            "--dataset-name", "generated-shared-prefix",
            "--gsp-num-groups", str(num_groups),
            "--gsp-prompts-per-group", str(prompts_per_group),
            "--gsp-system-prompt-len", str(system_prompt_len),
            "--gsp-question-len", str(question_len),
            "--gsp-output-len", str(output_len),
            "--request-rate", rate_str,
            "--num-prompts", str(total_prompts),  # 必须传递，覆盖 bench.sh 的计算值
        ]

        # 添加随机范围参数
        if range_ratio > 0.0:
            passthrough_args.extend(["--gsp-range-ratio", str(range_ratio)])

        # 添加并发限制参数（必须传递，覆盖 bench.sh 的 --max-concurrency 1）
        if max_concurrency > 0:
            passthrough_args.extend(["--max-concurrency", str(max_concurrency)])
        else:
            # 如果用户不限制并发，传递一个合理的值（或不传递让 bench_serving.py 自动处理）
            # 但为了确保覆盖 bench.sh 的 --max-concurrency 1，我们需要显式传递
            passthrough_args.extend(["--max-concurrency", str(total_prompts)])  # 不限制的实际效果

        args.extend(["--"] + passthrough_args)
    elif test_mode == "multi_turn":
        # 多轮对话测试：使用 bench_multiturn.py（bench.sh --mode multi-turn）
        args.extend([
            "--mode", "multi-turn",
        ])

        # 透传参数：bench.sh 追加到最后，覆盖其 base_cmd
        rate = run.get("request_rate", 0)
        rate_str = "inf" if rate <= 0 else str(rate)
        passthrough_args = [
            "--num-clients", str(run.get("mt_num_clients", 10)),
            "--num-rounds", str(run.get("mt_num_rounds", 5)),
            "--request-length", str(run.get("mt_request_length", 1024)),
            "--sub-question-input-length", str(run.get("mt_sub_question_input_length", 256)),
            "--output-length", str(run.get("mt_output_length", 256)),
            "--range-ratio", str(run.get("mt_range_ratio", 0.5)),
            "--max-context-length", str(run.get("mt_max_context_length", 0)),
            "--init-interval", str(run.get("mt_init_interval", 0.0)),
            "--request-rate", rate_str,
            "--total-requests", str(run.get("mt_total_requests", 0)),
            "--disable-auto-run",  # 平台控制参数，不走工具的自动速率阶梯
            "--log-file", f"/tmp/multiturn_run_{run['id']}.jsonl",
        ]
        if run.get("mt_workload_mode", "synthetic") == "replay":
            passthrough_args.extend([
                "--workload-mode", "replay",
                "--replay-dataset-path", str(DEFAULT_STORE),
            ])
            if run.get("mt_replay_sequential_sessions", 0):
                passthrough_args.append("--replay-sequential-sessions")
        elif run.get("mt_workload_mode") == "timestamp_replay":
            passthrough_args.extend([
                "--workload-mode", "timestamp_replay",
                "--replay-dataset-path", str(DEFAULT_STORE),
            ])
        # 发射分布不暴露：工具默认 uniform，不传 --distribution
        passthrough_args.extend(["--api-format", str(run.get("mt_api_format", "openai"))])
        # Thinking 模式：默认关（--no-thinking → chat_template_kwargs.thinking=false）
        passthrough_args.append("--thinking" if run.get("mt_thinking", 0) else "--no-thinking")
        if run.get("mt_seed"):
            # 固定种子（可复现）；留空=纯随机，bench.sh 用时间戳种子
            passthrough_args.extend(["--seed", str(run["mt_seed"])])
        if run.get("mt_host_header"):
            passthrough_args.extend(["--host-header", run["mt_host_header"]])
        if not run.get("mt_enable_session_key", 1):
            passthrough_args.append("--disable-session-key")
        if run.get("mt_enable_round_barrier"):
            passthrough_args.append("--enable-round-barrier")
        args.extend(["--"] + passthrough_args)
    else:
        # 普通性能测试
        args.extend([
            "--concurrencies", str(run["concurrency"]),
            "--prompt-multiplier", str(run["prompt_multiplier"]),
            "--random-input-len", str(run["random_input_len"]),
            "--random-output-len", str(run["random_output_len"]),
        ])

        # Passthrough args
        rate = run["request_rate"]
        rate_str = "inf" if rate <= 0 else str(rate)
        args.extend(["--", "--request-rate", rate_str])

    extra_args = json.loads(task.get("extra_args") or "[]")
    if extra_args:
        # Split each extra arg by whitespace to handle cases like "--seed 1024"
        for arg in extra_args:
            args.extend(arg.split())

    return args


def _parse_multiturn_jsonl(jsonl_path: str) -> dict:
    """Parse bench_multiturn.py JSONL output into a flat dict with ms units.

    JSONL line: {timestamp, tag, summary: {...}, round: {round_0: {...}}}.
    The tool records times in SECONDS; convert to ms to match the platform
    convention. round_N keys exist only when --enable-round-barrier is used
    (the tool uses 0-based round keys; display as 第 N 轮).
    """
    import math
    import re
    if not jsonl_path or not os.path.exists(jsonl_path):
        return {}
    try:
        with open(jsonl_path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if not lines:
            return {}
        data = json.loads(lines[-1])  # 取最后一行 = 本次运行的结果
    except (json.JSONDecodeError, OSError):
        return {}

    result = {}
    s = data.get("summary") or {}
    # 字段名 → 展示名；含 (ms) 的字段工具单位为秒，需要 ×1000
    ms_fields = {
        "total_requests": "Total requests",
        "request_rate": "Traffic request rate",
        "average_prompt_len": "Average Prompt Length",
        "p90_prompt_len": "P90 Prompt Length",
        "p99_prompt_len": "P99 Prompt Length",
        "average_output_len": "Average Output Length",
        "p90_output_len": "P90 Output Length",
        "p99_output_len": "P99 Output Length",
        "average_ttft": "Average TTFT (ms)",
        "p90_ttft": "P90 TTFT (ms)",
        "p99_ttft": "P99 TTFT (ms)",
        "median_ttft": "Median TTFT (ms)",
        "max_ttft": "Max TTFT (ms)",
        "average_tpot": "Average TPOT (ms)",
        "p90_tpot": "P90 TPOT (ms)",
        "p99_tpot": "P99 TPOT (ms)",
        "median_tpot": "Median TPOT (ms)",
        "max_tpot": "Max TPOT (ms)",
        "average_latency": "Average Latency (ms)",
        "p90_latency": "P90 Latency (ms)",
        "p99_latency": "P99 Latency (ms)",
        "median_latency": "Median Latency (ms)",
        "max_latency": "Max Latency (ms)",
        "input_token_throughput": "Input token throughput (tok/s)",
        "output_token_throughput": "Output token throughput (tok/s)",
        "throughput": "Request throughput (req/s)",
        "cache_hit_rate": "Cache hit rate",
        "cache_hit_rate_l1": "Cache hit rate L1",
        "cache_hit_rate_l2": "Cache hit rate L2",
        "cache_has_l2": "Cache has L2",
        # 理论命中率（工具按每轮实际输入/输出计算）
        "theoretical_hit_rate_pd_sep": "Theoretical Hit Rate (PD Sep)",
        "theoretical_hit_rate_pd_no_sep": "Theoretical Hit Rate (PD No Sep)",
        "completed_sessions": "Completed sessions",
        "dataset_sessions": "Replay dataset sessions",
        "dataset_cycles": "Replay dataset cycles",
    }
    for k, label in ms_fields.items():
        if k in s and isinstance(s[k], (int, float)):
            v = s[k]
            if isinstance(v, float) and not math.isfinite(v):
                # inf 速率：存为字符串，避免 JSON 序列化出 Infinity 导致前端解析失败
                result[label] = "inf"
                continue
            if "(ms)" in label:
                v = round(v * 1000, 4)
            result[label] = v

    # 逐轮指标（仅开启轮次屏障时存在）
    rounds = data.get("round") or {}
    for rk, rv in rounds.items():
        m = re.search(r"(\d+)", rk)
        if not m or not isinstance(rv, dict):
            continue
        n = int(m.group(1)) + 1  # 工具 0 基 → 展示第 N 轮
        if isinstance(rv.get("average_ttft"), (int, float)):
            result[f"Round {n} Average TTFT (ms)"] = round(rv["average_ttft"] * 1000, 4)
        if isinstance(rv.get("average_tpot"), (int, float)):
            result[f"Round {n} Average TPOT (ms)"] = round(rv["average_tpot"] * 1000, 4)
        if isinstance(rv.get("cache_hit_rate"), (int, float)):
            result[f"Round {n} Cache Hit Rate"] = rv["cache_hit_rate"]
        if isinstance(rv.get("cache_hit_rate_l1"), (int, float)):
            result[f"Round {n} Cache Hit Rate L1"] = rv["cache_hit_rate_l1"]
        if isinstance(rv.get("cache_hit_rate_l2"), (int, float)):
            result[f"Round {n} Cache Hit Rate L2"] = rv["cache_hit_rate_l2"]
        if isinstance(rv.get("request_count"), (int, float)):
            result[f"Round {n} Request Count"] = rv["request_count"]
    # L2 分层支持标记（全局，非逐轮）：后端是否返回过 L2 分层详情
    if isinstance(s.get("cache_has_l2"), bool):
        result["Cache has L2"] = s["cache_has_l2"]
    return result


def _monitor_error_text(exc: Exception) -> str:
    if isinstance(exc, MonitorClientError):
        return f"{exc.error_code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


def _safe_update_monitor_status(run_id: int, status: str, **kwargs):
    try:
        update_run_monitor_status(run_id, status, **kwargs)
    except Exception:
        # Monitor bookkeeping is optional and must never alter benchmark status.
        pass


async def _await_monitor_control(task, run_id: int):
    if not task:
        return
    try:
        await task
    except Exception as exc:
        _safe_update_monitor_status(
            run_id,
            "stop_failed",
            error=f"monitor control failed: {_monitor_error_text(exc)}",
            finished_at=datetime.now().isoformat(),
        )


def _save_monitor_response(run_id: int, session_id: str, result: dict, warning: str = ""):
    save_run_monitor_result(
        run_id,
        status=result["status"],
        session_id=session_id,
        summary=result.get("summary") or {},
        samples=result.get("samples") or [],
        metadata={
            key: value for key, value in result.items()
            if key not in {"summary", "samples"}
        },
        error="; ".join(filter(None, [warning, str(result.get("warning") or "")])),
        started_at=result.get("started_at"),
        finished_at=result.get("finished_at"),
    )


async def recover_interrupted_monitor_runs():
    """Stop Monitor sessions orphaned by an unclean platform shutdown."""
    for run in get_interrupted_monitor_runs():
        task = get_task(run["task_id"])
        if not task:
            continue
        session_id = str(run.get("monitor_session_id") or "")
        request_id = str(run.get("monitor_request_id") or "")
        try:
            if not session_id and request_id:
                recovered = await recover_monitor_session(task, run, request_id)
                session_id = str(recovered["session_id"])
            if session_id:
                result = await stop_monitor(task, session_id)
                _save_monitor_response(
                    run["id"], session_id, result,
                    "recovered after llm-benchmark-online restart",
                )
            elif task.get("monitor_enabled"):
                _safe_update_monitor_status(
                    run["id"], "unavailable",
                    error="platform restarted before a monitor session was established",
                    finished_at=datetime.now().isoformat(),
                )
        except Exception as exc:
            if isinstance(exc, MonitorClientError) and exc.error_code == "REQUEST_NOT_FOUND":
                _safe_update_monitor_status(
                    run["id"], "unavailable",
                    error="platform restarted but no monitor session exists for request_id",
                    finished_at=datetime.now().isoformat(),
                )
            else:
                _safe_update_monitor_status(
                    run["id"], "stop_failed",
                    session_id=session_id or None,
                    error=f"restart recovery failed: {_monitor_error_text(exc)}",
                    finished_at=datetime.now().isoformat(),
                )
        if run.get("status") == "running":
            update_run(
                run["id"],
                status="failed",
                raw_output=(run.get("raw_output") or "")
                + "\n\n--- llm-benchmark-online restarted during this run ---",
                finished_at=datetime.now().isoformat(),
            )
            update_task_status(run["task_id"], "failed")


async def _wait_for_file_or_process(path: Path, proc: asyncio.subprocess.Process) -> bool:
    while True:
        if path.exists():
            return True
        if proc.returncode is not None:
            return False
        await asyncio.sleep(0.05)


async def _control_monitor_session(
    task: dict,
    run: dict,
    proc: asyncio.subprocess.Process,
    warmup_ready: Path,
    benchmark_continue: Path,
    benchmark_complete: Path,
    gate_timeout: Path = None,
):
    """Own one run's Monitor lifecycle and guarantee at-most-once stop."""
    session_id = None
    stopped = False
    control_warning = ""
    ambiguous_start = False

    async def stop_once():
        nonlocal stopped, control_warning
        if not session_id or stopped:
            return
        stopped = True
        try:
            if gate_timeout and gate_timeout.exists():
                control_warning = (
                    "monitor gate timed out before start completed; "
                    "formal requests may be only partially covered"
                )
            result = await stop_monitor(task, session_id)
            _save_monitor_response(run["id"], session_id, result, control_warning)
        except Exception as exc:
            error = _monitor_error_text(exc)
            if control_warning:
                error = f"{control_warning}; {error}"
            _safe_update_monitor_status(
                run["id"],
                "stop_failed",
                session_id=session_id,
                error=error,
                finished_at=datetime.now().isoformat(),
            )

    try:
        warmup_succeeded = await _wait_for_file_or_process(warmup_ready, proc)
        if not warmup_succeeded:
            _safe_update_monitor_status(
                run["id"],
                "unavailable",
                error="benchmark exited before warmup completed",
                finished_at=datetime.now().isoformat(),
            )
            return

        request_id = str(run.get("monitor_request_id") or uuid.uuid4())
        _safe_update_monitor_status(
            run["id"],
            "pending",
            request_id=request_id,
            error="",
        )
        try:
            session = await start_monitor(task, run, request_id)
            session_id = str(session["session_id"])
            if gate_timeout and gate_timeout.exists():
                control_warning = (
                    "monitor gate timed out before start completed; "
                    "formal requests may be only partially covered"
                )
            _safe_update_monitor_status(
                run["id"],
                "running",
                session_id=session_id,
                error=control_warning,
                started_at=session.get("started_at") or datetime.now().isoformat(),
            )
        except Exception as exc:
            ambiguous_start = (
                isinstance(exc, MonitorClientError)
                and exc.error_code == "MONITOR_REQUEST_TIMEOUT"
            )
            error = _monitor_error_text(exc)
            if gate_timeout and gate_timeout.exists():
                error = (
                    "monitor gate timed out; formal requests continued without "
                    f"confirmed monitoring; {error}"
                )
            _safe_update_monitor_status(
                run["id"],
                "start_failed",
                error=error,
                finished_at=datetime.now().isoformat(),
            )
        finally:
            # Never let an optional Monitor failure block the benchmark.
            benchmark_continue.touch(exist_ok=True)

        # Both start attempts may time out even though Monitor accepted the
        # request. Query by request_id without creating a new profiler and
        # adopt the session while the formal benchmark is running.
        if not session_id and ambiguous_start:
            while True:
                try:
                    recovered = await recover_monitor_session(task, run, request_id)
                    session_id = str(recovered["session_id"])
                    _safe_update_monitor_status(
                        run["id"],
                        "running" if recovered["status"] == "running" else recovered["status"],
                        session_id=session_id,
                        error=(
                            "monitor start response timed out; session recovered by request_id"
                        ),
                        started_at=recovered.get("started_at") or datetime.now().isoformat(),
                    )
                    break
                except MonitorClientError as exc:
                    if exc.error_code not in {
                        "REQUEST_NOT_FOUND",
                        "MONITOR_REQUEST_TIMEOUT",
                        "MONITOR_CONNECTION_FAILED",
                    }:
                        break
                if benchmark_complete.exists() or proc.returncode is not None:
                    break
                await asyncio.sleep(0.5)
            if not session_id:
                # Close the race where the session becomes visible at the same
                # moment the formal benchmark completes.
                try:
                    recovered = await recover_monitor_session(task, run, request_id)
                    session_id = str(recovered["session_id"])
                    _safe_update_monitor_status(
                        run["id"],
                        "running" if recovered["status"] == "running" else recovered["status"],
                        session_id=session_id,
                        error="monitor start response timed out; session recovered by request_id",
                        started_at=recovered.get("started_at") or datetime.now().isoformat(),
                    )
                except MonitorClientError:
                    pass

        if session_id:
            await _wait_for_file_or_process(benchmark_complete, proc)
            await stop_once()
    finally:
        # Covers benchmark failure and user termination before complete.
        await stop_once()


async def execute_run(run_id: int, cancel_event: asyncio.Event = None):
    """Execute a single benchmark run. Returns False if cancelled."""
    run = get_run(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    task = get_task(run["task_id"])
    if not task:
        raise ValueError(f"Task {run['task_id']} not found")

    now = datetime.now().isoformat()
    update_run(run_id, status="running", started_at=now)

    model_name = _get_model_name(task)
    if not model_name:
        model_name = await resolve_model_name(task)

    cmd = build_bench_args(task, run, model_name)
    monitor_enabled = bool(task.get("monitor_enabled", 0))
    handshake_dir = None
    monitor_control_task = None
    if monitor_enabled:
        handshake_dir = Path(tempfile.mkdtemp(prefix=f"llm-benchmark-run-{run_id}-"))
        warmup_ready = handshake_dir / "warmup_ready"
        benchmark_continue = handshake_dir / "benchmark_continue"
        benchmark_complete = handshake_dir / "benchmark_complete"
        gate_timeout = handshake_dir / "gate_timeout"
        cmd.extend([
            "--monitor-warmup-ready-file", str(warmup_ready),
            "--monitor-continue-file", str(benchmark_continue),
            "--monitor-benchmark-complete-file", str(benchmark_complete),
            "--monitor-gate-timeout-file", str(gate_timeout),
            "--monitor-gate-timeout-seconds", "60",
        ])

    # 多轮模式：清理上次可能残留的结果文件
    if run.get("test_mode") == "multi_turn":
        stale_jsonl = f"/tmp/multiturn_run_{run_id}.jsonl"
        if os.path.exists(stale_jsonl):
            os.unlink(stale_jsonl)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BENCH_DIR),
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = proc.pid
        _running_procs[run_id] = (proc, pgid)
        if monitor_enabled:
            monitor_control_task = asyncio.create_task(
                _control_monitor_session(
                    task,
                    run,
                    proc,
                    warmup_ready,
                    benchmark_continue,
                    benchmark_complete,
                    gate_timeout,
                )
            )

        # Stream output in real-time (capture tqdm progress bars that use \r)
        raw_output_chunks = []
        last_update_time = time.time()
        update_interval = 1.0  # Update database every 1 second
        buffer = b""

        while True:
            try:
                # Read small chunks to capture \r updates from tqdm
                chunk = await asyncio.wait_for(proc.stdout.read(1024), timeout=0.1)
                if not chunk:
                    break
                raw_output_chunks.append(chunk)

                # Update database periodically
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    raw_output = b"".join(raw_output_chunks).decode("utf-8", errors="replace")
                    update_run(run_id, raw_output=raw_output)
                    last_update_time = current_time
            except asyncio.TimeoutError:
                # Check if cancelled during idle
                if cancel_event and cancel_event.is_set():
                    break
                continue

        # Wait for process to complete
        await proc.wait()
        _running_procs.pop(run_id, None)
        if monitor_control_task:
            await _await_monitor_control(monitor_control_task, run_id)
        raw_output = b"".join(raw_output_chunks).decode("utf-8", errors="replace")

        # Check if task was cancelled
        if cancel_event and cancel_event.is_set():
            now = datetime.now().isoformat()
            update_run(run_id, status="abort", raw_output=raw_output + "\n\n--- 任务已被用户终止 ---", finished_at=now)
            return False

        # Check for actual benchmark result (mode-aware marker; bench.sh may return 0 even on error)
        test_mode = run.get("test_mode", "normal")
        if test_mode == "multi_turn":
            has_result = "Performance metrics summary:" in raw_output
        else:
            has_result = "===== Serving Benchmark Result =====" in raw_output
        if proc.returncode != 0 or not has_result:
            now = datetime.now().isoformat()
            update_run(run_id, status="failed", raw_output=raw_output, finished_at=now)
            return False

        # Parse results
        parsed = {}
        if test_mode == "multi_turn":
            # bench_multiturn.py 结果在 JSONL（--log-file）里；解析完成后删除原始文件
            # （数据已全部映射进 parsed_result，避免 /tmp 长期累积）
            jsonl_path = f"/tmp/multiturn_run_{run_id}.jsonl"
            parsed = _parse_multiturn_jsonl(jsonl_path)
            if os.path.exists(jsonl_path):
                os.unlink(jsonl_path)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(raw_output)
                tmp_path = f.name
            try:
                parsed_list = parse_benchmark_file(tmp_path)
                if parsed_list:
                    parsed = parsed_list[0]
            finally:
                os.unlink(tmp_path)

        now = datetime.now().isoformat()
        update_run(
            run_id,
            status="done",
            raw_output=raw_output,
            parsed_result=json.dumps(parsed),
            finished_at=now,
        )
        return True

    except Exception as e:
        _running_procs.pop(run_id, None)
        if monitor_control_task:
            await _await_monitor_control(monitor_control_task, run_id)
        elif monitor_enabled:
            _safe_update_monitor_status(
                run_id,
                "unavailable",
                error="benchmark process failed before warmup could start",
                finished_at=datetime.now().isoformat(),
            )
        now = datetime.now().isoformat()
        update_run(run_id, status="failed", raw_output=str(e), finished_at=now)
        return False
    finally:
        if handshake_dir:
            shutil.rmtree(handshake_dir, ignore_errors=True)


# Per-task cancel events: task_id -> asyncio.Event
_cancel_events: dict[int, asyncio.Event] = {}


async def execute_task(task_id: int):
    """Execute all pending runs of a task sequentially."""
    task = get_task(task_id)
    if not task:
        return

    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    update_task_status(task_id, "running")
    runs = get_task_runs(task_id, include_monitor_samples=False)

    all_ok = True
    cancelled = False
    for run in runs:
        if cancel_event.is_set():
            cancelled = True
            break
        if run["status"] not in ("pending",):
            continue
        try:
            ok = await execute_run(run["id"], cancel_event)
        except Exception as e:
            # 参数构造等前置错误不应让 run 卡在 running
            now = datetime.now().isoformat()
            update_run(run["id"], status="failed", raw_output=f"执行失败: {e}", finished_at=now)
            all_ok = False
            ok = False
        if not ok:
            all_ok = False
            # Check if the run was aborted (cancelled)
            updated_run = get_run(run["id"])
            if updated_run and updated_run["status"] == "abort":
                cancelled = True

    # Mark remaining pending runs as abort if cancelled
    if cancelled:
        remaining = get_task_runs(task_id, include_monitor_samples=False)
        now = datetime.now().isoformat()
        for r in remaining:
            if r["status"] == "pending":
                update_run(r["id"], status="abort", raw_output="任务已被用户终止", finished_at=now)
                if task.get("monitor_enabled") and r.get("monitor_status") == "pending":
                    _safe_update_monitor_status(
                        r["id"],
                        "unavailable",
                        error="run aborted before monitor start",
                        finished_at=now,
                    )

    _cancel_events.pop(task_id, None)
    _killing_tasks.discard(task_id)
    status = "abort" if cancelled else ("failed" if not all_ok else "done")
    update_task_status(task_id, status)


def kill_task(task_id: int):
    """Kill the running subprocess and signal task cancellation."""
    # Signal the execute_task loop to stop
    event = _cancel_events.get(task_id)
    if event:
        event.set()

    # Mark this task as being killed (so UI can show "killing" state)
    _killing_tasks.add(task_id)

    # Kill any running subprocess (entire process group)
    for run_id, (proc, pgid) in list(_running_procs.items()):
        run = get_run(run_id)
        if run and run["task_id"] == task_id:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass


# ---------------------------------------------------------------------------
# Eval execution
# ---------------------------------------------------------------------------

EVAL_SCRIPT = str(BENCH_DIR / "eval_scope.py")

EVAL_DATASET_ARGS = {
    'mmlu': '{"mmlu": {"few_shot_num": 5}}',
    'gsm8k': '{"gsm8k": {"few_shot_num": 8}}',
    'bfcl_v3': '{"bfcl_v3": {"extra_params": {"underscore_to_dot": true, "is_fc_model": true}}}',
    'needle_haystack': '{"needle_haystack": {"extra_params": {"context_lengths_min": 1000, "context_lengths_max": 1000000, "context_lengths_num_intervals": 20, "document_depth_percent_min": 0, "document_depth_percent_max": 100, "document_depth_percent_intervals": 20}}}',
}

EVAL_GEN_CONFIGS = {
    'mmlu': '{"max_tokens": 8192, "temperature": 0.7, "top_p": 0.95}',
    'gsm8k': '{"max_tokens": 8192, "temperature": 0.0}',
    'bfcl_v3': '{"temperature": 0, "max_tokens": 32000, "parallel_tool_calls": true}',
    'longbench_v2': '{"batch_size": 1, "temperature": 0.0}',
    'needle_haystack': '{"batch_size": 1, "temperature": 0.0}',
}

# Judge model configuration for datasets that need LLM-based evaluation
EVAL_JUDGE_CONFIGS = {
    'needle_haystack': {
        'strategy': 'llm',
        'model_args': '{}',  # Will be filled dynamically with the same API endpoint
    },
}


def build_eval_args(task: dict, run: dict, model_name: str = "") -> list[str]:
    """Construct arg list for eval_scope.py from task + run config."""
    if task["connection_mode"] == "base_url":
        api_url = task["base_url"].rstrip("/") + "/v1/chat/completions"
    else:
        api_url = f"http://{task['host']}:{task['port']}/v1/chat/completions"

    model_id = model_name or task.get("request_model") or "default"
    dataset_name = run["dataset_name"]
    limit = run.get("limit_val") or 0
    batch_size = run.get("batch_size") or 1

    cmd = [
        "python3", EVAL_SCRIPT,
        "--api-url", api_url,
        "--model-id", model_id,
        "--datasets", json.dumps([dataset_name]),
        "--eval-batch-size", str(batch_size),
        "--output-dir", str(BENCH_DIR / "eval_outputs"),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if task.get("dataset_path"):
        cmd.extend(["--dataset-dir", task["dataset_path"]])
    if task.get("api_key"):
        cmd.extend(["--api-key", task["api_key"]])

    ds_args = EVAL_DATASET_ARGS.get(dataset_name, "{}")
    cmd.extend(["--dataset-args", ds_args])

    ds_gen = json.loads(EVAL_GEN_CONFIGS.get(dataset_name, "{}"))

    # If think mode is enabled, inject extra_body into generation_config
    extra_args = json.loads(task.get("extra_args") or "[]")
    if "--enable-think" in extra_args:
        ds_gen.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["thinking"] = True

    cmd.extend(["--gen-config", json.dumps(ds_gen)])

    # Add judge model configuration if needed
    if dataset_name in EVAL_JUDGE_CONFIGS:
        judge_cfg = EVAL_JUDGE_CONFIGS[dataset_name]
        cmd.extend(["--judge-strategy", judge_cfg["strategy"]])
        # Use the same API endpoint for judge model
        judge_model_args = {
            "model_id": model_id,  # Use the same model as the main model
            "api_url": api_url,
        }
        if task.get("api_key"):
            judge_model_args["api_key"] = task["api_key"]
        cmd.extend(["--judge-model-args", json.dumps(judge_model_args)])

    return cmd


async def execute_eval_run(run_id: int, cancel_event: asyncio.Event = None):
    """Execute a single eval run (one dataset). Returns False if cancelled/failed."""
    run = get_run(run_id)
    task = get_task(run["task_id"])

    now = datetime.now().isoformat()
    update_run(run_id, status="running", started_at=now)

    model_name = _get_model_name(task)
    if not model_name:
        model_name = await resolve_model_name(task)

    cmd = build_eval_args(task, run, model_name)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BENCH_DIR),
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = proc.pid
        _running_procs[run_id] = (proc, pgid)

        raw_output_chunks = []
        last_update_time = time.time()
        update_interval = 1.0

        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(1024), timeout=0.1)
                if not chunk:
                    break
                raw_output_chunks.append(chunk)
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    raw_output = b"".join(raw_output_chunks).decode("utf-8", errors="replace")
                    update_run(run_id, raw_output=raw_output)
                    last_update_time = current_time
            except asyncio.TimeoutError:
                if cancel_event and cancel_event.is_set():
                    break
                continue

        await proc.wait()
        _running_procs.pop(run_id, None)
        raw_output = b"".join(raw_output_chunks).decode("utf-8", errors="replace")

        if cancel_event and cancel_event.is_set():
            now = datetime.now().isoformat()
            update_run(run_id, status="abort", raw_output=raw_output + "\n\n--- 任务已被用户终止 ---", finished_at=now)
            return False

        parsed = _parse_eval_result(raw_output)
        now = datetime.now().isoformat()
        status = "done" if parsed else "failed"
        update_run(run_id, status=status, raw_output=raw_output,
                   parsed_result=json.dumps(parsed) if parsed else "{}",
                   finished_at=now)
        return status == "done"

    except Exception as e:
        _running_procs.pop(run_id, None)
        now = datetime.now().isoformat()
        update_run(run_id, status="failed", raw_output=str(e), finished_at=now)
        return False


def _parse_eval_result(raw_output: str) -> dict | None:
    marker_start = "===EVAL_RESULT==="
    marker_end = "===EVAL_RESULT_END==="
    idx = raw_output.find(marker_start)
    if idx == -1:
        return None
    start = idx + len(marker_start)
    end = raw_output.find(marker_end, start)
    if end == -1:
        return None
    try:
        return json.loads(raw_output[start:end].strip())
    except json.JSONDecodeError:
        return None


async def execute_eval_task(task_id: int):
    """Execute all eval runs of a task sequentially (one per dataset)."""
    task = get_task(task_id)
    if not task:
        return

    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    update_task_status(task_id, "running")
    runs = get_task_runs(task_id, include_monitor_samples=False)

    all_ok = True
    cancelled = False
    for run in runs:
        if cancel_event.is_set():
            cancelled = True
            break
        if run["status"] not in ("pending",):
            continue
        ok = await execute_eval_run(run["id"], cancel_event)
        if not ok:
            all_ok = False
            updated_run = get_run(run["id"])
            if updated_run and updated_run["status"] == "abort":
                cancelled = True

    if cancelled:
        remaining = get_task_runs(task_id, include_monitor_samples=False)
        now = datetime.now().isoformat()
        for r in remaining:
            if r["status"] == "pending":
                update_run(r["id"], status="abort", raw_output="任务已被用户终止", finished_at=now)

    _cancel_events.pop(task_id, None)
    _killing_tasks.discard(task_id)
    status = "abort" if cancelled else ("failed" if not all_ok else "done")
    update_task_status(task_id, status)


# ---------------------------------------------------------------------------
# API Test execution
# ---------------------------------------------------------------------------

from api_test_executor import APITestExecutor
from db import create_api_test_config, get_api_test_config, create_api_test_result, get_api_test_results
from db import create_compat_test_config, get_compat_test_config


async def execute_api_test_run(run_id: int, cancel_event: asyncio.Event = None):
    """Execute a single API test run. Returns False if cancelled/failed."""
    run = get_run(run_id)
    task = get_task(run["task_id"])

    now = datetime.now().isoformat()
    update_run(run_id, status="running", started_at=now)

    # Build base URL
    if task["connection_mode"] == "base_url":
        base_url = task["base_url"]
    else:
        base_url = f"http://{task['host']}:{task['port']}"

    # Get API test config
    config = get_api_test_config(run_id)
    if not config:
        config = {}

    # Build executor
    executor = APITestExecutor(
        base_url=base_url,
        model=config.get("model") or task.get("request_model") or "default",
        reasoning_model=config.get("reasoning_model") or "",
        fc_model=config.get("fc_model") or "",
        api_key=task.get("api_key") or "",
        enable_fc=bool(config.get("enable_fc_test")),
        enable_n=bool(config.get("enable_n_test")),
        n_value=config.get("n_value") or 3,
    )

    try:
        # Get test cases
        test_cases = executor.get_test_cases_list()

        # Run tests one by one and save results incrementally
        results = []
        for tc in test_cases:
            if cancel_event and cancel_event.is_set():
                break

            result = await executor.run_single_test(tc)
            results.append(result)

            # Save result immediately for real-time progress
            create_api_test_result(
                run_id=run_id,
                test_name=result.test_name,
                category=result.category,
                status=result.status,
                http_code=result.http_code,
                response_time_ms=result.response_time_ms,
                response_body=result.response_body[:10000] if result.response_body else "",
                error_message=result.error_message,
                assertion_details=result.assertion_details,
            )

            # Update raw_output for polling
            partial_summary = executor.get_summary(results)
            update_run(run_id, raw_output=json.dumps(partial_summary, ensure_ascii=False))

        # Check cancellation
        if cancel_event and cancel_event.is_set():
            now = datetime.now().isoformat()
            update_run(run_id, status="abort", raw_output="任务已被用户终止", finished_at=now)
            return False

        # Final summary
        summary = executor.get_summary(results)
        raw_output = json.dumps(summary, ensure_ascii=False, indent=2)
        now = datetime.now().isoformat()
        status = "done" if summary["failed"] == 0 and summary["error"] == 0 else "done"
        update_run(run_id, status=status, raw_output=raw_output,
                   parsed_result=json.dumps(summary), finished_at=now)
        return True

    except Exception as e:
        now = datetime.now().isoformat()
        update_run(run_id, status="failed", raw_output=str(e), finished_at=now)
        return False


async def execute_api_test_task(task_id: int):
    """Execute all API test runs of a task."""
    task = get_task(task_id)
    if not task:
        return

    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    update_task_status(task_id, "running")
    runs = get_task_runs(task_id, include_monitor_samples=False)

    all_ok = True
    cancelled = False
    for run in runs:
        if cancel_event.is_set():
            cancelled = True
            break
        if run["status"] not in ("pending",):
            continue
        ok = await execute_api_test_run(run["id"], cancel_event)
        if not ok:
            all_ok = False
            updated_run = get_run(run["id"])
            if updated_run and updated_run["status"] == "abort":
                cancelled = True

    if cancelled:
        remaining = get_task_runs(task_id, include_monitor_samples=False)
        now = datetime.now().isoformat()
        for r in remaining:
            if r["status"] == "pending":
                update_run(r["id"], status="abort", raw_output="任务已被用户终止", finished_at=now)

    _cancel_events.pop(task_id, None)
    _killing_tasks.discard(task_id)
    status = "abort" if cancelled else ("failed" if not all_ok else "done")
    update_task_status(task_id, status)


# ---------------------------------------------------------------------------
# Compat Test execution (OpenAI API compatibility test)
# ---------------------------------------------------------------------------

COMPAT_TEST_SCRIPT = str(BENCH_DIR / "openai_api_check/main.py")


def build_compat_test_args(task: dict, config: dict) -> list[str]:
    """Construct arg list for openai_api_check/main.py."""
    if task["connection_mode"] == "base_url":
        base_url = task["base_url"].rstrip("/")
    else:
        base_url = f"http://{task['host']}:{task['port']}"

    model = task.get("request_model") or "default"
    test_level = config.get("test_level", "standard")
    framework = config.get("framework", "standard")

    cmd = [
        "python3", COMPAT_TEST_SCRIPT,
        "--api-url", base_url,
        "--model", model,
        "--level", test_level,
        "--framework", framework,
    ]

    if config.get("include_extensions"):
        cmd.append("--include-framework-extensions")
    if config.get("enable_dangerous"):
        cmd.append("--enable-dangerous-tests")

    return cmd


def _parse_compat_test_result(raw_output: str) -> dict:
    """Parse test summary and individual results from openai_api_check output."""
    import re
    result = {"total": 0, "passed": 0, "failed": 0, "avg_time_ms": 0, "tests": []}

    # Parse summary line: "总测试数: 300\n通过: 280 (93.3%)\n失败: 20 (6.7%)"
    total_match = re.search(r"总测试数:\s*(\d+)", raw_output)
    passed_match = re.search(r"通过:\s*(\d+)", raw_output)
    failed_match = re.search(r"失败:\s*(\d+)", raw_output)

    if total_match:
        result["total"] = int(total_match.group(1))
    if passed_match:
        result["passed"] = int(passed_match.group(1))
    if failed_match:
        result["failed"] = int(failed_match.group(1))

    # Parse average response time
    avg_match = re.search(r"平均响应时间:\s*([\d.]+)s", raw_output)
    if avg_match:
        result["avg_time_ms"] = int(float(avg_match.group(1)) * 1000)

    # Parse individual test results from the execution log
    # Each test block starts with "测试: <name>" and ends with result line
    test_pattern = re.compile(
        r"测试:\s*(?P<name>\S+)\n"
        r".*?描述:\s*(?P<description>[^\n]+)\n"
        r".*?危险等级:\s*(?P<category>[^\n]+)\n"
        r".*?"  # Skip separator line and HTTP request log
        r"(?P<result>[✓✗⚠])\s*测试(?P<status>通过|失败|异常)"
        r"(?:\s*\((?P<time>[\d.]+)s\))?"
        r"(?:[^\n]*?Error code:\s*\d+\s*-\s*(?P<error>[^\n]+))?",
        re.DOTALL
    )

    current_category = ""
    category_match = re.finditer(r"执行(\S+)测试\s*\(\d+\s*个\)", raw_output)
    category_positions = [(m.start(), m.group(1)) for m in category_match]

    for match in test_pattern.finditer(raw_output):
        test_pos = match.start()
        # Determine which category this test belongs to
        for pos, cat_name in reversed(category_positions):
            if pos < test_pos:
                current_category = cat_name
                break

        status_text = match.group("status")
        status = "pass" if status_text == "通过" else ("fail" if match.group("result") == "✗" else "error")

        test_info = {
            "name": match.group("name"),
            "description": match.group("description"),
            "category": current_category,
            "status": status,
            "time_sec": float(match.group("time")) if match.group("time") else 0,
            "error": match.group("error") if match.group("error") else None,
        }
        result["tests"].append(test_info)

    return result


async def execute_compat_test_run(run_id: int, cancel_event: asyncio.Event = None):
    """Execute a single compat test run. Returns False if cancelled/failed."""
    run = get_run(run_id)
    task = get_task(run["task_id"])

    now = datetime.now().isoformat()
    update_run(run_id, status="running", started_at=now)

    config = get_compat_test_config(run_id) or {}
    cmd = build_compat_test_args(task, config)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BENCH_DIR),
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = proc.pid
        _running_procs[run_id] = (proc, pgid)

        raw_output_chunks = []
        last_update_time = time.time()
        update_interval = 1.0

        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(1024), timeout=0.1)
                if not chunk:
                    break
                raw_output_chunks.append(chunk)
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    raw_output = b"".join(raw_output_chunks).decode("utf-8", errors="replace")
                    update_run(run_id, raw_output=raw_output)
                    last_update_time = current_time
            except asyncio.TimeoutError:
                if cancel_event and cancel_event.is_set():
                    break
                continue

        await proc.wait()
        _running_procs.pop(run_id, None)
        raw_output = b"".join(raw_output_chunks).decode("utf-8", errors="replace")

        if cancel_event and cancel_event.is_set():
            now = datetime.now().isoformat()
            update_run(run_id, status="abort", raw_output=raw_output + "\n\n--- 任务已被用户终止 ---", finished_at=now)
            return False

        # Parse result - check if we got valid test output
        parsed = _parse_compat_test_result(raw_output)
        now = datetime.now().isoformat()

        # Status: done if we parsed results (total > 0), failed only if no results
        if parsed.get("total", 0) > 0:
            status = "done"
        else:
            status = "failed"

        update_run(run_id, status=status, raw_output=raw_output,
                   parsed_result=json.dumps(parsed), finished_at=now)
        return status == "done"

    except Exception as e:
        _running_procs.pop(run_id, None)
        now = datetime.now().isoformat()
        update_run(run_id, status="failed", raw_output=str(e), finished_at=now)
        return False


async def execute_compat_test_task(task_id: int):
    """Execute all compat test runs of a task."""
    task = get_task(task_id)
    if not task:
        return

    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    update_task_status(task_id, "running")
    runs = get_task_runs(task_id, include_monitor_samples=False)

    all_ok = True
    cancelled = False
    for run in runs:
        if cancel_event.is_set():
            cancelled = True
            break
        if run["status"] not in ("pending",):
            continue
        ok = await execute_compat_test_run(run["id"], cancel_event)
        if not ok:
            all_ok = False
            updated_run = get_run(run["id"])
            if updated_run and updated_run["status"] == "abort":
                cancelled = True

    if cancelled:
        remaining = get_task_runs(task_id, include_monitor_samples=False)
        now = datetime.now().isoformat()
        for r in remaining:
            if r["status"] == "pending":
                update_run(r["id"], status="abort", raw_output="任务已被用户终止", finished_at=now)

    _cancel_events.pop(task_id, None)
    _killing_tasks.discard(task_id)
    status = "abort" if cancelled else ("failed" if not all_ok else "done")
    update_task_status(task_id, status)
