import asyncio
import csv
import io
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db import (
    create_run,
    create_task,
    create_template,
    delete_template,
    delete_task,
    get_run,
    get_task,
    get_task_runs,
    get_template,
    init_db,
    list_tasks,
    list_templates,
    update_task_status,
    update_template,
    create_api_test_config,
    get_api_test_config,
    get_api_test_results,
    create_compat_test_config,
    get_compat_test_config,
)
from runner import execute_task, execute_eval_task, execute_api_test_task, execute_compat_test_task, kill_task, recover_interrupted_monitor_runs, _killing_tasks
from monitor_config import MonitorConfigError, parse_monitor_config
from monitor_client import MonitorClientError, _base_url

# ── Load configuration ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def _load_config() -> dict:
    """Load config.yaml using a simple parser (no PyYAML dependency)."""
    cfg = {
        "model_dirs": ["/nfs/ofs-llab-cold/model", "/nfs/ofs-luban-data/model"],
        "default_model_path": "/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V4-Flash",
        "dataset_dir": "/home/luban",
        "datasets": [{"name": "ShareGPT V3 (GPU)", "filename": "ShareGPT_V3_unfiltered_cleaned_split.json"}],
        "default_dataset": "ShareGPT V3 (GPU)",
    }
    if not CONFIG_PATH.exists():
        return cfg
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            user_cfg = yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: minimal YAML parsing for our flat structure
        user_cfg = {}
        current_list_key = None
        with open(CONFIG_PATH) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line.startswith("  - ") and current_list_key:
                    val = stripped[2:].strip().strip('"').strip("'")
                    if current_list_key not in user_cfg:
                        user_cfg[current_list_key] = []
                    user_cfg[current_list_key].append(val)
                elif ":" in line and not line.startswith(" "):
                    current_list_key = None
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if not val:
                        current_list_key = key
                    else:
                        user_cfg[key] = val
    cfg.setdefault("eval_dataset_dir", "/home/luban/ifx_llm_dist_auto_test/dataset")
    cfg.update({k: v for k, v in user_cfg.items() if v is not None})
    return cfg


CONFIG = _load_config()
MODEL_BASE_DIRS = CONFIG["model_dirs"]
MONITOR_HEALTH_PROXY_TIMEOUT_SECONDS = 15.0

app = FastAPI(title="LLM Benchmark Platform")

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def to_beijing_time(utc_time_str: str) -> str:
    """Convert UTC time string to Beijing time (UTC+8).

    Input format: '2026-04-29 08:34:42' or '2026-04-29T08:34:42.123456'
    Output format: '2026-04-29 16:34:42'
    """
    if not utc_time_str:
        return ""
    try:
        # Parse the time string (handle both formats)
        if 'T' in utc_time_str:
            dt = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
        else:
            dt = datetime.strptime(utc_time_str[:19], "%Y-%m-%d %H:%M:%S")

        # If naive datetime, assume it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to Beijing time
        beijing_tz = timezone(timedelta(hours=8))
        beijing_time = dt.astimezone(beijing_tz)

        return beijing_time.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return utc_time_str[:19] if len(utc_time_str) >= 19 else utc_time_str


# Register the filter with Jinja2
templates.env.filters["beijing_time"] = to_beijing_time


def scan_models() -> list[dict]:
    """Scan model directories and return list of {name, path} dicts."""
    models = []
    for base in MODEL_BASE_DIRS:
        if not os.path.isdir(base):
            continue
        for org in sorted(os.listdir(base)):
            org_path = os.path.join(base, org)
            if not os.path.isdir(org_path) or org.startswith("."):
                continue
            for model_name in sorted(os.listdir(org_path)):
                model_path = os.path.join(org_path, model_name)
                if os.path.isdir(model_path) and not model_name.startswith("."):
                    models.append({
                        "name": f"{org}/{model_name}",
                        "path": model_path,
                    })
    return models


# Cache model list at startup
MODEL_LIST = []

# Short display names for result fields to reduce table width
FIELD_SHORT = {
    "Traffic request rate": "请求率",
    "Successful requests": "成功请求",
    "Max request concurrency": "最大并发",
    "Mean TTFT (ms)": "Avg TTFT",
    "Median TTFT (ms)": "P50 TTFT",
    "P95 TTFT (ms)": "P95 TTFT",
    "P99 TTFT (ms)": "P99 TTFT",
    "Mean TPOT (ms)": "Avg TPOT",
    "Median TPOT (ms)": "P50 TPOT",
    "P95 TPOT (ms)": "P95 TPOT",
    "P99 TPOT (ms)": "P99 TPOT",
    "Mean ITL (ms)": "Avg ITL",
    "Median ITL (ms)": "P50 ITL",
    "P95 ITL (ms)": "P95 ITL",
    "P99 ITL (ms)": "P99 ITL",
    "Mean E2E Latency (ms)": "Avg E2E",
    "Median E2E Latency (ms)": "P50 E2E",
    "Request throughput (req/s)": "req/s",
    "Input token throughput (tok/s)": "输入tok/s",
    "Output token throughput (tok/s)": "输出tok/s",
    "Total token throughput (tok/s)": "总tok/s",
    "Concurrency": "并发数",
    "Cache hit rate": "缓存命中",
    "Cache hit rate L1": "缓存命中L1",
    "Cache hit rate L2": "缓存命中L2",
    "Accept length": "Accept长度",
    "Total input tokens": "总输入token",
    "Total generated tokens": "总生成token",
    "Benchmark duration (s)": "耗时(s)",
    "Total cached tokens": "缓存token数",
    # Multi-turn test fields (bench_multiturn.py)
    "Total requests": "总请求数",
    "Average TTFT (ms)": "Avg TTFT",
    "P90 TTFT (ms)": "P90 TTFT",
    "P99 TTFT (ms)": "P99 TTFT",
    "Median TTFT (ms)": "P50 TTFT",
    "Max TTFT (ms)": "Max TTFT",
    "Average TPOT (ms)": "Avg TPOT",
    "Median TPOT (ms)": "P50 TPOT",
    "P90 TPOT (ms)": "P90 TPOT",
    "P99 TPOT (ms)": "P99 TPOT",
    "Max TPOT (ms)": "Max TPOT",
    "Average Latency (ms)": "Avg延迟",
    "P90 Latency (ms)": "P90延迟",
    "P99 Latency (ms)": "P99延迟",
    "Median Latency (ms)": "P50延迟",
    "Input token throughput (tok/s)": "输入tok/s",
    "Output token throughput (tok/s)": "输出tok/s",
}


DATASET_LIST = []


def _task_display_labels(task: dict, runs: list | None = None) -> dict:
    """Return consistent user-facing type/mode labels for every task surface."""
    task_type = task.get("task_type", "bench")
    if task_type == "eval":
        return {"type": "精度测试", "mode": "模型精度评测"}
    if task_type == "api_test":
        return {"type": "API 测试", "mode": "OpenAI API 功能测试"}
    if task_type == "compat_test":
        return {"type": "兼容性测试", "mode": "接口兼容性验证"}

    first = (runs or [{}])[0] if runs else {}
    test_mode = first.get("test_mode", "normal")
    if test_mode == "shared_prefix":
        return {"type": "公共前缀性能测试", "mode": "公共前缀缓存压测"}
    if test_mode == "multi_turn":
        if first.get("mt_workload_mode", "synthetic") == "timestamp_replay":
            return {
                "type": "多轮对话性能测试",
                "mode": "时间戳响应门控回放",
            }
        if first.get("mt_workload_mode", "synthetic") == "replay":
            return {
                "type": "多轮对话性能测试",
                "mode": "响应驱动历史回放",
            }
        return {
            "type": "多轮对话性能测试",
            "mode": "长度合成多轮对话",
        }
    return {"type": "普通性能测试", "mode": "普通性能压测"}


@app.on_event("startup")
async def startup():
    global MODEL_LIST, DATASET_LIST
    init_db()
    await recover_interrupted_monitor_runs()
    MODEL_LIST = scan_models()
    # Build dataset list from config
    ds_dir = CONFIG["dataset_dir"]
    for ds in CONFIG["datasets"]:
        DATASET_LIST.append({
            "name": ds["name"] if isinstance(ds, dict) else ds,
            "path": os.path.join(ds_dir, ds["filename"] if isinstance(ds, dict) else ds),
        })


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, page: int = 1):
    data = list_tasks(page=page, per_page=20, task_type='bench')
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, **data},
    )


@app.get("/task/create", response_class=HTMLResponse)
async def task_create_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="task_create.html",
        context={"request": request},
    )


@app.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail_page(request: Request, task_id: int):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Route to different detail pages based on task_type
    task_type = task.get("task_type", "bench")
    if task_type == "compat_test":
        runs = get_task_runs(task_id, include_monitor_samples=False)
        return templates.TemplateResponse(
            request=request,
            name="compat_test_detail.html",
            context={"request": request, "task": task, "runs": runs},
        )
    elif task_type == "api_test":
        runs = get_task_runs(task_id, include_monitor_samples=False)
        return templates.TemplateResponse(
            request=request,
            name="api_test_detail.html",
            context={"request": request, "task": task, "runs": runs},
        )

    # Default: bench or eval task
    runs = get_task_runs(task_id, include_monitor_samples=False)
    labels = _task_display_labels(task, runs)
    task["display_type"] = labels["type"]
    task["display_mode"] = labels["mode"]

    # Collect all parsed field keys across all runs, with short display names
    result_fields = []       # list of (full_name, short_label) tuples
    seen = set()
    for r in runs:
        if r["parsed_result"]:
            try:
                parsed = json.loads(r["parsed_result"])
                r["parsed"] = parsed
                for k in parsed:
                    if k not in seen:
                        seen.add(k)
                        result_fields.append((k, FIELD_SHORT.get(k, k)))
            except (json.JSONDecodeError, TypeError):
                r["parsed"] = {}
        else:
            r["parsed"] = {}

        # 多轮：整理逐轮指标（开启轮次屏障时存在）
        if r.get("test_mode") == "multi_turn" and r.get("parsed"):
            import re
            rounds = {}
            for k, v in r["parsed"].items():
                m = re.match(r"^Round (\d+) (.*)$", k)
                if m:
                    rounds.setdefault(int(m.group(1)), {})[m.group(2)] = v
            r["mt_rounds"] = [
                {"n": n, **rounds[n]} for n in sorted(rounds)
            ] if rounds else None

        # Pre-compute accuracy for eval runs
        if task.get("task_type") == "eval":
            ds_name = r.get("dataset_name", "")
            if r.get("parsed") and ds_name:
                ds_data = r["parsed"].get("datasets", {}).get(ds_name, {})
                score = ds_data.get("score")
                if isinstance(score, (int, float)):
                    r["accuracy"] = f"{score * 100:.2f}%" if score <= 1 else str(score)
                else:
                    r["accuracy"] = "-"
            else:
                r["accuracy"] = "-"
            # Use duration from parsed result (more reliable than DB timestamps)
            eval_dur = r.get("parsed", {}).get("duration_sec")
            if isinstance(eval_dur, (int, float)) and eval_dur > 0:
                r["eval_duration"] = f"{eval_dur:.0f}s"
            else:
                r["eval_duration"] = None

        # Compute duration from DB timestamps
        if r.get("started_at") and r.get("finished_at"):
            try:
                from datetime import datetime
                t0 = datetime.fromisoformat(str(r["started_at"]))
                t1 = datetime.fromisoformat(str(r["finished_at"]))
                r["duration_sec"] = int((t1 - t0).total_seconds())
            except (ValueError, TypeError):
                r["duration_sec"] = None
        else:
            r["duration_sec"] = None

    # Progress info
    total_runs = len(runs)
    done_runs = sum(1 for r in runs if r["status"] == "done")
    failed_runs = sum(1 for r in runs if r["status"] == "failed")
    running_now = next((r for r in runs if r["status"] == "running"), None)

    # Detect iterations: group runs by unique parameter combination
    iteration_groups = []
    if task.get("task_type") != "eval" and runs:
        # Build a key for each unique parameter combination
        combo_map = {}
        for i, r in enumerate(runs):
            key = (r.get("concurrency"), r.get("random_input_len"),
                   r.get("random_output_len"), r.get("request_rate"))
            if key not in combo_map:
                combo_map[key] = []
            combo_map[key].append(i)

        # Check if all combinations have the same count > 1
        combo_counts = [len(v) for v in combo_map.values()]
        if combo_counts and min(combo_counts) == max(combo_counts) and min(combo_counts) > 1:
            iterations = min(combo_counts)
            # Build iteration groups: list of (combo_idx, iteration_idx, run_idx)
            combo_keys = list(combo_map.keys())
            for combo_idx, key in enumerate(combo_keys):
                for iter_idx, run_idx in enumerate(combo_map[key]):
                    runs[run_idx]["_combo_idx"] = combo_idx
                    runs[run_idx]["_iter_idx"] = iter_idx
            iteration_groups = combo_keys
            iterations_count = iterations
        else:
            iterations_count = 1
    else:
        iterations_count = 1

    # The page renders monitor summaries as first-class HTML sections. Keep
    # the initial payload small; full time-series data remains available from
    # the dedicated JSON/CSV/raw download endpoints.
    monitor_result_payloads = []
    if task.get("monitor_enabled"):
        for run in runs:
            monitor_result_payloads.append({
                "id": run["id"],
                "monitor_status": run.get("monitor_status", "pending"),
                "monitor_error": run.get("monitor_error", ""),
                "monitor_started_at": run.get("monitor_started_at"),
                "monitor_finished_at": run.get("monitor_finished_at"),
                "monitor_session_id": run.get("monitor_session_id"),
                "monitor_summary": run.get("monitor_summary") or {},
                "monitor_metadata": run.get("monitor_metadata") or {},
                "monitor_samples": [],
            })

    return templates.TemplateResponse(
        request=request,
        name="task_detail.html",
        context={
            "request": request,
            "task": task,
            "runs": runs,
            "result_fields": result_fields,
            "total_runs": total_runs,
            "done_runs": done_runs,
            "failed_runs": failed_runs,
            "running_now": running_now,
            "iterations_count": iterations_count,
            "monitor_result_payloads": monitor_result_payloads,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, page: int = 1):
    data = list_tasks(page=page, per_page=20, task_type='bench')
    for task in data["tasks"]:
        labels = _task_display_labels(task, get_task_runs(task["id"]))
        task["test_category"] = labels["type"]
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"request": request, **data},
    )


@app.get("/eval/create", response_class=HTMLResponse)
async def eval_create_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="eval_create.html",
        context={"request": request},
    )


@app.get("/eval/history", response_class=HTMLResponse)
async def eval_history_page(request: Request, page: int = 1):
    data = list_tasks(page=page, per_page=20, task_type='eval')
    return templates.TemplateResponse(
        request=request,
        name="eval_history.html",
        context={"request": request, **data},
    )


@app.get("/api-test/create", response_class=HTMLResponse)
async def api_test_create_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="api_test_create.html",
        context={"request": request},
    )


@app.get("/api-test/history", response_class=HTMLResponse)
async def api_test_history_page(request: Request, page: int = 1):
    data = list_tasks(page=page, per_page=20, task_type='api_test')
    return templates.TemplateResponse(
        request=request,
        name="api_test_history.html",
        context={"request": request, **data},
    )


@app.get("/api-test/{task_id}", response_class=HTMLResponse)
async def api_test_detail_page(request: Request, task_id: int):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    runs = get_task_runs(task_id, include_monitor_samples=False)
    return templates.TemplateResponse(
        request=request,
        name="api_test_detail.html",
        context={"request": request, "task": task, "runs": runs},
    )


@app.get("/compat-test/create", response_class=HTMLResponse)
async def compat_test_create_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="compat_test_create.html",
        context={"request": request},
    )


@app.get("/compat-test/history", response_class=HTMLResponse)
async def compat_test_history_page(request: Request, page: int = 1):
    data = list_tasks(page=page, per_page=20, task_type='compat_test')
    return templates.TemplateResponse(
        request=request,
        name="compat_test_history.html",
        context={"request": request, **data},
    )


@app.get("/compat-test/{task_id}", response_class=HTMLResponse)
async def compat_test_detail_page(request: Request, task_id: int):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    runs = get_task_runs(task_id, include_monitor_samples=False)
    return templates.TemplateResponse(
        request=request,
        name="compat_test_detail.html",
        context={"request": request, "task": task, "runs": runs},
    )


# ---------------------------------------------------------------------------
# API routes — models
# ---------------------------------------------------------------------------


@app.get("/api/models")
async def api_models():
    return JSONResponse(MODEL_LIST)


@app.get("/api/datasets")
async def api_datasets():
    return JSONResponse(DATASET_LIST)


# ---------------------------------------------------------------------------
# API routes — templates
# ---------------------------------------------------------------------------


@app.get("/api/templates")
async def api_list_templates():
    templates = list_templates()
    for t in templates:
        try:
            t["config"] = json.loads(t["config"])
        except (json.JSONDecodeError, TypeError):
            t["config"] = {}
    return JSONResponse(templates)


@app.post("/api/templates")
async def api_create_template(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="模板名称不能为空")
    config = body.get("config", {})
    if not config:
        raise HTTPException(status_code=400, detail="模板配置不能为空")
    try:
        tid = create_template(name, config)
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="模板名称已存在")
        raise
    return JSONResponse({"id": tid, "name": name})


@app.put("/api/templates/{template_id}")
async def api_update_template(template_id: int, request: Request):
    tmpl = get_template(template_id)
    if not tmpl:
        raise HTTPException(status_code=404, detail="模板不存在")
    body = await request.json()
    update_template(
        template_id,
        name=body.get("name"),
        config=body.get("config"),
    )
    return JSONResponse({"ok": True})


@app.delete("/api/templates/{template_id}")
async def api_delete_template(template_id: int):
    tmpl = get_template(template_id)
    if not tmpl:
        raise HTTPException(status_code=404, detail="模板不存在")
    delete_template(template_id)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# API routes — traffic check
# ---------------------------------------------------------------------------

METRICS_TIMEOUT = 5  # seconds for http requests


def _extract_metric(metrics: str, name: str) -> float:
    """Extract sum of values from a Prometheus metric line."""
    total = 0.0
    for line in metrics.splitlines():
        if line.startswith(name + "{") and "_bucket" not in line and "_sum" not in line:
            try:
                parts = line.rsplit("}", 1)
                if len(parts) == 2:
                    total += float(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return total


async def check_traffic(host: str, port: int, duration: int = 30, skip: bool = False) -> dict:
    """Check if a vllm/sglang service at host:port has active traffic."""
    if skip:
        return {"ok": True, "message": "已跳过流量检测", "log": ["用户选择跳过流量检测"]}

    metrics_url = f"http://{host}:{port}/metrics"
    log = []

    async with httpx.AsyncClient(timeout=METRICS_TIMEOUT) as client:
        # Initial fetch
        try:
            resp = await client.get(metrics_url)
            metrics = resp.text
        except Exception as e:
            log.append(f"无法获取 metrics: {e}")
            return {"ok": True, "message": "无法获取 metrics，跳过校验", "log": log}

        # Detect service type
        is_vllm = "vllm:" in metrics
        is_sglang = "sglang_" in metrics or "sgl_" in metrics

        if not is_vllm and not is_sglang:
            log.append("未检测到 vllm 或 sglang 关键字，跳过校验")
            return {"ok": True, "message": "非 vllm/sglang 服务，跳过校验", "log": log}

        service_type = "vllm" if is_vllm else "sglang"
        log.append(f"检测到 {service_type} 服务，开始流量校验...")

        # Extract metrics based on service type
        if is_vllm:
            initial_counter = _extract_metric(metrics, "vllm:e2e_request_latency_seconds_count")
            running = _extract_metric(metrics, "vllm:num_requests_running")
            waiting = _extract_metric(metrics, "vllm:num_requests_waiting")
        else:
            # sglang metrics
            initial_counter = _extract_metric(metrics, "sglang_num_requests_total")
            running = _extract_metric(metrics, "sglang_num_running_requests")
            waiting = _extract_metric(metrics, "sglang_num_waiting_requests")

        log.append(f"请求总数: {initial_counter}, 运行中: {running}, 等待中: {waiting}")

        if running > 0 or waiting > 0:
            log.append("当前有正在处理的请求")
            return {"ok": False, "message": "当前有正在处理的请求，不能进行测试", "log": log}

        # Polling loop
        interval = 10
        samples = max(1, duration // interval)
        log.append(f"开始监控: {samples} 次采样，间隔 {interval}s")

        for i in range(samples):
            if i > 0:
                await asyncio.sleep(interval)

            try:
                resp = await client.get(metrics_url)
                current_metrics = resp.text
            except Exception as e:
                log.append(f"采样 {i+1}/{samples}: 获取失败 ({e})")
                continue

            if is_vllm:
                cur_counter = _extract_metric(current_metrics, "vllm:e2e_request_latency_seconds_count")
                cur_running = _extract_metric(current_metrics, "vllm:num_requests_running")
                cur_waiting = _extract_metric(current_metrics, "vllm:num_requests_waiting")
            else:
                cur_counter = _extract_metric(current_metrics, "sglang_num_requests_total")
                cur_running = _extract_metric(current_metrics, "sglang_num_running_requests")
                cur_waiting = _extract_metric(current_metrics, "sglang_num_waiting_requests")

            log.append(f"采样 {i+1}/{samples}: 请求总数={cur_counter}, 运行中={cur_running}, 等待中={cur_waiting}")

            if cur_counter > initial_counter:
                new_reqs = int(cur_counter - initial_counter)
                log.append(f"检测到 {new_reqs} 个新请求")
                return {"ok": False, "message": f"检测到 {new_reqs} 个新请求，不能进行测试", "log": log}

            if cur_running > 0 or cur_waiting > 0:
                log.append("当前有正在处理的请求")
                return {"ok": False, "message": "当前有正在处理的请求，不能进行测试", "log": log}

        # Update initial counter for cumulative comparison
        initial_counter = cur_counter

    log.append("校验通过：未检测到流量")
    return {"ok": True, "message": f"在 {duration}s 内未检测到流量，可以测试", "log": log}


@app.post("/api/check-traffic")
async def api_check_traffic(request: Request):
    body = await request.json()
    host = body.get("host", "").strip()
    port = int(body.get("port", 0))
    duration = int(body.get("duration", 30))
    skip = body.get("skip", False)

    if not host or not port:
        raise HTTPException(status_code=400, detail="需要提供 host 和 port")

    result = await check_traffic(host, port, duration, skip)
    return JSONResponse(result)


@app.post("/api/monitor/health")
async def api_monitor_health(request: Request):
    """Proxy Monitor health checks so the browser never needs cross-origin access."""
    body = await request.json()
    try:
        config = parse_monitor_config(
            {
                **body,
                "monitor_enabled": True,
                "host": body.get("host", ""),
            },
            body.get("connection_mode", "host_port"),
        )
        url = _base_url(config) + "/health"
        # Monitor performs one batch CUDA/CUPTI health check for all selected
        # devices. Keep enough headroom for first-use driver initialization.
        async with httpx.AsyncClient(timeout=MONITOR_HEALTH_PROXY_TIMEOUT_SECONDS) as client:
            response = await client.get(url, params={"devices": config["monitor_devices"]})
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("health response is not a JSON object")
        return JSONResponse(
            {"ok": response.is_success and payload.get("profiler_available") is True,
             "url": str(response.url), "response": payload},
            status_code=200,
        )
    except (MonitorConfigError, MonitorClientError, ValueError, httpx.HTTPError) as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=200)


# ---------------------------------------------------------------------------
# API routes — tasks
# ---------------------------------------------------------------------------


@app.post("/api/task")
async def api_create_task(request: Request):
    body = await request.json()

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    connection_mode = body.get("connection_mode", "host_port")
    if connection_mode not in ("host_port", "base_url"):
        raise HTTPException(status_code=400, detail="invalid connection_mode")

    try:
        monitor_config = parse_monitor_config(body, connection_mode)
    except MonitorConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    monitor_enabled = monitor_config["monitor_enabled"]

    task_id = create_task(
        name=name,
        connection_mode=connection_mode,
        host=body.get("host", "").strip(),
        port=int(body.get("port", 0)),
        base_url=body.get("base_url", "").strip(),
        api_key=body.get("api_key", "").strip(),
        request_model=body.get("request_model", "").strip(),
        model_name=body.get("model_name", "").strip(),
        model_path=body.get("model_path") or CONFIG["default_model_path"],
        dataset_path=body.get("dataset_path", ""),
        extra_args=body.get("extra_args", []),
        **monitor_config,
    )

    # 检测测试模式
    test_mode = body.get("test_mode", "normal")

    if test_mode == "shared_prefix":
        # 公共前缀测试：创建单个 run
        num_groups = int(body.get("gsp_num_groups", 0))
        prompts_per_group = int(body.get("gsp_prompts_per_group", 0))
        system_prompt_len = int(body.get("gsp_system_prompt_len", 0))
        question_len = int(body.get("gsp_question_len", 0))
        output_len = int(body.get("gsp_output_len", 0))
        range_ratio = float(body.get("gsp_range_ratio", 0.0))
        request_rate = float(body.get("gsp_request_rate", 0))
        max_concurrency = int(body.get("gsp_max_concurrency", 0))

        total_prompts = num_groups * prompts_per_group
        if total_prompts <= 0 or total_prompts > 10000:
            raise HTTPException(status_code=400, detail="总请求数必须在 1-10000 之间")

        rid = create_run(
            task_id=task_id,
            test_mode="shared_prefix",
            total_prompts=total_prompts,
            random_output_len=output_len,
            request_rate=request_rate,
            gsp_num_groups=num_groups,
            gsp_system_prompt_len=system_prompt_len,
            gsp_question_len=question_len,
            gsp_prompts_per_group=prompts_per_group,
            gsp_range_ratio=range_ratio,
            gsp_max_concurrency=max_concurrency,
            monitor_status="pending" if monitor_enabled else "disabled",
        )

        return JSONResponse({"task_id": task_id, "run_count": 1})

    elif test_mode == "multi_turn":
        workload_mode = body.get("mt_workload_mode", "synthetic")
        if workload_mode not in ("synthetic", "replay", "timestamp_replay"):
            raise HTTPException(status_code=400, detail="不支持的多轮负载模式")
        # 多轮对话测试：显式测试组合 [[L, S, O, 客户端数, 请求数], ...]
        # 客户端数/请求数按用例独立（解耦）；兼容旧 3 元组 [L,S,O]（回退共享值）
        combos_raw = body.get("mt_combos")
        shared_clients = int(body.get("mt_num_clients", 10))
        shared_total = int(body.get("mt_total_requests", 0))
        combos = []
        if combos_raw:
            for c in combos_raw:
                if not isinstance(c, (list, tuple)) or len(c) < 3:
                    continue
                try:
                    L, S, O = int(c[0]), int(c[1]), int(c[2])
                except (TypeError, ValueError):
                    continue
                if len(c) >= 5:
                    try:
                        C, T = int(c[3]), int(c[4])
                    except (TypeError, ValueError):
                        C, T = shared_clients, shared_total
                else:
                    C, T = shared_clients, shared_total
                if L > 0 and S >= 0 and O > 0 and C > 0 and T >= 0:
                    combos.append((L, S, O, C, T))
        else:
            # 兼容旧单值字段 → 单条组合
            combos = [(
                int(body.get("mt_request_length", 1024)),
                int(body.get("mt_sub_question_input_length", 0)),
                int(body.get("mt_output_length", 256)),
                shared_clients,
                shared_total,
            )]
        if not combos or len(combos) > 200:
            raise HTTPException(status_code=400, detail=f"测试组合数必须在 1-200 之间 (当前: {len(combos)})")
        if workload_mode == "timestamp_replay":
            combos = combos[:1]

        run_ids = []
        for L, S, O, C, T in combos:
            rid = create_run(
                task_id=task_id,
                test_mode="multi_turn",
                request_rate=(
                    0.0 if workload_mode == "timestamp_replay"
                    else float(body.get("mt_request_rate", 0))
                ),
                mt_num_clients=C,
                mt_num_rounds=int(body.get("mt_num_rounds", 5)),
                mt_request_length=L,
                mt_sub_question_input_length=S,
                mt_output_length=O,
                mt_total_requests=T,
                mt_range_ratio=float(body.get("mt_range_ratio", 1.0)),
                mt_max_context_length=int(body.get("mt_max_context_length", 0)),
                mt_init_interval=float(body.get("mt_init_interval", 0.0)),
                mt_seed=int(body.get("mt_seed", 0)),
                mt_host_header=body.get("mt_host_header", "").strip(),
                mt_api_format=body.get("mt_api_format", "openai"),
                mt_thinking=int(body.get("mt_thinking", False)),
                mt_enable_session_key=int(body.get("mt_enable_session_key", True)),
                mt_enable_round_barrier=int(body.get("mt_enable_round_barrier", False)),
                monitor_status="pending" if monitor_enabled else "disabled",
                mt_workload_mode=workload_mode,
                mt_replay_sequential_sessions=int(
                    body.get("mt_replay_sequential_sessions", False)
                ),
            )
            run_ids.append(rid)

        return JSONResponse({"task_id": task_id, "run_count": len(run_ids)})

    else:
        # 普通性能测试：生成参数网格
        concurrencies = [int(c) for c in body.get("concurrency_list", [1]) if c is not None and int(c) > 0]
        total_prompts_list = [int(v) for v in body.get("total_prompts_list", body.get("total_prompts", [512]))
                              if v is not None and int(v) > 0]
        if not total_prompts_list:
            total_prompts_list = [512]
        input_lens = [int(v) for v in body.get("input_len_list", [8192]) if v is not None and int(v) > 0]
        output_lens = [int(v) for v in body.get("output_len_list", [2048]) if v is not None and int(v) > 0]
        rate_list = [float(v) for v in body.get("request_rate_list", [1]) if v is not None]

        if not concurrencies or not input_lens or not output_lens or not rate_list:
            raise HTTPException(status_code=400, detail="参数列表不能为空，请检查输入")

        iterations = max(1, int(body.get("iterations", 1)))

        run_ids = []
        for concurrency in concurrencies:
            for total_prompts in total_prompts_list:
                multiplier = max(1, total_prompts // concurrency)
                for inp_len in input_lens:
                    for out_len in output_lens:
                        for rate in rate_list:
                            for _ in range(iterations):
                                rid = create_run(
                                    task_id=task_id,
                                    concurrency=concurrency,
                                    prompt_multiplier=multiplier,
                                    total_prompts=total_prompts,
                                    random_input_len=inp_len,
                                    random_output_len=out_len,
                                    request_rate=rate,
                                    monitor_status="pending" if monitor_enabled else "disabled",
                                )
                                run_ids.append(rid)

        return JSONResponse({"task_id": task_id, "run_count": len(run_ids)})


@app.post("/api/eval/task")
async def api_create_eval_task(request: Request):
    body = await request.json()

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    connection_mode = body.get("connection_mode", "host_port")
    if connection_mode not in ("host_port", "base_url"):
        raise HTTPException(status_code=400, detail="invalid connection_mode")

    datasets = body.get("datasets", [])
    if not datasets or not isinstance(datasets, list):
        raise HTTPException(status_code=400, detail="至少选择一个数据集")

    task_id = create_task(
        name=name,
        task_type="eval",
        connection_mode=connection_mode,
        host=body.get("host", "").strip(),
        port=int(body.get("port", 0)),
        base_url=body.get("base_url", "").strip(),
        api_key=body.get("api_key", "").strip(),
        request_model=body.get("request_model", "").strip(),
        model_name=body.get("model_name", "").strip(),
        model_path=body.get("model_path") or CONFIG["default_model_path"],
        dataset_path=body.get("dataset_dir", "") or CONFIG.get("eval_dataset_dir", ""),
        extra_args=["--enable-think"] if body.get("enable_think") else [],
    )

    run_ids = []
    for ds in datasets:
        rid = create_run(
            task_id=task_id,
            dataset_name=ds["name"],
            limit_val=int(ds.get("limit", 0)),
            batch_size=int(ds.get("batch_size", 1)),
        )
        run_ids.append(rid)

    return JSONResponse({"task_id": task_id, "run_count": len(run_ids)})


@app.post("/api/api-test/task")
async def api_create_api_test_task(request: Request):
    """Create an OpenAI API test task."""
    body = await request.json()

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    connection_mode = body.get("connection_mode", "host_port")
    if connection_mode not in ("host_port", "base_url"):
        raise HTTPException(status_code=400, detail="invalid connection_mode")

    task_id = create_task(
        name=name,
        task_type="api_test",
        connection_mode=connection_mode,
        host=body.get("host", "").strip(),
        port=int(body.get("port", 0)),
        base_url=body.get("base_url", "").strip(),
        api_key=body.get("api_key", "").strip(),
        request_model=body.get("model", "").strip(),
        model_name="",
        model_path="",
        dataset_path="",
    )

    # Create a single run for api_test (single config per task)
    run_id = create_run(task_id=task_id)

    # Store API test config
    create_api_test_config(
        run_id=run_id,
        model=body.get("model", "").strip(),
        reasoning_model=body.get("reasoning_model", "").strip(),
        fc_model=body.get("fc_model", "").strip(),
        enable_fc_test=int(body.get("enable_fc_test", False)),
        enable_n_test=int(body.get("enable_n_test", False)),
        n_value=int(body.get("n_value", 3)),
    )

    return JSONResponse({"task_id": task_id, "run_id": run_id})


@app.post("/api/compat-test/task")
async def api_create_compat_test_task(request: Request):
    """Create a compatibility test task."""
    body = await request.json()

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    connection_mode = body.get("connection_mode", "host_port")
    if connection_mode not in ("host_port", "base_url"):
        raise HTTPException(status_code=400, detail="invalid connection_mode")

    task_id = create_task(
        name=name,
        task_type="compat_test",
        connection_mode=connection_mode,
        host=body.get("host", "").strip(),
        port=int(body.get("port", 0)),
        base_url=body.get("base_url", "").strip(),
        api_key=body.get("api_key", "").strip(),
        request_model=body.get("model", "").strip(),
        model_name="",
        model_path="",
        dataset_path="",
    )

    # Create a single run for compat_test
    run_id = create_run(task_id=task_id)

    # Store compat test config
    create_compat_test_config(
        run_id=run_id,
        test_level=body.get("test_level", "standard"),
        framework=body.get("framework", "standard"),
        include_extensions=int(body.get("include_extensions", False)),
        enable_dangerous=int(body.get("enable_dangerous", False)),
        timeout_seconds=int(body.get("timeout_seconds", 30)),
        max_retries=int(body.get("max_retries", 3)),
    )

    return JSONResponse({"task_id": task_id, "run_id": run_id})


@app.post("/api/task/{task_id}/run")
async def api_run_task(task_id: int, background_tasks: BackgroundTasks):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "running":
        raise HTTPException(status_code=400, detail="Task already running")

    update_task_status(task_id, "pending")
    if task.get("task_type") == "eval":
        background_tasks.add_task(execute_eval_task, task_id)
    elif task["task_type"] == "api_test":
        background_tasks.add_task(execute_api_test_task, task_id)
    elif task["task_type"] == "compat_test":
        background_tasks.add_task(execute_compat_test_task, task_id)
    else:
        background_tasks.add_task(execute_task, task_id)
    return JSONResponse({"ok": True, "task_id": task_id})


@app.post("/api/task/{task_id}/kill")
async def api_kill_task(task_id: int):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "running":
        raise HTTPException(status_code=400, detail="Task is not running")

    kill_task(task_id)
    return JSONResponse({"ok": True, "task_id": task_id})


@app.get("/api/task/{task_id}/killing")
async def api_is_killing(task_id: int):
    return JSONResponse({"killing": task_id in _killing_tasks})


@app.delete("/api/task/{task_id}")
async def api_delete_task(task_id: int):
    """Delete a task and all its associated records."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running task")

    delete_task(task_id)
    return JSONResponse({"ok": True, "message": f"Task {task_id} deleted"})


@app.get("/api/task/{task_id}")
async def api_get_task(task_id: int):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    # This endpoint is polled every few seconds. Return summary fields only;
    # full time-series samples are fetched lazily from /api/run/{run_id}.
    runs = get_task_runs(task_id, include_monitor_samples=False)
    return JSONResponse({"task": task, "runs": runs})


@app.get("/api/task/{task_id}/rerun-config")
async def api_get_rerun_config(task_id: int):
    """Get config for rerunning a task (for both bench and eval tasks)."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    runs = get_task_runs(task_id)

    config = {
        "name": task.get("name", ""),
        "connection_mode": task.get("connection_mode", "host_port"),
        "host": task.get("host", ""),
        "port": task.get("port", 0),
        "base_url": task.get("base_url", ""),
        "api_key": task.get("api_key", ""),
        "request_model": task.get("request_model", ""),
        "model_name": task.get("model_name", ""),
        "model_path": task.get("model_path", ""),
        "dataset_path": task.get("dataset_path", ""),
        "extra_args": json.loads(task.get("extra_args") or "[]"),
        "monitor_enabled": bool(task.get("monitor_enabled", 0)),
        "monitor_host": task.get("monitor_host", ""),
        "monitor_port": task.get("monitor_port", 30100),
        "monitor_device": task.get("monitor_device", 0),
        "monitor_devices": task.get("monitor_devices") or str(task.get("monitor_device", 0)),
        "monitor_interval_ms": task.get("monitor_interval_ms", 100),
        "monitor_max_duration": task.get("monitor_max_duration", 7200),
    }

    if task.get("task_type") == "eval":
        # Build dataset config from runs
        datasets = []
        seen_ds = set()
        for run in runs:
            ds_name = run.get("dataset_name", "")
            if ds_name and ds_name not in seen_ds:
                seen_ds.add(ds_name)
                datasets.append({
                    "name": ds_name,
                    "limit": run.get("limit_val", 0),
                    "batch_size": run.get("batch_size", 1),
                })
        config["datasets"] = datasets
        # Restore enable_think from extra_args
        extra_args = json.loads(task.get("extra_args") or "[]")
        config["enable_think"] = "--enable-think" in extra_args
    elif task.get("task_type") == "api_test":
        # Build api_test config from runs
        for run in runs:
            api_config = get_api_test_config(run["id"])
            if api_config:
                config["model"] = api_config.get("model", "")
                config["reasoning_model"] = api_config.get("reasoning_model", "")
                config["fc_model"] = api_config.get("fc_model", "")
                config["enable_fc_test"] = bool(api_config.get("enable_fc_test"))
                config["enable_n_test"] = bool(api_config.get("enable_n_test"))
                config["n_value"] = api_config.get("n_value", 3)
                break
    elif task.get("task_type") == "compat_test":
        # Build compat_test config from runs
        for run in runs:
            compat_config = get_compat_test_config(run["id"])
            if compat_config:
                config["model"] = task.get("request_model", "")
                config["test_level"] = compat_config.get("test_level", "standard")
                config["framework"] = compat_config.get("framework", "standard")
                config["include_extensions"] = bool(compat_config.get("include_extensions"))
                config["enable_dangerous"] = bool(compat_config.get("enable_dangerous"))
                config["timeout_seconds"] = compat_config.get("timeout_seconds", 30)
                config["max_retries"] = compat_config.get("max_retries", 3)
                break
    else:
        # Benchmark task - check test_mode
        first_run = runs[0] if runs else {}
        test_mode = first_run.get("test_mode", "normal")

        config["test_mode"] = test_mode

        if test_mode == "shared_prefix":
            # 公共前缀测试参数
            config["gsp_num_groups"] = first_run.get("gsp_num_groups", 10)
            config["gsp_prompts_per_group"] = first_run.get("gsp_prompts_per_group", 512)
            config["gsp_system_prompt_len"] = first_run.get("gsp_system_prompt_len", 2048)
            config["gsp_question_len"] = first_run.get("gsp_question_len", 128)
            config["gsp_output_len"] = first_run.get("random_output_len", 256)
            config["gsp_range_ratio"] = first_run.get("gsp_range_ratio", 0.0)
            config["gsp_request_rate"] = first_run.get("request_rate", 0)
            config["gsp_max_concurrency"] = first_run.get("gsp_max_concurrency", 0)
        elif test_mode == "multi_turn":
            # 多轮对话测试参数：显式组合（跨 run 按 (L,S,O,客户端,请求数) 五元组去重）
            seen = set()
            combos = []
            for r in runs:
                triple = (
                    r.get("mt_request_length", 0),
                    r.get("mt_sub_question_input_length", 0),
                    r.get("mt_output_length", 0),
                    r.get("mt_num_clients", 10),
                    r.get("mt_total_requests", 0),
                )
                if triple[0] > 0 and triple not in seen:
                    seen.add(triple)
                    combos.append(list(triple))
            config["mt_combos"] = combos or [[1024, 0, 256, 10, 0]]
            config["mt_num_clients"] = first_run.get("mt_num_clients", 10)
            config["mt_num_rounds"] = first_run.get("mt_num_rounds", 5)
            config["mt_range_ratio"] = first_run.get("mt_range_ratio", 1.0)
            config["mt_max_context_length"] = first_run.get("mt_max_context_length", 0)
            config["mt_init_interval"] = first_run.get("mt_init_interval", 0.0)
            config["mt_total_requests"] = first_run.get("mt_total_requests", 0)
            config["mt_seed"] = first_run.get("mt_seed", 0)
            config["mt_host_header"] = first_run.get("mt_host_header", "")
            config["mt_api_format"] = first_run.get("mt_api_format", "openai")
            config["mt_thinking"] = bool(first_run.get("mt_thinking", 0))
            config["mt_enable_session_key"] = bool(first_run.get("mt_enable_session_key", 1))
            config["mt_enable_round_barrier"] = bool(first_run.get("mt_enable_round_barrier", 0))
            config["mt_workload_mode"] = first_run.get("mt_workload_mode", "synthetic")
            config["mt_replay_sequential_sessions"] = bool(
                first_run.get("mt_replay_sequential_sessions", 0)
            )
            config["mt_request_rate"] = first_run.get("request_rate", 0)
        else:
            # 普通性能测试参数
            concurrencies = sorted(set(r.get("concurrency", 0) for r in runs if r.get("concurrency")))
            input_lens = sorted(set(r.get("random_input_len", 0) for r in runs if r.get("random_input_len")))
            output_lens = sorted(set(r.get("random_output_len", 0) for r in runs if r.get("random_output_len")))
            rates = sorted(set(r.get("request_rate", 0) for r in runs))

            # Get total_prompts directly from stored values
            total_prompts = sorted(set(r.get("total_prompts", 0) for r in runs if r.get("total_prompts")))
            if not total_prompts:
                total_prompts = [512]

            config["concurrency_list"] = concurrencies
            config["total_prompts_list"] = total_prompts
            config["input_len_list"] = input_lens
            config["output_len_list"] = output_lens
            config["request_rate_list"] = rates

            # Calculate iterations from total runs / combinations
            combinations = len(concurrencies) * len(total_prompts) * len(input_lens) * len(output_lens) * len(rates)
            if combinations > 0:
                iterations = len(runs) // combinations
                config["iterations"] = max(1, iterations)

    return JSONResponse(config)


@app.get("/api/task/{task_id}/runs")
async def api_get_task_runs(task_id: int):
    runs = get_task_runs(task_id, include_monitor_samples=False)
    return JSONResponse(runs)


@app.get("/api/run/{run_id}")
async def api_get_run(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(run)


@app.get("/api/api-test/results/{run_id}")
async def api_get_api_test_results(run_id: int):
    """Get API test results for a run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    results = get_api_test_results(run_id)
    return JSONResponse({"run": run, "results": results})


@app.get("/api/history")
async def api_history(page: int = 1, per_page: int = 20, task_type: str = None):
    return JSONResponse(list_tasks(page=page, per_page=per_page, task_type=task_type))


# ---------------------------------------------------------------------------
# API routes — download
# ---------------------------------------------------------------------------


def _format_value(v, field: str = "") -> str:
    """Format a value for markdown table output, matching bench_result_parser style."""
    # 命中率类字段（实测缓存命中率 L1/L2 / 理论命中率）：小数 → 百分数，保留两位
    if field in ("Cache hit rate", "Cache hit rate L1", "Cache hit rate L2",
                 "Theoretical Hit Rate (PD Sep)",
                 "Theoretical Hit Rate (PD No Sep)",
                 "Theoretical Max Prefix Hit Rate") and isinstance(v, (int, float)):
        return f"{v * 100:.2f}%"
    if isinstance(v, float):
        s = f"{v:.2f}"
        # Add thousand separators to the integer part
        parts = s.split(".")
        integer_part = parts[0]
        if len(integer_part) > 3:
            # Add commas every 3 digits from right
            negative = integer_part.startswith("-")
            digits = integer_part.lstrip("-")
            grouped = ""
            for i, d in enumerate(reversed(digits)):
                if i > 0 and i % 3 == 0:
                    grouped = "," + grouped
                grouped = d + grouped
            parts[0] = ("-" + grouped) if negative else grouped
        return ".".join(parts)
    return str(v)


def _generate_markdown(results: list[dict], fields: list[str], title: str = "Benchmark Results") -> str:
    """Generate a markdown table string from parsed benchmark results."""
    lines = [f"# {title}\n"]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for result in results:
        vals = []
        for f in fields:
            v = result.get(f, "")
            vals.append(_format_value(v, f))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _append_monitor_markdown(md: str, runs: list[dict]) -> str:
    monitored_runs = [r for r in runs if r.get("monitor_status") != "disabled"]
    if not monitored_runs:
        return md

    def stat(summary, metric, field="avg"):
        value = (summary or {}).get(metric)
        return value.get(field, "-") if isinstance(value, dict) else "-"

    def escape(value):
        return str(value if value not in (None, "") else "-").replace("|", "\\|").replace("\n", " ")

    hardware_device = None
    hardware_fields = (
        "sm_count", "cuda_core_count", "tensor_core_count",
        "peak_memory_bandwidth_gbps", "peak_nvlink_bandwidth_gbps",
        "peak_pcie_bandwidth_gbps",
    )
    for run in monitored_runs:
        metadata = run.get("monitor_metadata") or {}
        candidates = metadata.get("devices")
        if not isinstance(candidates, list):
            candidates = []
        for candidate in candidates:
            if isinstance(candidate, dict) and any(
                candidate.get(field) is not None for field in hardware_fields
            ):
                hardware_device = candidate
                break
        if hardware_device is not None:
            break

    if hardware_device is not None:
        device_name = (
            hardware_device.get("name")
            or hardware_device.get("device_name")
            or hardware_device.get("chip_name")
            or "-"
        )
        md += "\n## GPU Hardware Capacity (Per GPU)\n\n"
        md += "| Device | SMs | CUDA Cores | Tensor Cores | Peak HBM BW (GB/s) | Peak NVLink BW (GB/s, bidirectional) | Peak PCIe BW (GB/s, bidirectional) |\n"
        md += "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        md += (
            f"| {escape(device_name)} | {escape(hardware_device.get('sm_count'))} | "
            f"{escape(hardware_device.get('cuda_core_count'))} | "
            f"{escape(hardware_device.get('tensor_core_count'))} | "
            f"{escape(hardware_device.get('peak_memory_bandwidth_gbps'))} | "
            f"{escape(hardware_device.get('peak_nvlink_bandwidth_gbps'))} | "
            f"{escape(hardware_device.get('peak_pcie_bandwidth_gbps'))} |\n"
        )

    md += "\n## GPU Monitoring Summary\n\n"
    md += "| Run | Monitor Status | Avg GPU Util (%) | Avg CUDA Core Util (%) | Avg SM Util (%) | Avg Tensor Core Util (%) | HBM BW Util (%) | NVLink BW Util (%) | PCIe BW Util (%) | Samples | Data Complete | Warning |\n"
    md += "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"

    for run in monitored_runs:
        metadata = run.get("monitor_metadata") or {}
        warning = str(run.get("monitor_error") or "").replace("|", "\\|").replace("\n", " ")
        complete = metadata.get("data_complete")
        complete_text = "yes" if complete is True else ("no" if complete is False else "-")
        summary = run.get("monitor_summary") or {}
        md += (
            f"| {run['id']} | {run.get('monitor_status', '-')} | "
            f"{stat(summary, 'gpu_util_pct')} | {stat(summary, 'cuda_core_pct')} | "
            f"{stat(summary, 'sm_util_pct')} | {stat(summary, 'tensor_core_pct')} | "
            f"{stat(summary, 'hbm_bw_pct')} | "
            f"{stat(summary, 'nvlink_bw_pct')} | "
            f"{stat(summary, 'pcie_bw_pct')} | "
            f"{metadata.get('sample_count', '-')} | "
            f"{complete_text} | {warning or '-'} |\n"
        )

    per_device_runs = []
    for run in monitored_runs:
        device_summaries = (run.get("monitor_metadata") or {}).get("device_summaries")
        if isinstance(device_summaries, dict) and device_summaries:
            per_device_runs.append((run, device_summaries))

    if per_device_runs:
        md += "\n## Per-GPU Monitoring Details\n\n"
        md += "| Run | GPU | Device | Samples | Avg GPU Util (%) | P95 GPU Util (%) | Avg CUDA Core Util (%) | Avg SM Util (%) | Avg Tensor Core Util (%) |\n"
        md += "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for run, device_summaries in per_device_runs:
            def device_sort_key(item):
                try:
                    return (0, int(item[0]))
                except (TypeError, ValueError):
                    return (1, str(item[0]))

            for device_key, item in sorted(device_summaries.items(), key=device_sort_key):
                if not isinstance(item, dict):
                    continue
                device = item.get("device") if isinstance(item.get("device"), dict) else {}
                summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
                device_name = device.get("name") or device.get("device_name") or device.get("chip_name") or "-"
                md += (
                    f"| {run['id']} | {escape(device_key)} | {escape(device_name)} | "
                    f"{escape(item.get('sample_count'))} | {stat(summary, 'gpu_util_pct')} | "
                    f"{stat(summary, 'gpu_util_pct', 'p95')} | {stat(summary, 'cuda_core_pct')} | "
                    f"{stat(summary, 'sm_util_pct')} | {stat(summary, 'tensor_core_pct')} | "
                    "\n"
                )

        md += "\n## Per-GPU Bandwidth Details\n\n"
        md += "| Run | GPU | HBM Read | HBM Write | HBM Total | HBM % | L2 Traffic | NVLink RX | NVLink TX | NVLink Total | NVLink % | PCIe Read | PCIe Write | PCIe Total | PCIe % |\n"
        md += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for run, device_summaries in per_device_runs:
            for device_key, item in sorted(device_summaries.items(), key=device_sort_key):
                if not isinstance(item, dict):
                    continue
                summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
                bandwidth_metrics = (
                    "hbm_read_bw_gbps", "hbm_write_bw_gbps", "hbm_total_bw_gbps", "hbm_bw_pct",
                    "l2_bw_gbps", "nvlink_rx_bw_gbps", "nvlink_tx_bw_gbps",
                    "nvlink_total_bw_gbps", "nvlink_bw_pct", "pcie_read_bw_gbps",
                    "pcie_write_bw_gbps", "pcie_total_bw_gbps", "pcie_bw_pct",
                )
                values = " | ".join(str(stat(summary, metric)) for metric in bandwidth_metrics)
                md += f"| {run['id']} | {escape(device_key)} | {values} |\n"

    return md


def _build_task_markdown(task_id: int, include_monitor: bool = True) -> tuple[str, str]:
    """Build markdown content for a task. Returns (markdown_text, task_name)."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.get("task_type") == "eval":
        return _build_eval_markdown(task)

    from bench_result_parser import DEFAULT_FIELDS

    runs = get_task_runs(task_id)
    done_runs = [r for r in runs if r["status"] == "done"]
    if not done_runs:
        raise HTTPException(status_code=404, detail="No completed runs to download")

    all_results = []
    test_mode = done_runs[0].get("test_mode", "normal")

    if test_mode == "multi_turn":
        md, task_name = _build_multiturn_markdown(task, done_runs)
        return (_append_monitor_markdown(md, runs) if include_monitor else md), task_name

    for run in done_runs:
        parsed = _parse_run_result(run)

        # 注入测试参数到 parsed（仅 shared_prefix 模式需要额外参数列）
        if test_mode == "shared_prefix":
            parsed["组数"] = run["gsp_num_groups"]
            parsed["前缀长度"] = run["gsp_system_prompt_len"]
            parsed["问题长度"] = run["gsp_question_len"]
            parsed["每组请求数"] = run["gsp_prompts_per_group"]
            parsed["总请求数"] = run["total_prompts"]

        all_results.append(parsed)

    # 构建字段列表
    if test_mode == "shared_prefix":
        meta_fields = ["组数", "前缀长度", "问题长度", "每组请求数", "总请求数"]
    else:
        meta_fields = []

    # These legacy fields belong to conversation-style benchmarks. Current
    # multi-turn runs use _build_multiturn_markdown() above, so showing them in
    # normal/shared-prefix exports only creates empty, misleading columns.
    conversation_only_fields = {
        "Completed conversations",
        "Conversation throughput (conv/s)",
        "Mean conversation duration (s)",
        "Mean turns per conversation",
    }
    seen = set(meta_fields)
    ordered_data = [
        f for f in DEFAULT_FIELDS
        if f not in seen and f not in conversation_only_fields
    ]
    seen.update(ordered_data)
    # Append any extra fields not in DEFAULT_FIELDS
    for r in all_results:
        for k in r:
            if k not in seen:
                seen.add(k)
                ordered_data.append(k)
    fields = meta_fields + ordered_data

    # 添加测试模式标识
    title = f"Benchmark Results — {task['name']}"
    if test_mode == "shared_prefix":
        title += " (公共前缀测试)"

    md = _generate_markdown(all_results, fields, title)
    return (_append_monitor_markdown(md, runs) if include_monitor else md), task["name"]


# 多轮测试专属字段顺序（Markdown 导出/预览用）
MT_FIELD_ORDER = [
    "Traffic request rate",
    "Total requests",
    "Average Prompt Length", "P90 Prompt Length", "P99 Prompt Length",
    "Average Output Length", "P90 Output Length", "P99 Output Length",
    "Average TTFT (ms)", "Median TTFT (ms)", "P90 TTFT (ms)", "P99 TTFT (ms)", "Max TTFT (ms)",
    "Average TPOT (ms)", "Median TPOT (ms)", "P90 TPOT (ms)", "P99 TPOT (ms)", "Max TPOT (ms)",
    "Average Latency (ms)", "Median Latency (ms)", "P90 Latency (ms)", "P99 Latency (ms)", "Max Latency (ms)",
    "Input token throughput (tok/s)", "Output token throughput (tok/s)",
    "Request throughput (req/s)",
    # 命中率展示口径：L1 / (L1+L2 累加)
    # Cache hit rate L1 = 仅 L1 命中；Cache hit rate = L1+L2 累加（vLLM 无 L2 时即 L1）
    "Cache hit rate L1", "Cache hit rate",
    "Theoretical Max Prefix Hit Rate",
    "Completed sessions", "Replay dataset sessions", "Replay dataset cycles",
]


def _build_multiturn_markdown(task: dict, done_runs: list) -> tuple[str, str]:
    """Build a multi-turn specific markdown table.

    按多轮自己的字段顺序出表，并跳过所有用例都没有值的列（避免
    bench 的 DEFAULT_FIELDS 模板导致大量空列/值错位）。
    """
    all_results = []
    for run in done_runs:
        parsed = _parse_run_result(run)
        if run.get("mt_workload_mode") in ("replay", "timestamp_replay"):
            parsed["Theoretical Max Prefix Hit Rate"] = parsed.pop(
                "Theoretical Hit Rate (PD Sep)", None
            )
            parsed.pop("Theoretical Hit Rate (PD No Sep)", None)
        # 注入用例标识列（多组合时区分）
        if run.get("mt_workload_mode") == "timestamp_replay":
            parsed["用例"] = (
                f"Timestamp clients={run.get('mt_num_clients', 0)} "
                f"requests={run.get('mt_total_requests', 0) or 'initial_sessions'}"
            )
        elif run.get("mt_workload_mode") == "replay":
            parsed["用例"] = f"Replay O={run.get('mt_output_length', 0)}"
        else:
            parsed["用例"] = (
                f"L={run.get('mt_request_length', 0)} "
                f"S={run.get('mt_sub_question_input_length', 0)} "
                f"O={run.get('mt_output_length', 0)}"
            )
        parsed["客户端数"] = run.get("mt_num_clients", 0)
        parsed["轮数"] = "数据集原始轮数" if run.get("mt_workload_mode") in ("replay", "timestamp_replay") else run.get("mt_num_rounds", 0)
        all_results.append(parsed)

    meta_fields = ["用例", "客户端数", "轮数"]
    # 动态过滤：整列无值（None/空串）则去掉
    fields = [
        f for f in (meta_fields + MT_FIELD_ORDER)
        if any(r.get(f) not in (None, "") for r in all_results)
    ]
    title = f"Multi-Turn Results — {task['name']}"
    return _generate_markdown(all_results, fields, title), task["name"]


def _build_eval_markdown(task: dict) -> tuple[str, str]:
    """Build markdown summary for eval tasks."""
    runs = get_task_runs(task["id"])
    done_runs = [r for r in runs if r["status"] == "done"]
    if not done_runs:
        raise HTTPException(status_code=404, detail="No completed runs to download")

    # Resolve model name: prefer task field, then first parsed result's model_id
    model_name = task.get("model_name") or task.get("request_model") or ""
    if not model_name:
        for run in done_runs:
            try:
                p = json.loads(run.get("parsed_result") or "{}")
                mid = p.get("model_id", "").strip()
                if mid and mid != "default":
                    model_name = mid
                    break
            except json.JSONDecodeError:
                pass
    if not model_name:
        model_name = "unknown"

    if task["connection_mode"] == "host_port":
        conn_str = f"{task['host']}:{task['port']}"
    else:
        conn_str = task["base_url"]

    lines = [f"# Eval Results — {task['name']}\n"]
    lines.append(f"**Model:** {model_name}  ")
    lines.append(f"**API:** {conn_str}\n")

    total_score = 0.0
    total_count = 0

    lines.append("")
    lines.append("| Dataset | Accuracy | Limit | Batch Size | Duration |")
    lines.append("|---------|----------|-------|------------|----------|")

    for run in done_runs:
        parsed = {}
        if run.get("parsed_result"):
            try:
                parsed = json.loads(run["parsed_result"])
            except json.JSONDecodeError:
                pass
        ds_name = run.get("dataset_name", "?")
        ds_data = parsed.get("datasets", {}).get(ds_name, {})
        score = ds_data.get("score", 0)
        if isinstance(score, (int, float)):
            total_score += score
            total_count += 1
        score_str = f"{score*100:.2f}%" if isinstance(score, (int, float)) and score <= 1 else f"{score}"
        limit = run.get("limit_val", "?")
        batch = run.get("batch_size", "?")
        dur = parsed.get("duration_sec", "?")
        dur_str = f"{dur:.0f}s" if isinstance(dur, (int, float)) else str(dur)

        lines.append(f"| {ds_name} | {score_str} | {limit} | {batch} | {dur_str} |")

    if total_count > 0:
        avg = total_score / total_count * 100
        lines.append(f"\n**Average Accuracy:** {avg:.2f}%")

    return "\n".join(lines) + "\n", task["name"]


def _parse_run_result(run: dict) -> dict:
    """Parse a run's result from parsed_result or raw_output fallback."""
    parsed = {}
    if run.get("parsed_result"):
        try:
            parsed = json.loads(run["parsed_result"])
        except json.JSONDecodeError:
            pass
    if not parsed and run.get("raw_output"):
        import tempfile
        raw = run["raw_output"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        try:
            from bench_result_parser import parse_benchmark_file
            parsed_list = parse_benchmark_file(tmp_path)
            if parsed_list:
                parsed = parsed_list[0]
        finally:
            os.unlink(tmp_path)
    return parsed


def _build_task_json(task_id: int) -> tuple[str, str]:
    """Build a structured JSON export containing every completed run."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    done_runs = [r for r in get_task_runs(task_id) if r["status"] == "done"]
    if not done_runs:
        raise HTTPException(status_code=404, detail="No completed runs to download")

    # Keep task/run configuration useful for reproducing the benchmark, but do
    # not put credentials or the verbose captured stdout into the download.
    task_data = dict(task)
    task_data.pop("api_key", None)
    exported_runs = []
    for run in done_runs:
        run_data = dict(run)
        run_data.pop("raw_output", None)
        run_data.pop("parsed_result", None)
        run_data["result"] = _parse_run_result(run)
        exported_runs.append(run_data)

    payload = {
        "format": "llm-benchmark-task-result-v1",
        "task": task_data,
        "completed_run_count": len(exported_runs),
        "runs": exported_runs,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2), task["name"]


@app.get("/api/task/{task_id}/markdown")
async def api_get_task_markdown(task_id: int):
    md, _ = _build_task_markdown(task_id, include_monitor=False)
    return Response(content=md, media_type="text/plain")


@app.get("/api/task/{task_id}/download")
async def api_download_task(task_id: int):
    md, _ = _build_task_markdown(task_id)
    filename = f"bench_task_{task_id}.md"
    return Response(content=md, media_type="text/markdown", headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })


@app.get("/api/run/{run_id}/monitor.json")
async def api_download_run_monitor_json(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    payload = {
        "run_id": run_id,
        "request_id": run.get("monitor_request_id"),
        "session_id": run.get("monitor_session_id"),
        "status": run.get("monitor_status"),
        "error": run.get("monitor_error"),
        "started_at": run.get("monitor_started_at"),
        "finished_at": run.get("monitor_finished_at"),
        "summary": run.get("monitor_summary") or {},
        "metadata": run.get("monitor_metadata") or {},
        "samples": run.get("monitor_samples") or [],
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=run_{run_id}_gpu_monitor.json"},
    )


@app.get("/api/run/{run_id}/monitor.csv")
async def api_download_run_monitor_csv(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    samples = run.get("monitor_samples") or []
    fields = []
    for sample in samples:
        for key in sample:
            if key not in fields:
                fields.append(key)
    output = io.StringIO()
    if fields:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=run_{run_id}_gpu_monitor.csv"},
    )


@app.get("/api/run/{run_id}/monitor.raw.jsonl")
async def api_download_run_monitor_raw(run_id: int):
    """Stream the complete node-side JSONL file without buffering it locally."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    session_id = str(run.get("monitor_session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=404, detail="Run has no monitor session")
    task = get_task(run["task_id"])
    if not task or not task.get("monitor_enabled"):
        raise HTTPException(status_code=404, detail="Monitoring is not enabled for this run")

    try:
        url = _base_url(task) + f"/api/v1/monitor/sessions/{session_id}/samples"
    except MonitorClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
    try:
        request = client.build_request("GET", url, params={"run_id": run_id})
        upstream = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Monitor download failed: {exc}") from exc

    if not upstream.is_success:
        error_body = (await upstream.aread()).decode("utf-8", errors="replace")
        status_code = upstream.status_code if upstream.status_code in {403, 404, 409} else 502
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=status_code,
            detail=f"Monitor download failed (HTTP {upstream.status_code}): {error_body}",
        )

    async def stream_samples():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_samples(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename=run_{run_id}_gpu_monitor_full.jsonl"
        },
    )


@app.get("/api/task/{task_id}/download-json")
async def api_download_task_json(task_id: int):
    content, _ = _build_task_json(task_id)
    filename = f"bench_task_{task_id}.json"
    return Response(content=content, media_type="application/json", headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })
