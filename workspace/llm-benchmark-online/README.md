# LLM Benchmark Platform

Web-based LLM serving benchmark orchestration platform. Wraps the [sglang benchmark tool](https://github.com/sgl-project/sglang) to provide visual task configuration, batch parameter grid execution, and structured result comparison.

## Quick Start

```bash
# 1. Setup environment (first time only)
#    Use sudo to mount model directories; without sudo only pip/dataset steps run.
sudo bash setup_env.sh

# 2. Start the server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080
```

Open `http://<host>:8080/` in a browser.

## Configuration

All environment-specific paths are in `config.yaml`. Edit this file to match your deployment:

- **model_dirs** — directories to scan for tokenizer models
- **dataset_dir** — base directory for dataset files
- **datasets** — available datasets (name + filename), shown as dropdown in UI
- **default_model_path** — fallback model when none is selected

No code changes needed when directories change — just edit `config.yaml`.

## Usage

1. **Create a task** — `/task/create`: configure connection mode (host:port or base_url+API key), model, dataset, and select parameter combinations (concurrency, total prompts, input/output lengths, request rates) via the tile grid. Save/load parameter presets as **templates**.
2. **Run** — Click "开始执行" on the task detail page. Traffic check runs automatically for host:port mode (20s). If traffic is detected, a warning is shown — you can proceed or cancel.
3. **View results** — The results table shows run status and duration. Expand **Markdown 预览** for the full benchmark report (TTFT, TPOT, ITL, E2E latency percentiles, throughput, etc.).
4. **Download** — "下载 Markdown 结果" exports all runs as a markdown file.

### Environment Notes

- **Model directories**: Mounted via `setup_env.sh`. Paths are read from `config.yaml`.
- **Datasets**: Selected via dropdown. Add new datasets to `config.yaml` under `datasets:`.
- **Traffic check**: Automatically runs before task execution for host:port services. Polls the `/metrics` endpoint for vllm services.

## Code Structure

```
auto_test/
├── main.py                   # FastAPI app: page routes, API endpoints, config loading
├── runner.py                 # Subprocess management: calls bench.sh per run, parses output
├── db.py                     # SQLite CRUD (task + run + template tables)
├── schema.sql                # Table definitions
├── config.yaml               # Environment configuration (paths, datasets)
├── setup_env.sh              # One-time environment setup (pip, mounts, datasets)
├── bench.sh                  # Benchmark wrapper script (parameter routing → bench_serving.py)
├── bench_serving.py          # sglang bench_serving.py (benchmark execution engine)
├── bench_result_parser.py    # Parses benchmark raw output → structured dict
├── templates/
│   ├── base.html             # Common layout (topbar, container)
│   ├── index.html            # Homepage: new task button + task history list
│   ├── task_create.html      # Task creation form with template selector and tile-grid params
│   ├── task_detail.html      # Task detail: progress bar, results table, markdown preview, log viewer
│   └── history.html          # History browser with pagination
├── static/
│   └── style.css             # All styles
└── requirements.txt          # Python dependencies
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Homepage (task list) |
| GET | `/task/create` | Task creation form |
| GET | `/task/<id>` | Task detail page |
| GET | `/history` | History browser |
| GET | `/api/models` | List available models (JSON) |
| GET | `/api/datasets` | List available datasets (JSON) |
| POST | `/api/check-traffic` | Check vllm service for active traffic |
| POST | `/api/task` | Create task + generate parameter grid runs |
| POST | `/api/task/<id>/run` | Start task execution |
| GET | `/api/task/<id>` | Task detail + runs (JSON) |
| GET | `/api/task/<id>/markdown` | Markdown report (plain text) |
| GET | `/api/task/<id>/download` | Download all runs as markdown |
| GET | `/api/run/<id>` | Single run detail (JSON) |
| GET | `/api/run/<id>/monitor.json` | Download returned/display GPU samples and summary as JSON |
| GET | `/api/run/<id>/monitor.csv` | Download returned/display GPU samples as CSV |
| GET | `/api/run/<id>/monitor.raw.jsonl` | Stream all original GPU samples from the Monitor node |
| GET | `/api/history` | Task history (JSON, paginated) |
| GET | `/api/templates` | List templates |
| POST | `/api/templates` | Create template |
| PUT | `/api/templates/<id>` | Update template |
| DELETE | `/api/templates/<id>` | Delete template |
