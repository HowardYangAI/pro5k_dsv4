#!/usr/bin/env python3
"""
EvalScope evaluation wrapper for LLM Benchmark Platform.
Invoked as a subprocess by the runner. Streams all output to stdout and
emits a structured JSON result block for parsing.
"""
import argparse
import json
import logging
import os
import sys
import time
import types

# bfcl_eval's model_config.py eagerly imports every model handler, each of
# which pulls in its own third-party deps (boto3, qwen_agent, cohere, …).
# We only talk to an OpenAI-compatible API — let missing handlers slide.
_original_import = __builtins__.__import__


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("bfcl_eval.model_handler."):
        try:
            return _original_import(name, globals, locals, fromlist, level)
        except ImportError:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
            # Populate requested names so `from X import A,B` doesn't fail
            for attr in fromlist or ():
                if not attr.startswith("_"):
                    setattr(mod, attr, None)
            return mod
    return _original_import(name, globals, locals, fromlist, level)


__builtins__.__import__ = _patched_import


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--api-url', required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-path', default='')
    p.add_argument('--dataset-dir', default='')
    p.add_argument('--datasets', required=True)  # JSON array string
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--dataset-args', default='{}')
    p.add_argument('--gen-config', default='{}')
    p.add_argument('--api-key', default='EMPTY')
    p.add_argument('--output-dir', default='./eval_outputs')
    p.add_argument('--judge-strategy', default='auto')
    p.add_argument('--judge-model-args', default='{}')
    return p.parse_args()


def main():
    args = parse_args()
    datasets = json.loads(args.datasets)
    dataset_args = json.loads(args.dataset_args)
    gen_config = json.loads(args.gen_config)
    judge_model_args = json.loads(args.judge_model_args)

    # Redirect all logging to stdout for streaming capture
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root.addHandler(handler)

    from evalscope import TaskConfig, run_task

    # Build TaskConfig with optional judge_model_args
    task_cfg_kwargs = {
        'model': args.model_id,
        'api_url': args.api_url,
        'api_key': args.api_key,
        'eval_type': 'openai_api',
        'datasets': datasets,
        'dataset_dir': args.dataset_dir,
        'eval_batch_size': args.eval_batch_size,
        'limit': args.limit,
        'dataset_args': dataset_args,
        'generation_config': gen_config,
        'work_dir': args.output_dir,
        'ignore_errors': True,  # 忽略错误，继续处理其他样本
    }

    # Add judge configuration if provided
    if args.judge_strategy and judge_model_args:
        task_cfg_kwargs['judge_strategy'] = args.judge_strategy
        task_cfg_kwargs['judge_model_args'] = judge_model_args

    task_cfg = TaskConfig(**task_cfg_kwargs)

    print(f"\n{'='*60}")
    print(f"Starting eval: datasets={datasets}, model={args.model_id}")
    print(f"api_url={args.api_url}, limit={args.limit}, batch_size={args.eval_batch_size}")
    print(f"{'='*60}\n")

    start = time.time()
    result = run_task(task_cfg=task_cfg)
    elapsed = time.time() - start

    summary = _build_summary(args.model_id, elapsed, result)

    print(f"\n{'='*60}")
    print(f"Evaluation completed in {elapsed:.1f}s")
    print(f"{'='*60}")
    print("\n===EVAL_RESULT===")
    print(json.dumps(summary, ensure_ascii=False))
    print("===EVAL_RESULT_END===")


def _build_summary(model_id: str, elapsed: float, result: dict) -> dict:
    summary = {
        'model_id': model_id,
        'duration_sec': round(elapsed, 1),
        'datasets': {},
    }
    for ds_name, report in result.items():
        metrics_list = []
        for m in report.metrics:
            for c in m.categories:
                cat_name = c.name if isinstance(c.name, str) else '|'.join(c.name)
                metrics_list.append({
                    'name': cat_name,
                    'score': c.score,
                    'num': c.num,
                })
        summary['datasets'][ds_name] = {
            'score': report.score,
            'metric_name': report.metrics[0].name if report.metrics else '',
            'categories': metrics_list,
        }
    return summary


if __name__ == '__main__':
    main()
