#!/usr/bin/env python3
"""
Multi-turn benchmark for sgl-router.

Unified bench supporting BOTH sglang and vllm backends (no more V1/V2 split).
  - Does NOT depend on sglang/vllm (uses transformers + huggingface_hub directly)
  - API formats: --api-format sglang (native /generate) or openai (/v1/chat/completions)
  - Session-aware routing ON by default: each run gets a random session prefix, and
    every request carries X-Session-Id: <prefix>-client-{id}. Different runs
    use different prefixes (so they don't share router binding state); within one
    run each client keeps a stable key for session affinity. Use --disable-session-key
    to turn it off (baseline runs, or backends whose router ignores the header).
  - Conversation recycling (--min-recycle-rounds/--max-recycle-rounds): break each
    client's num_rounds into multiple shorter conversations so prompt lengths stagger
    across clients, instead of all clients marching from short to long in lockstep
  - Stagger mode (--stagger): decouple sending from responses so prompt lengths mix
    right from the start. Model output is NOT appended to history (only sub-questions
    are), so every round's payload is pre-built and each client fires its rounds on
    its own clock (random start offset + inter-round interval). Different clients
    then sit at different rounds at the same wall-clock time -> prompt lengths mix.
    Combine with --start-offset-max / --interval-min / --interval-max to control the
    spread. NOTE: discarding output means later rounds no longer carry the real
    generated text as prefix (the first round's input is still a shared prefix).

Usage:
    # Session key is on by default (random prefix isolates runs).
    python bench_multiturn.py --host localhost --port 30000

    # sglang native /generate endpoint
    python bench_multiturn.py --api-format sglang

    # vllm / OpenAI-compatible endpoint
    python bench_multiturn.py --api-format openai --model-path meta-llama/Llama-3.1-8B-Instruct

    # With ShareGPT dataset
    python bench_multiturn.py --dataset-path /dfs/liuxing/data/ShareGPT_V3_unfiltered_cleaned_split.json

    # Custom Host header is OFF by default. Enable LB virtual-host routing with:
    #   python bench_multiturn.py --host-header glm51-lb-pd.test.example.com

    # Baseline run without session key (e.g. round_robin/random, or a router that
    # ignores the header)
    python bench_multiturn.py --disable-session-key

    # Stagger prompt lengths via conversation recycling (each client still sends
    # num_rounds requests, but each conversation is only [2,4] rounds long and
    # restarts when done, so different clients sit at different stages at once)
    python bench_multiturn.py --num-rounds 5 --min-recycle-rounds 2 --max-recycle-rounds 4

    # Decouple sending from responses: all payloads are pre-built into per-client
    # ordered queues. The sender randomly picks a client, pops its next round, and
    # fires at the rate controlled by --request-rate. Responses do not gate sending.
    python bench_multiturn.py --stagger --request-rate 10 --num-clients 32

    # Long-context growth scenario: first request is ~30k tokens, each round
    # appends only 1k more input (model output NOT appended to history), so
    # prompts grow 30k -> 31k -> 32k -> ... while rounds still advance on
    # responses as usual.
    python bench_multiturn.py --request-length 30000 --sub-question-input-length 1000 \
        --output-length 64 --exclude-output-from-history

    # Mixed-length scenario (cache-aware context mixing): half the clients run
    # long prompts (30k, +1k/round) and the other half run short ones (5k,
    # +1k/round), so long and short prompts hit the router at the same time.
    python bench_multiturn.py --request-length 30000 --request-length2 5000 \
        --sub-question-input-length 1000 --exclude-output-from-history

    # Cache-aware weakness scenario: all clients share an identical 25k prefix,
    # so cache_aware routing concentrates them on one worker, exposing the
    # KV-cache eviction vs router optimism gap (router reports high cache hit
    # rate while workers suffer cold prefills from eviction).
    python bench_multiturn.py --shared-prefix-tokens 25000 --request-length 30000 \
        --sub-question-input-length 1000 --exclude-output-from-history \
        --num-rounds 5 --num-clients 24 --max-parallel 64
"""

import argparse
import asyncio
import json
import math
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from tqdm.asyncio import tqdm

import aiohttp
from replay_store import DEFAULT_STORE, ReplayStore

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=600)

# Header name for session-aware routing
SESSION_ID_HEADER = "X-Session-Id"

# ShareGPT defaults (same as sglang)
SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


# ─── Data classes ─────────────────────────────────────────────


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    output_ids: list = field(default_factory=list)
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list = field(default_factory=list)
    prompt_len: int = 0
    cached_tokens: int = 0
    # 缓存命中分层：L1(device/NPU HBM)、L2(host/DRAM)
    # sglang 从 sglext.cached_tokens_details.{device,host} 读取；
    # vLLM 只有 L1(usage.prompt_tokens_details.cached_tokens)，L2 恒为 0。
    cached_tokens_device: int = 0
    cached_tokens_host: int = 0
    # 后端是否在本次响应中返回了 L2 分层详情（sglang=True，vLLM=False）。
    # 用于展示侧区分「L2=0.00%（支持但未命中）」与「L2=-（不支持）」。
    has_l2_details: bool = False
    generated_len: int = 0
    error: str = ""


@dataclass
class DatasetRow:
    prompt: object  # List[int] (token ids) or str (text)
    prompt_len: int
    output_len: int


# ─── Tokenizer & Dataset utilities (standalone, no sglang) ───


def get_tokenizer(model_path):
    """Load tokenizer using transformers directly (no sglang dependency)."""
    from transformers import AutoTokenizer

    if model_path.endswith(".json") or model_path.endswith(".model"):
        from sentencepiece import SentencePieceProcessor
        return SentencePieceProcessor(model_file=model_path)

    # If local path doesn't exist, try downloading from HF
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model_path)

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def is_file_valid_json(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (JSONDecodeError, OSError):
        return False


def download_and_cache_hf_file(repo_id, filename, repo_type="dataset"):
    """Download a file from Hugging Face and cache it locally."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def compute_random_lens(full_len, range_ratio, num):
    if full_len <= 0:
        return [0] * num
    # range_ratio 直接作为波动幅度：0 = 精确无波动，0.5 = ±50%，1.0 = ±100%（最大）
    delta = int(full_len * range_ratio)
    return np.random.randint(
        max(full_len - delta, 1),
        full_len + delta + 1,
        size=num,
    ).tolist()


# Parsed ShareGPT dataset cache (dataset_path -> filtered (q, a) pairs).
# sample_random_requests is called multiple times at startup (first-round
# pool, sub-question pool, lazy restart pool); re-reading + re-parsing the
# multi-GB JSON on every call dominated startup time.
_SHAREGPT_CACHE = {}


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
    quiet: bool = False,
):
    """
    Generate request samples. Full port of sglang's sample_random_requests.

    Two modes:
      - random_sample=True: Load ShareGPT dataset, tokenize, truncate/repeat to target length
      - random_sample=False: Generate random token ids
    """
    input_lens = compute_random_lens(full_len=input_len, range_ratio=range_ratio, num=num_prompts)
    output_lens = compute_random_lens(full_len=output_len, range_ratio=range_ratio, num=num_prompts)

    if return_text:
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        for i in range(num_prompts):
            input_lens[i] = max(1, input_lens[i] - num_special_tokens)

    if random_sample:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy
        # the input_lens. The parsed+filtered dataset is cached module-wide:
        # only the first call pays the full read/parse, later calls (first-
        # round pool, sub-question pool, lazy restart pool) reuse it.
        if dataset_path not in _SHAREGPT_CACHE:
            if not is_file_valid_json(dataset_path):
                dataset_path = download_and_cache_hf_file(
                    repo_id=SHAREGPT_REPO_ID,
                    filename=SHAREGPT_FILENAME,
                )

            with open(dataset_path) as f:
                raw = json.load(f)
            raw = [
                data
                for data in raw
                if len(data.get("conversations", data.get("conversation", []))) >= 2
            ]
            _SHAREGPT_CACHE[dataset_path] = [
                (
                    data.get("conversations", data.get("conversation", []))[0]["value"],
                    data.get("conversations", data.get("conversation", []))[1]["value"],
                )
                for data in raw
            ]
        dataset = list(_SHAREGPT_CACHE[dataset_path])  # copy: shuffle must not mutate the cache
        random.shuffle(dataset)

        input_requests = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            input_content = input_ids
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )
    else:
        # Sample token ids from random integers
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            input_content = [
                int((offsets[i] + j) % tokenizer.vocab_size)
                for j in range(input_lens[i])
            ]
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )

    if not quiet:
        print(f"#Input tokens: {np.sum(input_lens)}")
        print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


# ─── Payload generators ──────────────────────────────────────


def gen_payload(input_ids, output_len, lora_path="", max_context_length=0):
    if max_context_length > 0 and len(input_ids) > max_context_length:
        input_ids = input_ids[-max_context_length:]
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "lora_path": lora_path,
        "return_logprob": False,
        "logprob_start_len": -1,
        # sglang：开启后响应回带 sglext.cached_tokens_details.{device,host}；
        # vLLM：extra=allow，该参数被忽略（其 L1 由服务端 --enable-prompt-tokens-details 控制）
        "return_cached_tokens_details": True,
    }


def gen_payload_openai(messages, output_len, model, max_context_length=0, thinking=None):
    if max_context_length > 0 and messages:
        # approximate: keep last N chars (token count not available here)
        content = messages[-1].get("content", "")
        if len(content) > max_context_length * 4:
            messages[-1] = {**messages[-1], "content": content[-(max_context_length * 4):]}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # sglang：开启后响应回带 sglext.cached_tokens_details.{device,host}；
        # vLLM：extra=allow，该参数被忽略（其 L1 由服务端 --enable-prompt-tokens-details 控制）
        "return_cached_tokens_details": True,
    }
    # Thinking 模式开关：chat_template_kwargs.thinking（默认关，显式注入）
    if thinking is not None:
        payload["chat_template_kwargs"] = {"thinking": thinking}
    return payload


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# ─── Async request functions (with optional extra headers) ────


async def async_request_sglang_generate(payload, url, pbar=None, extra_headers=None):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        generated_text = ""
        all_output_ids = []
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
                    cached_tokens_device = 0
                    cached_tokens_host = 0
                    has_l2_details = False

                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            if data.get("output_ids"):
                                all_output_ids = data["output_ids"]
                                generated_text = data.get("text", "")
                                timestamp = time.perf_counter()

                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    meta_info = data.get("meta_info") or {}
                                    prompt_tokens = meta_info.get("prompt_tokens", 0)
                                    cached_tokens = meta_info.get("cached_tokens", 0)
                                    # sglang 分层：meta_info.cached_tokens_details.{device,host}
                                    ctd = meta_info.get("cached_tokens_details") or {}
                                    if ctd:
                                        cached_tokens_device = ctd.get("device", 0) or 0
                                        cached_tokens_host = ctd.get("host", 0) or 0
                                        has_l2_details = True
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.output_ids = all_output_ids
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.cached_tokens_device = cached_tokens_device
                    output.cached_tokens_host = cached_tokens_host
                    output.has_l2_details = has_l2_details
                    output.generated_len = len(output.itl) + 1
                else:
                    try:
                        body = await response.text()
                    except Exception:
                        body = ""
                    output.error = f"{response.reason or ''} {body[:500]}".strip()
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(payload, url, pbar=None, extra_headers=None):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        generated_text = ""
        ttft = 0.0
        latency = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
                    cached_tokens_device = 0
                    cached_tokens_host = 0
                    has_l2_details = False
                    sglext_seen = False
                    completion_tokens = 0

                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st

                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            if data.get("choices"):
                                raw_delta = data["choices"][0].get("delta")
                                text = raw_delta.get("content", "") if raw_delta else ""
                                if text:
                                    generated_text += text
                                    timestamp = time.perf_counter()

                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp

                            # sglang 流式：sglext 可能与 usage 在不同 chunk 里
                            # （sglext chunk 无 usage，usage chunk 无 sglext），
                            # 必须独立解析，不能嵌在 usage 分支内。
                            if data.get("sglext"):
                                ctd = (data["sglext"] or {}).get("cached_tokens_details") or {}
                                if ctd:
                                    cached_tokens_device = ctd.get("device", 0) or 0
                                    cached_tokens_host = ctd.get("host", 0) or 0
                                    cached_tokens = (
                                        cached_tokens_device + cached_tokens_host
                                    )
                                    has_l2_details = True
                                    sglext_seen = True

                            if data.get("usage"):
                                usage = data["usage"]
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                # vLLM：usage.cached_tokens(顶层) 或
                                #       usage.prompt_tokens_details.cached_tokens(嵌套)
                                #       只有 L1，无 L2
                                # sglang 的分层已在上面 sglext 分支处理；
                                # 此处仅在未见过 sglext 时回退读合计(L1)。
                                if not sglext_seen:
                                    if usage.get("cached_tokens") is not None:
                                        cached_tokens = usage["cached_tokens"]
                                    else:
                                        details = usage.get("prompt_tokens_details") or {}
                                        cached_tokens = details.get("cached_tokens", 0) or 0
                                    cached_tokens_device = cached_tokens
                                    cached_tokens_host = 0

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.cached_tokens_device = cached_tokens_device
                    output.cached_tokens_host = cached_tokens_host
                    output.has_l2_details = has_l2_details
                    output.generated_len = completion_tokens or (len(output.itl) + 1)
                else:
                    try:
                        body = await response.text()
                    except Exception:
                        body = ""
                    output.error = f"{response.reason or ''} {body[:500]}".strip()
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


# ─── Logging ─────────────────────────────────────────────────


def log_to_jsonl_file(data, file_path="performance_metrics.jsonl", tag=""):
    timestamped_data = {"timestamp": datetime.now().isoformat(), "tag": tag, **data}
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "a") as file:
            file.write(json.dumps(timestamped_data) + "\n")
    except IOError as e:
        print(f"Error writing to JSONL file: {e}")


# ─── ReadyQueue ──────────────────────────────────────────────


class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self._lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self._lock:
            self.requests.append(item)

    def pop(self):
        with self._lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                raise ValueError(f"{self.policy} not implemented")


# ─── WorkloadGenerator ──────────────────────────────────────


class WorkloadGenerator:
    def __init__(self, args):
        self.api_format = args.api_format
        self.model_path = args.model_path
        self.api_model_name = args.api_model_name or args.model_path

        # Construct the base URL and select request/payload functions
        if self.api_format == "openai":
            self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
            self.request_func = async_request_openai_chat_completions
        else:
            self.url = f"http://{args.host}:{args.port}/generate"
            self.request_func = async_request_sglang_generate

        # Session-aware routing: ON by default. Each test run gets a random
        # session prefix so different runs don't share router binding state, while
        # within one run each client keeps a stable key (prefix-client-{id}) for
        # session affinity. Pass --disable-session-key to turn it off (e.g. for
        # baseline policies that don't use session affinity, or backends whose
        # router ignores the header).
        import secrets
        self.enable_session_key = not args.disable_session_key
        self.session_prefix = secrets.token_hex(4) if self.enable_session_key else None
        # Per-client session counter, bumped each time a client restarts
        self.client_sessions = {i: 0 for i in range(args.num_clients)}
        self.host_header = args.host_header
        self.api_key = args.api_key
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.init_interval = args.init_interval
        self.start_time = None
        self.finished_time = None
        self.lora_path = args.lora_path
        # Long-context scenario: when set, model output is NOT appended to
        # history, so each round's prompt grows by exactly
        # --sub-question-input-length tokens (e.g. 30k -> 31k -> 32k ...).
        self.exclude_output_from_history = args.exclude_output_from_history
        self.max_context_length = args.max_context_length

        self.sent_requests = 0
        self.completed_requests = 0

        # Load tokenizer
        self.tokenizer = get_tokenizer(args.model_path)

        # Resolve per-client round counts
        min_rounds = args.min_rounds
        max_rounds = args.max_rounds
        if min_rounds == 0 and max_rounds == 0:
            min_rounds = args.num_rounds
            max_rounds = args.num_rounds
        elif min_rounds == 0:
            min_rounds = max_rounds
        elif max_rounds == 0:
            max_rounds = min_rounds
        if min_rounds < 1:
            raise ValueError(f"--min-rounds must be >= 1, got {min_rounds}")
        if min_rounds > max_rounds:
            raise ValueError(
                f"--min-rounds ({min_rounds}) must be <= --max-rounds ({max_rounds})"
            )

        self.min_rounds = min_rounds
        self.max_rounds = max_rounds

        if min_rounds == max_rounds:
            self.client_total_rounds = [min_rounds] * args.num_clients
        else:
            self.client_total_rounds = [
                random.randint(min_rounds, max_rounds) for _ in range(args.num_clients)
            ]

        # ─── Conversation recycling (segment a client's num_rounds into
        # multiple shorter conversations so prompt lengths stagger) ──
        recycle_min = args.min_recycle_rounds
        recycle_max = args.max_recycle_rounds
        if recycle_min == 0 and recycle_max == 0:
            self.recycle_enabled = False
        else:
            if recycle_min == 0:
                recycle_min = recycle_max
            elif recycle_max == 0:
                recycle_max = recycle_min
            if recycle_min < 1:
                raise ValueError(
                    f"--min-recycle-rounds must be >= 1, got {recycle_min}"
                )
            if recycle_min > recycle_max:
                raise ValueError(
                    f"--min-recycle-rounds ({recycle_min}) must be <= "
                    f"--max-recycle-rounds ({recycle_max})"
                )
            self.recycle_enabled = True

        if self.recycle_enabled:
            self.recycle_min = recycle_min
            self.recycle_max = recycle_max
        else:
            self.recycle_min = None
            self.recycle_max = None

        # clients_per_round[r] = number of clients participating in round r
        self.clients_per_round = [
            sum(1 for t in self.client_total_rounds if t > r) for r in range(max_rounds)
        ]
        self.total_requests = sum(self.client_total_rounds)

        # ─── total-requests mode: override the total ───
        self.total_requests_target = args.total_requests
        # Requests committed to be sent (initial pool + every enqueue).
        # Stop-gate is this counter, NOT pbar.n: pbar counts completed
        # responses, so gating on it overshoots — a restart decided while
        # n == target-1 is still sent after n passes the target.
        self.enqueued_requests = 0
        if self.total_requests_target > 0:
            self.total_requests = self.total_requests_target
            # Will need fresh prompts & sub-questions for extra sessions
            self._extra_prompt_pool = []  # lazy-filled

        range_ratio = args.range_ratio
        self.range_ratio = range_ratio
        self.dataset_path = args.dataset_path
        self.disable_random_sample = args.disable_random_sample
        self.request_length = args.request_length
        self.request_length2 = args.request_length2
        self.shared_prefix_tokens_stored = args.shared_prefix_tokens

        # Shared prefix: a single random token sequence prepended to EVERY
        # client's first message (and recycled conversations). All clients
        # share the identical prefix so cache_aware routing concentrates them
        # on one worker, exposing KV-cache eviction vs router optimism gap.
        self.shared_prefix = []
        if args.shared_prefix_tokens > 0:
            shared_sample = sample_random_requests(
                input_len=args.shared_prefix_tokens,
                output_len=args.output_length,
                num_prompts=1,
                range_ratio=1.0,  # fixed length, no variation
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                random_sample=not args.disable_random_sample,
                return_text=False,
                quiet=True,
            )
            self.shared_prefix = list(shared_sample[0].prompt)

        # Mixed-length mode: when --request-length2 > 0, the first half of
        # clients run --request-length (e.g. 30k) and the second half run
        # --request-length2 (e.g. 5k), so long and short prompts mix in one
        # run. Group sizes: num_clients // 2 each (second half gets any
        # remainder).
        split = args.num_clients // 2 if args.request_length2 > 0 else args.num_clients
        self.client_group = [
            0 if i < split else 1 for i in range(args.num_clients)
        ]
        first_round_samples = []
        for group, num in ((0, split), (1, args.num_clients - split)):
            if num == 0:
                continue
            group_len = (args.request_length if group == 0 else args.request_length2)
            unique_len = group_len - args.shared_prefix_tokens
            if unique_len < 1:
                raise ValueError(
                    f"--shared-prefix-tokens ({args.shared_prefix_tokens}) must be "
                    f"less than --request-length ({group_len})"
                )
            first_round_samples += sample_random_requests(
                input_len=unique_len,
                output_len=args.output_length,
                num_prompts=num,
                range_ratio=range_ratio,
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                random_sample=not args.disable_random_sample,
                return_text=False,
            )
        # Store per-sample output_len for first round
        first_round_output_lens = [row.output_len for row in first_round_samples]
        # Prepend the shared prefix so every client's prompt starts with the
        # same tokens (the unique portion follows).
        self.candidate_inputs = [
            self.shared_prefix + list(i.prompt) for i in first_round_samples
        ]

        # Per-group appended input length for subsequent rounds
        if args.sub_question_input_length != 0:
            sub_question_input_length = args.sub_question_input_length
        else:
            sub_question_input_length = args.request_length
        if args.sub_question_input_length2 != 0:
            sub_question_input_length2 = args.sub_question_input_length2
        else:
            sub_question_input_length2 = sub_question_input_length
        self.sub_question_input_length = sub_question_input_length
        self.sub_question_input_length2 = sub_question_input_length2

        # Per-client sub-question pools (one pop per round transition).
        #
        # Mixed-length mode (--request-length2 > 0) requires per-client
        # pools because each group uses a different sub-question length.
        # Non-mixed mode uses a single shared pool for speed (one file
        # read + one shuffle for all clients instead of N).
        num_sub_questions = sum(max(t - 1, 0) for t in self.client_total_rounds)
        # When total-requests mode is active, pad the pool so we don't run out
        # during repeated session restarts.
        if self.total_requests_target > num_sub_questions + args.num_clients:
            num_sub_questions = self.total_requests_target
        if args.request_length2 > 0 or args.sub_question_input_length2 != 0:
            self.sub_question_inputs = {}
            for i in range(args.num_clients):
                pool_len = (
                    sub_question_input_length
                    if self.client_group[i] == 0
                    else sub_question_input_length2
                )
                self.sub_question_inputs[i] = sample_random_requests(
                    input_len=pool_len,
                    output_len=args.output_length,
                    num_prompts=max(self.client_total_rounds[i] - 1, 1),
                    range_ratio=range_ratio,
                    tokenizer=self.tokenizer,
                    dataset_path=args.dataset_path,
                    random_sample=not args.disable_random_sample,
                    return_text=False,
                    quiet=True,
                )
        else:
            shared_pool = sample_random_requests(
                input_len=sub_question_input_length,
                output_len=args.output_length,
                num_prompts=max(num_sub_questions, 1),
                range_ratio=range_ratio,
                tokenizer=self.tokenizer,
                dataset_path=args.dataset_path,
                random_sample=not args.disable_random_sample,
                return_text=False,
                quiet=False,
            )
            self.sub_question_inputs = shared_pool  # list, indexed by round

        if self.api_format == "openai":
            # OpenAI mode: history is a messages list for /v1/chat/completions
            initial_messages = {
                i: [
                    {
                        "role": "user",
                        "content": self.tokenizer.decode(self.candidate_inputs[i]),
                    }
                ]
                for i in range(args.num_clients)
            }
            init_requests = [
                (
                    i,
                    gen_payload_openai(
                        initial_messages[i],
                        first_round_output_lens[i],
                        self.api_model_name,
                        self.max_context_length,
                        thinking=args.thinking,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": initial_messages[i],
                    "total_rounds": self.client_total_rounds[i],
                    "conv_round": 0,
                    "conv_length": self.client_total_rounds[i]
                    if not self.recycle_enabled
                    else random.randint(self.recycle_min, self.recycle_max),
                }
                for i in range(args.num_clients)
            }
        else:
            # SGLang mode: history is List[int] (token ids)
            init_requests = [
                (
                    i,
                    gen_payload(
                        self.candidate_inputs[i],
                        first_round_output_lens[i],
                        args.lora_path,
                        self.max_context_length,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": list(self.candidate_inputs[i]),
                    "total_rounds": self.client_total_rounds[i],
                    "conv_round": 0,
                    "conv_length": self.client_total_rounds[i]
                    if not self.recycle_enabled
                    else random.randint(self.recycle_min, self.recycle_max),
                }
                for i in range(args.num_clients)
            }
        self.ready_queue = ReadyQueue(
            init_requests=init_requests, policy=args.ready_queue_policy
        )
        self.enqueued_requests = len(init_requests)
        self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=self.total_requests)
        self.performance_metrics = {
            "ttft": [],
            "tpot": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
            "cached_tokens_device": [],
            "cached_tokens_host": [],
            "generated_len": [],
        }
        # 后端是否返回过 L2 分层详情（sglang=True，vLLM=False）。
        # 任一响应命中即置真，用于展示侧区分「L2=0.00%」与「L2=-」。
        self.cache_has_l2 = False
        # Theoretical cache reuse must be paired per client/session. Global
        # round buckets are invalid when clients complete different rounds.
        self.previous_request_metrics = {
            i: None for i in range(args.num_clients)
        }
        self.theoretical_hit_tokens_pd_sep = 0
        self.theoretical_hit_tokens_pd_no_sep = 0
        self.enable_round_barrier = args.enable_round_barrier
        # 逐轮桶始终初始化（理论命中率需要每轮实际输入/输出长度）；
        # 屏障只控制执行节奏，不影响逐轮数据收集
        for i in range(self.max_rounds):
            self.performance_metrics[f"round_{i}"] = {
                "ttft": [],
                "tpot": [],
                "latency": [],
                "prompt_len": [],
                "cached_tokens": [],
                "cached_tokens_device": [],
                "cached_tokens_host": [],
                "generated_len": [],
            }
        self.num_clients = args.num_clients

        self.num_rounds = self.max_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length
        self.max_context_length = args.max_context_length

        # ─── Stagger mode ─────────────────────────────────────────────
        # All rounds' payloads are pre-built into per-client ordered queues.
        # The sender loop randomly picks a client, pops its next round, and
        # fires at the rate controlled by --request-rate. Responses do not
        # gate sending, so different clients sit at different rounds at the
        # same wall-clock time -> prompt lengths mix naturally.
        self.stagger = args.stagger
        # Per-client ordered queues: client_id -> list[payload], filled by
        # _prebuild_payloads(). Only used in stagger mode.
        self.stagger_queues = {}  # client_id -> list[payload]
        if self.stagger:
            self._prebuild_payloads()

    def _prebuild_payloads(self):
        """
        Pre-build every round's payload for every client without waiting for
        any response. History only accumulates sub-questions (model output is
        discarded), so the full conversation is known at init time. This lets
        each client fire its rounds on its own clock.
        """
        for client_id in range(self.num_clients):
            payloads = []
            total = self.client_records[client_id]["total_rounds"]
            conv_len = self.client_records[client_id]["conv_length"]
            conv_round = 0

            # Round 0: use the candidate input already stored in client_records
            if self.api_format == "openai":
                history = list(self.client_records[client_id]["history"])  # [ {role:user,...} ]
            else:
                history = list(self.client_records[client_id]["history"])  # List[int]

            output_len = self.output_length
            r = 0
            while r < total:
                payloads.append(self._build_stagger_payload(history, output_len))
                conv_round += 1
                r += 1

                # Decide next history. Recycle resets the conversation; otherwise
                # append a fresh sub-question.
                if r >= total:
                    break
                if self.recycle_enabled and conv_round >= conv_len:
                    # start a new conversation segment
                    sub_q = self._pop_sub_question(client_id)
                    remaining = total - r
                    conv_len = min(
                        random.randint(self.recycle_min, self.recycle_max), remaining
                    )
                    conv_round = 0
                    if self.api_format == "openai":
                        prefix_text = self.tokenizer.decode(self.shared_prefix) if self.shared_prefix else ""
                        sub_q_text = self.tokenizer.decode(list(sub_q.prompt))
                        history = [
                            {"role": "user", "content": prefix_text + sub_q_text}
                        ]
                    else:
                        history = list(self.shared_prefix) + list(sub_q.prompt)
                else:
                    sub_q = self._pop_sub_question(client_id)
                    if self.api_format == "openai":
                        history = history + [
                            {
                                "role": "user",
                                "content": self.tokenizer.decode(list(sub_q.prompt)),
                            }
                        ]
                    else:
                        history = history + list(sub_q.prompt)

            self.stagger_queues[client_id] = payloads

    def _pop_sub_question(self, client_id):
        """Pop a sub-question for the given client from either a per-client
        dict (mixed-length mode) or a shared list (fast path).
        Auto-refills when the pool runs dry (total-requests mode)."""
        if isinstance(self.sub_question_inputs, dict):
            pool = self.sub_question_inputs[client_id]
            if not pool:
                pool_len = (
                    self.sub_question_input_length
                    if self.client_group[client_id] == 0
                    else self.sub_question_input_length2
                )
                pool.extend(
                    sample_random_requests(
                        input_len=pool_len,
                        output_len=self.output_length,
                        num_prompts=50,
                        range_ratio=self.range_ratio,
                        tokenizer=self.tokenizer,
                        dataset_path=self.dataset_path,
                        random_sample=not self.disable_random_sample,
                        return_text=False,
                        quiet=True,
                    )
                )
            return pool.pop()
        else:
            try:
                return self.sub_question_inputs.pop()
            except IndexError:
                # Refill the shared pool
                self.sub_question_inputs = sample_random_requests(
                    input_len=self.sub_question_input_length,
                    output_len=self.output_length,
                    num_prompts=max(self.total_requests_target, 200),
                    range_ratio=self.range_ratio,
                    tokenizer=self.tokenizer,
                    dataset_path=self.dataset_path,
                    random_sample=not self.disable_random_sample,
                    return_text=False,
                    quiet=True,
                )
                return self.sub_question_inputs.pop()

    def _fresh_first_prompt(self, client_id):
        """Generate a new first-round prompt for a client starting a new session.
        Returns (token_id_list, output_len)."""
        group = self.client_group[client_id]
        unique_len = self.request_length - self.shared_prefix_tokens_stored
        if group == 1 and self.request_length2 > 0:
            unique_len = self.request_length2 - self.shared_prefix_tokens_stored
        if unique_len < 1:
            unique_len = 1
        sample = sample_random_requests(
            input_len=unique_len,
            output_len=self.output_length,
            num_prompts=1,
            range_ratio=self.range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=self.dataset_path,
            random_sample=not self.disable_random_sample,
            return_text=False,
            quiet=True,
        )
        prompt_ids = self.shared_prefix + list(sample[0].prompt)
        return prompt_ids, sample[0].output_len

    def _build_stagger_payload(self, history, output_len):
        """Build a payload for the given history. History is NOT mutated here."""
        if self.api_format == "openai":
            return gen_payload_openai(history, output_len, self.api_model_name,
                                      self.max_context_length, thinking=args.thinking)
        else:
            return gen_payload(history, output_len, self.lora_path,
                               self.max_context_length)

    def _record_metrics(self, current_round, response):
        """Append a response's metrics to the global (and per-round) buckets."""
        if response.has_l2_details:
            self.cache_has_l2 = True
        self.performance_metrics["ttft"].append(response.ttft)
        tpot = None
        if response.generated_len > 1:
            tpot = max(0.0, response.latency - response.ttft) / (
                response.generated_len - 1
            )
            self.performance_metrics["tpot"].append(tpot)
        self.performance_metrics["latency"].append(response.latency)
        self.performance_metrics["prompt_len"].append(response.prompt_len)
        self.performance_metrics["cached_tokens"].append(response.cached_tokens)
        self.performance_metrics["cached_tokens_device"].append(response.cached_tokens_device)
        self.performance_metrics["cached_tokens_host"].append(response.cached_tokens_host)
        self.performance_metrics["generated_len"].append(response.generated_len)
        # 逐轮桶始终收集（理论命中率需要每轮实际输入/输出长度）
        bucket = self.performance_metrics[f"round_{current_round}"]
        bucket["ttft"].append(response.ttft)
        if tpot is not None:
            bucket["tpot"].append(tpot)
        bucket["latency"].append(response.latency)
        bucket["prompt_len"].append(response.prompt_len)
        bucket["cached_tokens"].append(response.cached_tokens)
        bucket["cached_tokens_device"].append(response.cached_tokens_device)
        bucket["cached_tokens_host"].append(response.cached_tokens_host)
        bucket["generated_len"].append(response.generated_len)

    def _record_theoretical_hit(self, client_id, response):
        """Accumulate reusable tokens against this client's preceding request."""
        previous = self.previous_request_metrics.get(client_id)
        if previous is not None:
            previous_prompt, previous_output = previous
            self.theoretical_hit_tokens_pd_sep += min(
                previous_prompt, response.prompt_len
            )
            self.theoretical_hit_tokens_pd_no_sep += min(
                previous_prompt + previous_output, response.prompt_len
            )
        self.previous_request_metrics[client_id] = (
            response.prompt_len, response.generated_len
        )

    def _reset_theoretical_session(self, client_id):
        self.previous_request_metrics[client_id] = None

    async def handle_request(self, item):
        client_id, payload = item
        # Session key (default on): prefix-client-{id}. The random prefix isolates
        # this run's sessions from other runs in the router's binding table.
        # When --disable-session-key is set, no header is sent.
        extra_headers = None
        if self.enable_session_key:
            session_id = f"{self.session_prefix}-client-{client_id}-s{self.client_sessions[client_id]}"
            extra_headers = {SESSION_ID_HEADER: session_id}
        if self.api_key:
            if extra_headers is None:
                extra_headers = {}
            extra_headers["Authorization"] = f"Bearer {self.api_key}"
        if self.host_header:
            if extra_headers is None:
                extra_headers = {}
            extra_headers["Host"] = self.host_header
        try:
            response = await self.request_func(
                payload, self.url, self.pbar, extra_headers=extra_headers
            )
            if self.pbar.n == self.pbar.total:
                self.finished_time = time.perf_counter()
            self.response_queue.put((client_id, response))
        except Exception as e:
            print(f"Request failed for client {client_id}: {e}")
            failed_response = RequestFuncOutput()
            failed_response.success = False
            failed_response.error = str(e)
            self.response_queue.put((client_id, failed_response))

    def request_sender(self):
        if self.stagger:
            return self._stagger_sender()

        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.01)
                    continue

                # Break only when target reached AND no more requests waiting
                if self.pbar.n >= self.pbar.total and not self.ready_queue.requests:
                    # Drain: wait for any still-in-flight tasks to settle
                    await asyncio.sleep(0.2)
                    break

                # First round: use --init-interval (fixed gap between clients' R0).
                # Subsequent rounds: fire immediately (gated only by max_parallel).
                if self.sent_requests < self.num_clients and self.init_interval > 0:
                    sleep_time = self.init_interval
                elif self.sent_requests >= self.num_clients:
                    # Subsequent rounds — response-driven, yield to event loop
                    await asyncio.sleep(0)
                else:
                    # init_interval == 0 for first round: use Poisson/uniform
                    if self.distribution == "poisson":
                        sleep_time = random.expovariate(self.request_rate)
                    elif self.distribution == "uniform":
                        avg_interval = (
                            1.0 / self.request_rate if self.request_rate > 0 else 1.0
                        )
                        sleep_time = random.uniform(0, 2 * avg_interval)
                    else:
                        raise ValueError("Invalid distribution type")
                await asyncio.sleep(sleep_time)

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        # Drain any remaining pending tasks before closing
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def _stagger_sender(self):
        """
        Stagger sender: all payloads are pre-built into per-client ordered queues.
        Each iteration randomly picks a client, pops its next round (preserving
        per-client round order), and fires it.  Sending rate is controlled by
        --request-rate (Poisson/Uniform); max_parallel gates in-flight count.
        Responses do not gate sending, so different clients naturally sit at
        different rounds at the same wall-clock time.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def sender_loop():
            # Track which clients still have pending rounds
            active_clients = [cid for cid, q in self.stagger_queues.items() if q]
            # Diagnostics: track expovariate values vs actual send intervals
            diag_exp_sum = 0.0
            diag_exp_count = 0
            diag_first_send_ts = None
            while active_clients:
                if self.pbar.n >= self.pbar.total:
                    break

                # Gate on global in-flight cap
                while self.sent_requests - self.completed_requests >= self.max_parallel:
                    if self.pbar.n >= self.pbar.total:
                        return
                    await asyncio.sleep(0.02)

                # Randomly pick a client that still has rounds
                client_id = random.choice(active_clients)
                queue = self.stagger_queues[client_id]

                payload = queue.pop(0)  # FIFO — preserve round order per client
                if not queue:
                    active_clients.remove(client_id)

                asyncio.create_task(self.handle_request((client_id, payload)))
                self.sent_requests += 1
                now = time.perf_counter()
                if diag_first_send_ts is None:
                    diag_first_send_ts = now

                if self.pbar.n >= self.pbar.total:
                    break

                # Rate limit per --request-rate (Poisson or Uniform)
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / self.request_rate if self.request_rate > 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                diag_exp_sum += sleep_time
                diag_exp_count += 1
                await asyncio.sleep(sleep_time)

            # Print diagnostic: compare expovariate theoretical vs actual send interval
            if diag_exp_count > 0 and diag_first_send_ts is not None:
                diag_actual_dur = time.perf_counter() - diag_first_send_ts
                diag_exp_mean = diag_exp_sum / diag_exp_count
                diag_actual_mean = diag_actual_dur / diag_exp_count if diag_exp_count > 0 else 0
                print(f"\n[stagger diag] N_sent={self.sent_requests} request_rate={self.request_rate}")
                print(f"  expovariate: sum={diag_exp_sum:.1f}s count={diag_exp_count} mean={diag_exp_mean:.4f}s -> theoretical QPS={1/diag_exp_mean:.3f}")
                print(f"  actual send: dur={diag_actual_dur:.1f}s count={diag_exp_count} mean={diag_actual_mean:.4f}s -> actual send QPS={1/diag_actual_mean:.3f}")

            # Wait until all in-flight requests finish
            while self.pbar.n < self.pbar.total:
                await asyncio.sleep(0.1)

        loop.run_until_complete(sender_loop())
        loop.close()

    def _restart_client_session(self, client_id):
        """Start a new session for a client that finished all its rounds.
        Generates a fresh first-round prompt, resets history, and enqueues
        the new request with an incremented session counter."""
        self.client_sessions[client_id] += 1
        self._reset_theoretical_session(client_id)
        session_num = self.client_sessions[client_id]

        # Fresh first-round prompt
        new_history_ids, output_len = self._fresh_first_prompt(client_id)

        # Random round count for the new session
        total_rounds = random.randint(self.min_rounds, self.max_rounds)

        # Reset client record
        rec = self.client_records[client_id]
        rec["round"] = 0
        rec["conv_round"] = 0
        rec["total_rounds"] = total_rounds
        if self.recycle_enabled:
            rec["conv_length"] = random.randint(self.recycle_min, self.recycle_max)
        else:
            rec["conv_length"] = total_rounds

        if self.api_format == "openai":
            rec["history"] = [
                {"role": "user", "content": self.tokenizer.decode(new_history_ids)}
            ]
            new_req = (
                client_id,
                gen_payload_openai(
                    rec["history"], output_len, self.api_model_name,
                    self.max_context_length, thinking=args.thinking,
                ),
            )
        else:
            rec["history"] = list(new_history_ids)
            new_req = (
                client_id,
                gen_payload(
                    rec["history"], output_len, self.lora_path,
                    self.max_context_length,
                ),
            )

        self.ready_queue.append(new_req)
        self.enqueued_requests += 1

    def _can_enqueue(self):
        """True while the send target (if set) has not been committed yet.

        Counts requests that WILL be sent (initial pool + every enqueue),
        unlike pbar.n which counts completed responses. Gating on pbar
        overshoots: a restart/next-round decided while n == target-1 is
        still enqueued and sent after n passes the target.
        """
        return (
            self.total_requests_target <= 0
            or self.enqueued_requests < self.total_requests_target
        )

    def response_handler(self):
        next_round_reqs = []
        current_barrier_round = 0
        barrier_round_completed = 0
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    print(f"Request failed for client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue

                # ─── Stagger mode: payloads are pre-built; do NOT append model
                # output to history and do NOT schedule the next round from here
                # (the per-client sender already fires all rounds on its clock).
                # Just record metrics + advance the round counter.
                if self.stagger:
                    current_round = self.client_records[client_id]["round"]
                    self.client_records[client_id]["round"] += 1
                    self._record_metrics(current_round, response)
                    self._record_theoretical_hit(client_id, response)
                    self.completed_requests += 1
                    continue

                # Extend history with response. When --exclude-output-from-history
                # is set, only the sub-question input is appended each round so
                # prompts grow by exactly --sub-question-input-length tokens.
                if not self.exclude_output_from_history:
                    if self.api_format == "openai":
                        if response.generated_text:
                            self.client_records[client_id]["history"].append(
                                {"role": "assistant", "content": response.generated_text}
                            )
                    else:
                        self.client_records[client_id]["history"].extend(
                            response.output_ids
                        )
                current_round = self.client_records[client_id]["round"]
                self.client_records[client_id]["round"] += 1
                self.client_records[client_id]["conv_round"] += 1
                self._record_metrics(current_round, response)
                self._record_theoretical_hit(client_id, response)
                self.completed_requests += 1

                client_total = self.client_records[client_id]["total_rounds"]
                if (
                    self.client_records[client_id]["round"] < client_total
                    and self._can_enqueue()
                ):
                    conv_length = self.client_records[client_id]["conv_length"]
                    recycle_now = (
                        self.recycle_enabled
                        and self.client_records[client_id]["conv_round"] >= conv_length
                    )

                    if recycle_now:
                        self._reset_theoretical_session(client_id)
                        # Start a fresh conversation segment so prompt lengths
                        # stagger across clients at the same wall-clock time.
                        sub_q = self._pop_sub_question(client_id)
                        remaining = client_total - self.client_records[client_id]["round"]
                        new_conv_len = min(
                            random.randint(self.recycle_min, self.recycle_max),
                            remaining,
                        )
                        self.client_records[client_id]["conv_round"] = 0
                        self.client_records[client_id]["conv_length"] = new_conv_len
                        if self.api_format == "openai":
                            prefix_text = self.tokenizer.decode(self.shared_prefix) if self.shared_prefix else ""
                            sub_q_text = self.tokenizer.decode(list(sub_q.prompt))
                            self.client_records[client_id]["history"] = [
                                {"role": "user", "content": prefix_text + sub_q_text}
                            ]
                            new_req = (
                                client_id,
                                gen_payload_openai(
                                    self.client_records[client_id]["history"],
                                    sub_q.output_len,
                                    self.api_model_name,
                                    self.max_context_length,
                                    thinking=args.thinking,
                                ),
                            )
                        else:
                            self.client_records[client_id]["history"] = (
                                list(self.shared_prefix) + list(sub_q.prompt)
                            )
                            new_req = (
                                client_id,
                                gen_payload(
                                    self.client_records[client_id]["history"],
                                    sub_q.output_len,
                                    self.lora_path,
                                    self.max_context_length,
                                ),
                            )
                    else:
                        sub_q = self._pop_sub_question(client_id)
                        if self.api_format == "openai":
                            # Append sub-question as a new user message
                            sub_q_text = self.tokenizer.decode(list(sub_q.prompt))
                            self.client_records[client_id]["history"].append(
                                {"role": "user", "content": sub_q_text}
                            )
                            new_req = (
                                client_id,
                                gen_payload_openai(
                                    self.client_records[client_id]["history"],
                                    sub_q.output_len,
                                    self.api_model_name,
                                    self.max_context_length,
                                    thinking=args.thinking,
                                ),
                            )
                        else:
                            # Append sub-question token ids to client's history
                            sub_q_ids = list(sub_q.prompt)
                            self.client_records[client_id]["history"].extend(sub_q_ids)
                            new_req = (
                                client_id,
                                gen_payload(
                                    self.client_records[client_id]["history"],
                                    sub_q.output_len,
                                    self.lora_path,
                                    self.max_context_length,
                                ),
                            )
                    if self.enable_round_barrier:
                        next_round_reqs.append(new_req)
                    else:
                        self.ready_queue.append(new_req)
                        self.enqueued_requests += 1
                else:
                    # Client finished all its rounds
                    if self.total_requests_target > 0 and self.enqueued_requests < self.total_requests_target:
                        self._restart_client_session(client_id)

                # Barrier logic: release next round when all clients for
                # current barrier round have completed
                if (
                    self.enable_round_barrier
                    and current_barrier_round < self.max_rounds
                ):
                    barrier_round_completed += 1
                    expected = self.clients_per_round[current_barrier_round]
                    if barrier_round_completed == expected:
                        print(
                            f"\n  Barrier: round {current_barrier_round} complete "
                            f"({expected} clients), releasing {len(next_round_reqs)} "
                            f"requests for round {current_barrier_round + 1}"
                        )
                        self._send_heartbeat(input_len=100, output_len=100)
                        time.sleep(10)
                        for req in next_round_reqs:
                            self.ready_queue.append(req)
                        self.enqueued_requests += len(next_round_reqs)
                        next_round_reqs = []
                        current_barrier_round += 1
                        barrier_round_completed = 0
            except queue.Empty:
                # >=: overshoot requests can push pbar.n past total (see
                # request_loop); == would never match and hang the thread.
                if self.pbar.n >= self.pbar.total:
                    break
            except ValueError as e:
                print(f"Error processing response for client {client_id}: {e}")
                continue

    def _send_heartbeat(self, input_len=100, output_len=20):
        """Send a small heartbeat request to the server."""
        heartbeat_input = [1] * input_len
        payload = gen_payload(heartbeat_input, output_len, self.lora_path)
        headers = {}
        if self.host_header:
            headers["Host"] = self.host_header
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers = headers or None
        try:
            requests.post(self.url, json=payload, headers=headers, timeout=30)
        except Exception as e:
            print(f"Heartbeat request failed: {e}")

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()
        self.pbar.close()

        duration = self.finished_time - self.start_time
        sorted_ttft = sorted(self.performance_metrics["ttft"])
        sorted_latency = sorted(self.performance_metrics["latency"])
        sorted_tpot = sorted(self.performance_metrics["tpot"])
        sorted_prompt_len = sorted(self.performance_metrics["prompt_len"])
        sorted_output_len = sorted(self.performance_metrics["generated_len"])

        def percentile(sorted_vals, q):
            if not sorted_vals:
                return 0.0
            idx = int(q * len(sorted_vals))
            if idx >= len(sorted_vals):
                idx = len(sorted_vals) - 1
            return sorted_vals[idx]

        def max_or_zero(sorted_vals):
            return sorted_vals[-1] if sorted_vals else 0.0

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "request_rate": self.request_rate,
                "average_prompt_len": (
                    sum(self.performance_metrics["prompt_len"])
                    / len(self.performance_metrics["prompt_len"])
                    if self.performance_metrics["prompt_len"]
                    else 0.0
                ),
                "average_output_len": (
                    sum(self.performance_metrics["generated_len"])
                    / len(self.performance_metrics["generated_len"])
                    if self.performance_metrics["generated_len"]
                    else 0.0
                ),
                "p90_prompt_len": percentile(sorted_prompt_len, 0.9),
                "p99_prompt_len": percentile(sorted_prompt_len, 0.99),
                "p90_output_len": percentile(sorted_output_len, 0.9),
                "p99_output_len": percentile(sorted_output_len, 0.99),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": percentile(sorted_ttft, 0.9),
                "p99_ttft": percentile(sorted_ttft, 0.99),
                "median_ttft": percentile(sorted_ttft, 0.5),
                "max_ttft": max_or_zero(sorted_ttft),
                "average_tpot": (
                    sum(self.performance_metrics["tpot"])
                    / len(self.performance_metrics["tpot"])
                    if self.performance_metrics["tpot"]
                    else 0.0
                ),
                "p90_tpot": percentile(sorted_tpot, 0.9),
                "p99_tpot": percentile(sorted_tpot, 0.99),
                "median_tpot": percentile(sorted_tpot, 0.5),
                "max_tpot": max_or_zero(sorted_tpot),
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": percentile(sorted_latency, 0.9),
                "p99_latency": percentile(sorted_latency, 0.99),
                "median_latency": percentile(sorted_latency, 0.5),
                "max_latency": max_or_zero(sorted_latency),
                "input_token_throughput": sum(self.performance_metrics["prompt_len"])
                / duration,
                "output_token_throughput": sum(
                    self.performance_metrics["generated_len"]
                )
                / duration,
                "throughput": self.pbar.total / duration,
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
                # 缓存命中率分层（分母同上 Σprompt_len）：
                #   L1 = Σdevice / Σprompt_len（GPU/NPU HBM 命中）
                #   L2 = Σhost  / Σprompt_len（host/DRAM 命中，仅 sglang 有）
                # cache_has_l2：后端是否返回过 L2 分层详情，决定 L2 展示数值还是「-」
                "cache_hit_rate_l1": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens_device"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
                "cache_hit_rate_l2": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens_host"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
                "cache_has_l2": self.cache_has_l2,
                # Per-client/session theoretical prefix reuse. Session first
                # requests contribute zero; unequal client round counts and a
                # total-request cutoff are therefore handled correctly.
                "theoretical_hit_rate_pd_sep": (
                    self.theoretical_hit_tokens_pd_sep
                    / sum(self.performance_metrics["prompt_len"])
                    if sum(self.performance_metrics["prompt_len"]) > 0
                    else 0
                ),
                "theoretical_hit_rate_pd_no_sep": (
                    self.theoretical_hit_tokens_pd_no_sep
                    / sum(self.performance_metrics["prompt_len"])
                    if sum(self.performance_metrics["prompt_len"]) > 0
                    else 0
                ),
            },
        }
        if self.enable_round_barrier:
            performance_data["round"] = {}
            for round_num in range(self.num_rounds):
                round_key = f"round_{round_num}"
                round_metrics = self.performance_metrics[round_key]
                performance_data["round"][round_key] = {
                    "average_ttft": (
                        sum(round_metrics["ttft"]) / len(round_metrics["ttft"])
                        if round_metrics["ttft"]
                        else 0
                    ),
                    "average_tpot": (
                        sum(round_metrics["tpot"]) / len(round_metrics["tpot"])
                        if round_metrics["tpot"]
                        else 0
                    ),
                    "cache_hit_rate": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "cache_hit_rate_l1": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens_device"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "cache_hit_rate_l2": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens_host"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "request_count": len(round_metrics["ttft"]),
                }
        print("All requests completed")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {performance_data['summary']['total_requests']} at {performance_data['summary']['request_rate']} requests per second"
        )
        print(
            f"  Average Prompt Length: {performance_data['summary']['average_prompt_len']:.2f} tokens"
        )
        print(
            f"  Average Output Length: {performance_data['summary']['average_output_len']:.2f} tokens"
        )
        print(
            f"  P90 Prompt Length: {performance_data['summary']['p90_prompt_len']:.0f} tokens"
        )
        print(
            f"  P99 Prompt Length: {performance_data['summary']['p99_prompt_len']:.0f} tokens"
        )
        print(
            f"  P90 Output Length: {performance_data['summary']['p90_output_len']:.0f} tokens"
        )
        print(
            f"  P99 Output Length: {performance_data['summary']['p99_output_len']:.0f} tokens"
        )
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  P99 TTFT: {performance_data['summary']['p99_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(f"  Max TTFT: {performance_data['summary']['max_ttft']:.2f}")
        print(f"  Average TPOT: {performance_data['summary']['average_tpot']:.4f}")
        print(f"  P90 TPOT: {performance_data['summary']['p90_tpot']:.4f}")
        print(f"  P99 TPOT: {performance_data['summary']['p99_tpot']:.4f}")
        print(f"  Median TPOT: {performance_data['summary']['median_tpot']:.4f}")
        print(f"  Max TPOT: {performance_data['summary']['max_tpot']:.4f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  P99 latency: {performance_data['summary']['p99_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(f"  Max latency: {performance_data['summary']['max_latency']:.2f}")
        print(
            f"  Input token throughput: {performance_data['summary']['input_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Output token throughput: {performance_data['summary']['output_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Request Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")
        _summary = performance_data["summary"]
        # 展示口径：L1 / (L1+L2 累加) / (L1+L2+L3 累加)
        # 第二列 = 合计命中率，仅在 sglang(支持 L2)时展示；vLLM 无 L2 显示「-」
        # 第三列 = L3 累加，两种后端当前均无 L3，统一显示「-」
        if _summary["cache_has_l2"]:
            _l2_str = f"{_summary['cache_hit_rate'] * 100:.2f}%"
        else:
            _l2_str = "-"
        print(
            f"  Cache Hit Rate (L1/L2/L3): "
            f"{_summary['cache_hit_rate_l1'] * 100:.2f}%/"
            f"{_l2_str}/-"
        )

        if self.enable_round_barrier:
            # Print round-based summary
            print("Per-round metrics:")
            if "round" in performance_data:
                for round_num in range(self.num_rounds):
                    round_key = f"round_{round_num}"
                    if round_key in performance_data["round"]:
                        round_data = performance_data["round"][round_key]
                        avg_ttft = round_data["average_ttft"]
                        cache_hit_rate = round_data["cache_hit_rate"]
                        request_count = round_data["request_count"]
                        clients_in_round = self.clients_per_round[round_num]
                        print(
                            f"  Round {round_num}: Average TTFT = {avg_ttft:.2f}s, "
                            f"Cache Hit Rate = {cache_hit_rate:.6f} "
                            f"({request_count} requests, "
                            f"{clients_in_round} clients)"
                        )
                    else:
                        print(f"  Round {round_num}: No requests completed")

        return performance_data


class ReplayWorkloadGenerator(WorkloadGenerator):
    """Closed-loop replay: one in-flight request per client/session.

    A client sends the next recorded history prefix only after the previous
    response completes.  When a session ends it immediately claims another
    dataset session.  --request-rate is a global emission ceiling; inf means
    responses alone pace the workload.
    """

    def __init__(self, args):
        if args.api_format != "openai":
            raise ValueError("Replay workload currently requires --api-format openai")
        import secrets
        self.api_format = "openai"
        self.model_path = args.model_path
        self.api_model_name = args.api_model_name or args.model_path
        self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
        self.request_func = async_request_openai_chat_completions
        self.enable_session_key = not args.disable_session_key
        self.session_prefix = secrets.token_hex(4) if self.enable_session_key else None
        self.client_sessions = {i: -1 for i in range(args.num_clients)}
        self.host_header = args.host_header
        self.api_key = args.api_key
        self.request_rate = args.request_rate
        self.init_interval = args.init_interval
        self.start_time = self.finished_time = None
        self.max_parallel = args.num_clients
        self.num_clients = args.num_clients
        self.output_length = args.output_length
        self.max_context_length = args.max_context_length
        self.thinking = args.thinking
        self.stagger = False
        self.enable_round_barrier = False
        self.sent_requests = self.completed_requests = self.enqueued_requests = 0
        self.completed_sessions = 0
        self.dataset_cycles = 0
        self.total_requests_target = args.total_requests
        self.replay_path = args.replay_dataset_path
        self._replay_store = ReplayStore(self.replay_path)
        self.replay_sequential_sessions = args.replay_sequential_sessions
        self._session_ids = self._replay_store.session_ids(
            ordered=self.replay_sequential_sessions
        )
        if not self._session_ids:
            raise ValueError(f"Replay dataset is empty: {self.replay_path}")
        if not self.replay_sequential_sessions:
            random.shuffle(self._session_ids)
        self._next_session = 0

        # With no explicit request target, replay every dataset session once.
        if self.total_requests_target <= 0:
            self.total_requests_target = int(
                self._replay_store.metadata().get("span_count", 0)
            )
        self.total_requests = self.total_requests_target

        self.client_records = {}
        initial = []
        for client_id in range(args.num_clients):
            if self.enqueued_requests >= self.total_requests_target:
                break
            if self._claim_session(client_id):
                initial.append((client_id, self._build_replay_payload(client_id)))
                self.enqueued_requests += 1
        if not initial:
            raise ValueError("No replay requests could be initialized")
        self.ready_queue = ReadyQueue(initial, policy=args.ready_queue_policy)
        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=self.total_requests_target)

        max_turns = max(len(r["spans"]) for r in self.client_records.values())
        # The dataset can contain longer sessions claimed later. The extracted
        # dataset's metadata is optional, so grow round buckets on demand too.
        self.max_rounds = max_turns
        self.num_rounds = max_turns
        self.clients_per_round = [0] * max_turns
        self.performance_metrics = {
            "ttft": [], "tpot": [], "latency": [], "prompt_len": [],
            "cached_tokens": [], "cached_tokens_device": [],
            "cached_tokens_host": [], "generated_len": [],
        }
        self.cache_has_l2 = False
        self.previous_request_metrics = {
            i: None for i in range(args.num_clients)
        }
        self.theoretical_hit_tokens_pd_sep = 0
        self.theoretical_hit_tokens_pd_no_sep = 0
        for i in range(max_turns):
            self._ensure_round_bucket(i)

    def _ensure_round_bucket(self, round_num):
        while self.max_rounds <= round_num:
            self.max_rounds += 1
            self.num_rounds = self.max_rounds
            self.clients_per_round.append(0)
        self.performance_metrics.setdefault(f"round_{round_num}", {
            "ttft": [], "tpot": [], "latency": [], "prompt_len": [],
            "cached_tokens": [], "cached_tokens_device": [],
            "cached_tokens_host": [], "generated_len": [],
        })

    def _claim_session(self, client_id):
        if self._next_session >= len(self._session_ids):
            self.dataset_cycles += 1
            if not self.replay_sequential_sessions:
                random.shuffle(self._session_ids)
            self._next_session = 0
        session_id = self._session_ids[self._next_session]
        self._next_session += 1
        spans = self._replay_store.load_session(session_id)
        if not spans:
            return False
        self.client_sessions[client_id] += 1
        if hasattr(self, "previous_request_metrics"):
            self._reset_theoretical_session(client_id)
        self.client_records[client_id] = {
            "session_id": session_id, "spans": spans,
            "round": 0, "total_rounds": len(spans),
        }
        return True

    def _build_replay_payload(self, client_id):
        rec = self.client_records[client_id]
        span = rec["spans"][rec["round"]]
        template = dict(span.get("request") or {})
        recorded_max = template.get("max_completion_tokens") or template.get("max_tokens")
        output_limit = self.output_length
        if isinstance(recorded_max, int) and recorded_max > 0:
            output_limit = min(recorded_max, self.output_length)
        # Preserve useful OpenAI request parameters (tools, tool_choice,
        # response_format, parallel_tool_calls), while benchmark controls win.
        template.update({
            "model": self.api_model_name,
            "messages": template.get("messages") or [],
            "max_tokens": output_limit,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        template.pop("max_completion_tokens", None)
        if self.thinking is not None:
            template["chat_template_kwargs"] = {"thinking": self.thinking}
        return template

    def _record_metrics(self, current_round, response):
        self._ensure_round_bucket(current_round)
        super()._record_metrics(current_round, response)

    def request_sender(self):
        async def request_loop():
            first_sent = 0
            next_allowed = 0.0
            while True:
                new_request = self.ready_queue.pop()
                if new_request:
                    # First client launches use the configured startup stagger.
                    if first_sent < self.num_clients and self.init_interval > 0:
                        if first_sent > 0:
                            await asyncio.sleep(self.init_interval)
                        first_sent += 1
                    # Global deterministic ceiling; inf/0 means unlimited.
                    if math.isfinite(self.request_rate) and self.request_rate > 0:
                        now = time.perf_counter()
                        if now < next_allowed:
                            await asyncio.sleep(next_allowed - now)
                        next_allowed = max(now, next_allowed) + 1.0 / self.request_rate
                    asyncio.create_task(self.handle_request(new_request))
                    self.sent_requests += 1
                else:
                    await asyncio.sleep(0.001)
                if self.pbar.n >= self.pbar.total and not self.ready_queue.requests:
                    await asyncio.sleep(0.1)
                    break

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(timeout=0.1)
                rec = self.client_records[client_id]
                current_round = rec["round"]
                if response.success:
                    self._record_metrics(current_round, response)
                    self._record_theoretical_hit(client_id, response)
                    print(
                        f"[replay] client={client_id} "
                        f"session={self.client_sessions[client_id]} "
                        f"round={current_round + 1}/{rec['total_rounds']} "
                        f"prompt_tokens={response.prompt_len} "
                        f"output_tokens={response.generated_len}"
                    )
                else:
                    print(f"Request failed for client {client_id}: {response.error}")
                self.completed_requests += 1
                rec["round"] += 1
                session_finished = rec["round"] >= rec["total_rounds"]
                if session_finished:
                    self.completed_sessions += 1
                if self.enqueued_requests < self.total_requests_target:
                    if session_finished:
                        self._claim_session(client_id)
                    self.ready_queue.append((client_id, self._build_replay_payload(client_id)))
                    self.enqueued_requests += 1
            except queue.Empty:
                if self.pbar.n >= self.pbar.total:
                    break

    def run(self):
        try:
            data = super().run()
            data["summary"].update({
                "workload_mode": "replay",
                "completed_sessions": self.completed_sessions,
                "dataset_sessions": len(self._session_ids),
                "dataset_cycles": self.dataset_cycles,
                "replay_sequential_sessions": self.replay_sequential_sessions,
            })
            return data
        finally:
            self._replay_store.close()


class TimestampReplayWorkloadGenerator(WorkloadGenerator):
    """Replay recorded timing with per-session response gating.

    A session's next turn is emitted after both the previous response has
    completed and the recorded inter-turn interval has elapsed. Therefore the
    actual send interval is max(response time, recorded timestamp delta).
    """

    def __init__(self, args):
        if args.api_format != "openai":
            raise ValueError("Timestamp replay requires --api-format openai")
        self.api_format = "openai"
        self.model_path = args.model_path
        self.api_model_name = args.api_model_name or args.model_path
        self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
        self.request_func = async_request_openai_chat_completions
        self.enable_session_key = not args.disable_session_key
        self.host_header = args.host_header
        self.api_key = args.api_key
        self.request_rate = args.request_rate
        self.init_interval = args.init_interval
        # At most one request per selected session can be in flight.
        self.max_parallel = args.num_clients
        self.output_length = args.output_length
        self.max_context_length = args.max_context_length
        self.thinking = args.thinking
        self.enable_round_barrier = False
        self.start_time = self.finished_time = None
        self.sent_requests = self.completed_requests = 0
        self._replay_store = ReplayStore(args.replay_dataset_path)
        session_ids = self._replay_store.session_ids()
        session_rng = random.Random(args.seed)
        session_rng.shuffle(session_ids)
        if not session_ids:
            raise ValueError("Timestamp replay dataset is empty")
        counts = dict(self._replay_store.conn.execute(
            "SELECT session_id, span_count FROM sessions"
        ))
        all_events = [
            {
                "session_id": session_id,
                "turn": turn,
                "timestamp": timestamp,
                "span_count": counts[session_id],
            }
            for session_id, turn, timestamp
            in self._replay_store.events_for_sessions(session_ids)
        ]
        events_by_session = {}
        for event in all_events:
            events_by_session.setdefault(event["session_id"], []).append(event)
        if not events_by_session:
            raise ValueError("No timestamp replay requests selected")

        # num_clients controls active session slots; total_requests controls
        # workload size. Pre-plan assignment so response timing cannot change
        # which sessions/spans are selected between benchmark runs.
        slot_count = min(args.num_clients, len(session_ids))
        self.session_batches = [[] for _ in range(slot_count)]
        request_target = args.total_requests
        remaining = request_target if request_target > 0 else None
        session_index = batch_index = 0
        self.dataset_cycles = 0
        self.completed_sessions = 0
        planned_events = []
        while remaining is None or remaining > 0:
            if session_index >= len(session_ids):
                if remaining is None:
                    break
                self.dataset_cycles += 1
                session_rng.shuffle(session_ids)
                session_index = 0
            session_id = session_ids[session_index]
            session_index += 1
            rows = events_by_session.get(session_id) or []
            if not rows:
                continue
            if remaining is None:
                # Compatibility mode: one complete Session per active slot.
                if batch_index >= slot_count:
                    break
                selected_rows = rows
            else:
                selected_rows = rows[:remaining]
            selected_rows = [
                {**event, "replay_instance": batch_index}
                for event in selected_rows
            ]
            is_complete = len(selected_rows) == len(rows)
            batch = {
                "session_id": session_id,
                "events": selected_rows,
                "complete": is_complete,
            }
            self.session_batches[batch_index % slot_count].append(batch)
            planned_events.extend(selected_rows)
            batch_index += 1
            if remaining is not None:
                remaining -= len(selected_rows)

        self.events = planned_events
        if not self.events:
            raise ValueError("No timestamp replay requests selected")
        self.dataset_sessions = len(session_ids)
        self.selected_sessions = batch_index
        self.total_requests = len(self.events)
        self.total_requests_target = self.total_requests
        self.num_clients = slot_count
        self.max_rounds = max(e["turn"] for e in self.events) + 1
        self.num_rounds = self.max_rounds
        self.clients_per_round = [0] * self.max_rounds
        self.performance_metrics = {
            "ttft": [], "tpot": [], "latency": [], "prompt_len": [],
            "cached_tokens": [], "cached_tokens_device": [],
            "cached_tokens_host": [], "generated_len": [],
        }
        self.cache_has_l2 = False
        for i in range(self.max_rounds):
            self.performance_metrics[f"round_{i}"] = {
                "ttft": [], "tpot": [], "latency": [], "prompt_len": [],
                "cached_tokens": [], "cached_tokens_device": [],
                "cached_tokens_host": [], "generated_len": [],
            }
        self.previous_request_metrics = {}
        self.theoretical_hit_tokens_pd_sep = 0
        self.theoretical_hit_tokens_pd_no_sep = 0
        self._responses_by_session = {}
        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=self.total_requests)
        self._session_cache = {}
        self._session_cache_limit = max(8, slot_count)

    def _load_payload(self, event):
        session_id = event["session_id"]
        if session_id not in self._session_cache:
            if len(self._session_cache) >= self._session_cache_limit:
                self._session_cache.pop(next(iter(self._session_cache)))
            self._session_cache[session_id] = self._replay_store.load_session(session_id)
        span = self._session_cache[session_id][event["turn"]]
        payload = dict(span.get("request") or {})
        recorded_max = payload.get("max_completion_tokens") or payload.get("max_tokens")
        max_tokens = self.output_length
        if isinstance(recorded_max, int) and recorded_max > 0:
            max_tokens = min(recorded_max, self.output_length)
        payload.update({
            "model": self.api_model_name,
            "messages": payload.get("messages") or [],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        payload.pop("max_completion_tokens", None)
        if self.thinking is not None:
            payload["chat_template_kwargs"] = {"thinking": self.thinking}
        return payload

    async def _send_event(
        self, client_id, event, payload, limited_by, recorded_gap, service_elapsed
    ):
        headers = {}
        if self.enable_session_key:
            headers[SESSION_ID_HEADER] = (
                f"{event['session_id']}-replay-{event.get('replay_instance', 0)}"
            )
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.host_header:
            headers["Host"] = self.host_header
        try:
            response = await self.request_func(payload, self.url, self.pbar, extra_headers=headers or None)
        except Exception as exc:
            response = RequestFuncOutput(success=False, error=str(exc))
        if self.pbar.n >= self.pbar.total:
            self.finished_time = time.perf_counter()
        self.response_queue.put(
            (client_id, event, response, limited_by, recorded_gap, service_elapsed)
        )

    def request_sender(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def sender():
            wall_start = time.perf_counter()
            next_rate_slot = wall_start
            rate_lock = asyncio.Lock()

            async def emit_at_or_after(
                client_id, event, natural_time, recorded_gap=0.0, first_turn=False
            ):
                nonlocal next_rate_slot
                now = time.perf_counter()
                service_elapsed = (
                    0.0 if first_turn else max(0.0, now - (natural_time - recorded_gap))
                )
                delay = natural_time - now
                if first_turn:
                    limited_by = "initial"
                elif delay > 0:
                    limited_by = "timestamp_delta"
                else:
                    limited_by = "service_response"
                if delay > 0:
                    await asyncio.sleep(delay)

                payload = self._load_payload(event)
                # All sessions share the optional global request-rate ceiling.
                async with rate_lock:
                    if math.isfinite(self.request_rate) and self.request_rate > 0:
                        delay = next_rate_slot - time.perf_counter()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        sent_at = time.perf_counter()
                        next_rate_slot = sent_at + 1.0 / self.request_rate
                    else:
                        sent_at = time.perf_counter()
                    self.sent_requests += 1
                    request_task = asyncio.create_task(
                        self._send_event(
                            client_id,
                            event,
                            payload,
                            limited_by,
                            recorded_gap,
                            service_elapsed,
                        )
                    )
                await request_task
                return sent_at

            async def replay_session(client_id, session_events, initial_time=None):
                previous_timestamp = None
                previous_sent_at = None
                for event in session_events:
                    first_turn = previous_timestamp is None
                    recorded_gap = 0.0
                    if first_turn:
                        natural_time = (
                            initial_time
                            if initial_time is not None
                            else time.perf_counter()
                        )
                    else:
                        recorded_gap = max(
                            0.0, event["timestamp"] - previous_timestamp
                        )
                        natural_time = previous_sent_at + recorded_gap

                    # Awaiting the response here provides the other side of
                    # max(response time, recorded_gap).
                    previous_sent_at = await emit_at_or_after(
                        client_id, event, natural_time, recorded_gap, first_turn
                    )
                    previous_timestamp = event["timestamp"]

            async def replay_slot(slot_index, batches):
                if slot_index > 0 and self.init_interval > 0:
                    await asyncio.sleep(slot_index * self.init_interval)
                for batch in batches:
                    # A newly claimed Session starts immediately. Recorded
                    # timestamps only control gaps between turns inside it.
                    await replay_session(slot_index, batch["events"])
                    if batch["complete"]:
                        self.completed_sessions += 1

            tasks = [
                asyncio.create_task(replay_slot(slot_index, batches))
                for slot_index, batches in enumerate(self.session_batches)
                if batches
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        loop.run_until_complete(sender())
        loop.close()

    def response_handler(self):
        while self.completed_requests < self.total_requests:
            try:
                (
                    client_id,
                    event,
                    response,
                    limited_by,
                    recorded_gap,
                    service_elapsed,
                ) = self.response_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if response.success:
                self._record_metrics(event["turn"], response)
                response_key = (
                    event["session_id"], event.get("replay_instance", 0)
                )
                self._responses_by_session.setdefault(response_key, []).append(
                    (event["turn"], response.prompt_len, response.generated_len)
                )
                print(
                    f"[timestamp replay] client={client_id} "
                    f"round={event['turn'] + 1}/{event['span_count']} "
                    f"limited_by={limited_by} "
                    f"recorded_gap={recorded_gap:.3f}s "
                    f"service_elapsed={service_elapsed:.3f}s "
                    f"prompt_tokens={response.prompt_len} output_tokens={response.generated_len}"
                )
            else:
                print(f"Request failed for client {client_id}: {response.error}")
            self.completed_requests += 1

    def run(self):
        try:
            data = super().run()
            hit_sep = hit_no_sep = 0
            for rows in self._responses_by_session.values():
                previous = None
                for _, prompt_len, output_len in sorted(rows):
                    if previous is not None:
                        hit_sep += min(previous[0], prompt_len)
                        hit_no_sep += min(previous[0] + previous[1], prompt_len)
                    previous = (prompt_len, output_len)
            prompt_total = sum(self.performance_metrics["prompt_len"])
            data["summary"].update({
                "workload_mode": "timestamp_replay",
                "dataset_sessions": self.dataset_sessions,
                "selected_sessions": self.selected_sessions,
                "completed_sessions": self.completed_sessions,
                "dataset_cycles": self.dataset_cycles,
                "theoretical_hit_rate_pd_sep": hit_sep / prompt_total if prompt_total else 0,
                "theoretical_hit_rate_pd_no_sep": hit_no_sep / prompt_total if prompt_total else 0,
            })
            return data
        finally:
            self._replay_store.close()


# ─── CLI ─────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-turn benchmark for sgl-router V2 (standalone, no sglang dependency). "
        "Session-aware routing is always on with a random per-run session prefix."
    )
    parser.add_argument(
        "--workload-mode",
        choices=["synthetic", "replay", "timestamp_replay"],
        default="synthetic",
        help="synthetic keeps the existing generated workload; replay consumes recorded sessions",
    )
    parser.add_argument(
        "--replay-dataset-path",
        default=str(DEFAULT_STORE),
        help="Compact SQLite session replay store",
    )
    parser.add_argument(
        "--replay-sequential-sessions",
        action="store_true",
        help="Select replay sessions deterministically by start timestamp instead "
        "of shuffling them. The ordering is preserved across dataset cycles.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=256,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10000,
        help="Maximum number of parallel requests",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=512,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=64,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="uniform",
        choices=["poisson", "uniform"],
        help="Distribution type for request emission intervals (uniform or poisson)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Average number of requests per second (default: inf = fire immediately)",
    )
    parser.add_argument(
        "--init-interval",
        type=float,
        default=0.0,
        help="Fixed gap (seconds) between sending each client's first-round "
        "request. 0 = use --request-rate for first round too. "
        "Subsequent rounds always fire immediately (response-driven).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    parser.add_argument(
        "--host-header",
        type=str,
        default="",
        help="Value for the Host request header (for LB virtual-host routing). "
        "Empty by default (no custom Host header); e.g. "
        "--host-header glm51-lb-pd.test.example.com",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Bearer API key sent as 'Authorization: Bearer <key>' on every "
        "request (bench + flush_cache + heartbeat). Empty by default "
        "(no Authorization header).",
    )
    parser.add_argument(
        "--cache-backend",
        choices=["auto", "sglang", "vllm"],
        default="auto",
        help="Backend used to reset prefix cache before each run. 'auto' "
        "detects vLLM/SGLang from /metrics and falls back to endpoint probing.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Model path or HuggingFace repo id (required).",
    )
    parser.add_argument(
        "--api-model-name",
        type=str,
        default="",
        help="Model name sent in OpenAI-format requests (the 'model' field). "
        "Empty = use --model-path. Set this when the serving backend registers the "
        "model under a different name than the local tokenizer path (e.g. vLLM "
        "--served-model-name).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="local dataset to sample tokens from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="File to log performance metrics",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--disable-random-sample",
        action="store_true",
        help="If set, disable random sampling of requests from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--enable-round-barrier",
        action="store_true",
        help="If set, only send i-th turn requests after all (i-1)-th turn requests finished.",
    )
    parser.add_argument(
        "--sub-question-input-length",
        type=int,
        default=0,
        help="Length of the sub question input for each request, if set 0 use request_length",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=0,
        help="Maximum prompt length in tokens. When > 0, prompts exceeding this "
        "are truncated (keep last N tokens). Useful with --range-ratio to "
        "ensure prompts never exceed the model's context window.",
    )
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Number of tokens shared by ALL clients at the beginning of every "
        "conversation. When > 0, a single random token sequence is generated "
        "once and prepended to every client's first message (and recycled "
        "conversations). The per-client unique portion is shortened accordingly "
        "(request_length - shared_prefix_tokens). This creates a scenario where "
        "cache_aware routing concentrates all requests on one worker, exposing "
        "KV-cache eviction vs router optimism mismatch.",
    )
    parser.add_argument(
        "--request-length2",
        type=int,
        default=0,
        help="First-round request length for the second client group. When > 0, "
        "the first half of clients use --request-length and the second half use "
        "this value, so long and short prompts mix in one run (e.g. 30k vs 5k). "
        "0 = all clients use --request-length.",
    )
    parser.add_argument(
        "--sub-question-input-length2",
        type=int,
        default=0,
        help="Per-round appended input length for the second client group "
        "(0 = same as --sub-question-input-length).",
    )
    parser.add_argument(
        "--exclude-output-from-history",
        action="store_true",
        help="Do not append the model output to the client's history; each round "
        "only appends the sub-question input, so prompts grow by exactly "
        "--sub-question-input-length tokens per round. Use for the long-context "
        "growth scenario, e.g. --request-length 30000 --sub-question-input-length "
        "1000 gives prompts of 30k, 31k, 32k, ... tokens. (Stagger mode already "
        "behaves this way.)",
    )
    parser.add_argument(
        "--ready-queue-policy",
        type=str,
        default="random",
        help="Policy for popping requests from the ready queue (random or fifo)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag of a certain run in the log file",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=0,
        help="Min rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Max rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.0,
        help="Length variation ratio for prompts and outputs, centered on the target length. "
        "0.0 = exact (no variation), 0.5 = ±50%%, 1.0 = ±100%% (maximum variation). "
        "Range is [full_len*(1-ratio), full_len*(1+ratio)].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The random seed.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="String of LoRA path. Currently we only support benchmarking on a single LoRA adaptor.",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default="openai",
        choices=["sglang", "openai"],
        help="API format to use: 'openai' for OpenAI-compatible /v1/chat/completions "
        "endpoint, 'sglang' for native /generate endpoint.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Thinking mode switch: injects chat_template_kwargs.thinking into the "
        "OpenAI payload (--thinking = true, --no-thinking = false). "
        "Default: off (thinking disabled).",
    )
    parser.add_argument(
        "--min-recycle-rounds",
        type=int,
        default=0,
        help="Min rounds per conversation segment before recycling (0 = disabled, "
        "conversation runs the full num_rounds). When enabled, a client restarts a "
        "fresh conversation after each segment until it has sent num_rounds requests.",
    )
    parser.add_argument(
        "--max-recycle-rounds",
        type=int,
        default=0,
        help="Max rounds per conversation segment before recycling (0 = disabled).",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="Decouple sending from responses to stagger prompt lengths. When set, "
        "all rounds' payloads are pre-built and enqueued into per-client ordered "
        "queues. A global sender loop randomly picks a client, pops its next round "
        "(preserving per-client round order), and fires it at the rate controlled "
        "by --request-rate (Poisson/Uniform). Responses do not gate sending — "
        "different clients sit at different rounds at the same wall-clock time "
        "-> prompt lengths mix naturally.",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=0,
        help="Total requests to send across all clients. When > 0, clients "
        "that finish all their rounds start a new session with a fresh "
        "session-id until the total is reached. 0 = disabled (one conversation "
        "per client).",
    )
    parser.add_argument(
        "--disable-session-key",
        action="store_true",
        help="Do NOT attach the X-Session-Id session header. By default the "
        "header is on (prefix-client-{id}) for session-aware routing; set this "
        "flag for baseline runs or backends whose router ignores the header.",
    )
    parser.add_argument("--monitor-warmup-ready-file", default=None)
    parser.add_argument("--monitor-continue-file", default=None)
    parser.add_argument("--monitor-benchmark-complete-file", default=None)
    parser.add_argument("--monitor-gate-timeout-file", default=None)
    parser.add_argument("--monitor-gate-timeout-seconds", type=float, default=45.0)
    return parser.parse_args()


def _touch_control_file(path):
    if not path:
        return
    control_path = Path(path)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.touch(exist_ok=True)


def _wait_for_monitor(args):
    if not args.monitor_warmup_ready_file or not args.monitor_continue_file:
        return
    _touch_control_file(args.monitor_warmup_ready_file)
    print("BENCHMARK_WAITING_FOR_MONITOR", flush=True)
    deadline = time.monotonic() + max(0.0, args.monitor_gate_timeout_seconds)
    while not Path(args.monitor_continue_file).exists():
        if time.monotonic() >= deadline:
            _touch_control_file(args.monitor_gate_timeout_file)
            print(
                "Monitor gate timed out; continuing benchmark without waiting.",
                flush=True,
            )
            break
        time.sleep(0.05)


def _detect_cache_backend(args, headers):
    """Best-effort backend detection without coupling it to the API format."""
    if args.cache_backend != "auto":
        return args.cache_backend

    # The native /generate API is SGLang-specific. OpenAI format is shared by
    # SGLang and vLLM, so it still requires metrics detection.
    if args.api_format == "sglang":
        return "sglang"

    metrics_url = f"http://{args.host}:{args.port}/metrics"
    try:
        response = requests.get(metrics_url, headers=headers, timeout=5)
        if response.ok:
            metrics = response.text.lower()
            if "vllm:" in metrics or "vllm_" in metrics:
                return "vllm"
            if "sglang_" in metrics or "sgl_" in metrics:
                return "sglang"
        else:
            print(
                f"Cache backend detection: GET /metrics returned "
                f"HTTP {response.status_code}; probing reset endpoints.",
                flush=True,
            )
    except requests.RequestException as exc:
        print(
            f"Cache backend detection: /metrics unavailable ({exc}); "
            "probing reset endpoints.",
            flush=True,
        )
    return None


def _reset_prefix_cache(args, headers):
    """Reset SGLang/vLLM prefix cache and make the outcome visible in logs."""
    backend = _detect_cache_backend(args, headers)
    endpoint_by_backend = {
        "sglang": "/flush_cache",
        "vllm": "/reset_prefix_cache",
    }

    if backend:
        candidates = [(backend, endpoint_by_backend[backend])]
    else:
        # Unknown services are probed in the historically compatible order.
        candidates = [
            ("sglang", endpoint_by_backend["sglang"]),
            ("vllm", endpoint_by_backend["vllm"]),
        ]

    errors = []
    for candidate_backend, endpoint in candidates:
        url = f"http://{args.host}:{args.port}{endpoint}"
        try:
            response = requests.post(url, headers=headers, timeout=10)
            if response.ok:
                print(
                    f"Prefix cache reset succeeded: backend={candidate_backend}, "
                    f"endpoint={endpoint}, status={response.status_code}",
                    flush=True,
                )
                return True
            errors.append(f"{endpoint}=HTTP {response.status_code}")
        except requests.RequestException as exc:
            errors.append(f"{endpoint}={exc}")

    print(
        "WARNING: prefix cache reset failed; benchmark will continue. "
        + "; ".join(errors),
        flush=True,
    )
    return False


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.disable_auto_run:
        print("Running with specified request rate...")
        request_rates = [args.request_rate]
    else:
        print("Auto-running with different request rates...")
        request_rates = [16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    headers = {}
    if args.host_header:
        headers["Host"] = args.host_header
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    headers = headers or None
    monitor_started = False
    try:
        for rate in request_rates:
            args.request_rate = rate
            _reset_prefix_cache(args, headers)
            time.sleep(1)
            generator_cls = {
                "synthetic": WorkloadGenerator,
                "replay": ReplayWorkloadGenerator,
                "timestamp_replay": TimestampReplayWorkloadGenerator,
            }[args.workload_mode]
            workload = generator_cls(args)
            if not monitor_started:
                _wait_for_monitor(args)
                monitor_started = True
            performance_data = workload.run()
            log_to_jsonl_file(performance_data, args.log_file, tag=args.tag)
    finally:
        if monitor_started:
            _touch_control_file(args.monitor_benchmark_complete_file)
            print("BENCHMARK_REQUESTS_COMPLETE", flush=True)
