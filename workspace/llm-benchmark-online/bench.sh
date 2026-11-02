#!/bin/bash

MODEL_PATH="/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V4-Flash"
BENCH_SCRIPT="bench_serving.py"
DATASET_PATH="/home/luban/ShareGPT_V3_unfiltered_cleaned_split.json"
HOST="10.93.161.29"
PORT="8020"
RANDOM_INPUT_LEN="32768"
RANDOM_OUTPUT_LEN="2048"
MODE="normal"                # 测试模式: normal | multi-turn
# 新增默认参数
CONCURRENCIES=(1)          # 默认并发数列表
PROMPT_MULTIPLIER=10       # 默认提示数乘数 (num-prompts = max-concurrency * 10)

# 网关配置 (使用网关时设置这些参数)
BASE_URL=""                # 网关地址 (如: https://tokenservice-gateway-ys-pre.intra.xiaojukeji.com)
API_KEY=""                 # API Key (Authorization: Bearer 的值)
REQUEST_MODEL=""           # 请求时使用的模型名 (如: TokenService/GLM-5)

# Array to store passthrough arguments
PASSTHROUGH_ARGS=()

# Usage function
usage() {
echo "用法: $0 [选项]"
echo "选项:"
echo " --model-path PATH 设置模型路径 (默认: $MODEL_PATH)"
echo " --bench-script SCRIPT 设置基准测试脚本 (默认: $BENCH_SCRIPT)"
echo " --dataset-path PATH 设置数据集路径 (默认: $DATASET_PATH)"
echo " --host HOST 设置主机地址 (默认: $HOST)"
echo " --port PORT 设置端口 (默认: $PORT)"
echo " --base-url URL 设置网关地址 (如: https://tokenservice-gateway-ys-pre.intra.xiaojukeji.com)"
echo " --api-key KEY 设置 API Key (Authorization: Bearer 的值)"
echo " --request-model NAME 设置请求时使用的模型名 (如: TokenService/GLM-5)"
echo " --random-input-len LEN 设置随机输入长度 (默认: $RANDOM_INPUT_LEN)"
echo " --random-output-len LEN 设置随机输出长度 (默认: $RANDOM_OUTPUT_LEN)"
echo " --concurrencies LIST 设置并发数列表，多个值用空格分隔 (默认: ${CONCURRENCIES[*]})"
echo " --prompt-multiplier NUM 设置num-prompts的乘数 (num-prompts = max-concurrency * NUM, 默认: $PROMPT_MULTIPLIER)"
echo " --mode MODE 测试模式 (默认: normal; multi-turn 时使用 bench_multiturn.py)"
echo " -h, --help 显示帮助信息"
echo ""
echo "透传参数:"
echo " 使用 '--' 分隔符，后面的所有参数将被直接传递给 bench_serving.py"
echo " 例如: -- --request-rate 10 --backend vllm --dataset-name sharegpt"
echo ""
echo "示例:"
echo " # 使用 host:port 方式:"
echo " $0 --model-path /path/to/model --host localhost --port 30000 --random-input-len 1024 --random-output-len 512"
echo " # 使用网关方式:"
echo " $0 --base-url https://tokenservice-gateway-ys-pre.intra.xiaojukeji.com --api-key YOUR_KEY --request-model TokenService/GLM-5"
echo " $0 --concurrencies 1 2 4 8 --prompt-multiplier 20 -- --request-rate 5 --backend vllm"
}

# Parse command line arguments
parse_args() {
while [[ $# -gt 0 ]]; do
case $1 in
--model-path)
MODEL_PATH="$2"
shift 2
;;
--bench-script)
BENCH_SCRIPT="$2"
shift 2
;;
--dataset-path)
DATASET_PATH="$2"
shift 2
;;
--host)
HOST="$2"
shift 2
;;
--port)
PORT="$2"
shift 2
;;
--base-url)
BASE_URL="$2"
shift 2
;;
--api-key)
API_KEY="$2"
shift 2
;;
--request-model)
REQUEST_MODEL="$2"
shift 2
;;
--random-input-len)
RANDOM_INPUT_LEN="$2"
shift 2
;;
--random-output-len)
RANDOM_OUTPUT_LEN="$2"
shift 2
;;
# 新增参数解析
--concurrencies)
# 将传入的多个值赋值给CONCURRENCIES数组
CONCURRENCIES=("$2")
shift 2
# 继续读取后续的并发数值，直到遇到下一个以--开头的参数或结束
while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
CONCURRENCIES+=("$1")
shift
done
;;
--prompt-multiplier)
PROMPT_MULTIPLIER="$2"
shift 2
;;
--mode)
MODE="$2"
shift 2
;;
# 透传参数分隔符
--)
shift
PASSTHROUGH_ARGS=("$@")
break
;;
-h | --help)
usage
exit 0
;;
*)
echo "未知选项: $1"
usage
exit 1
;;
esac
done
}

# Parse command line arguments FIRST
parse_args "$@"

# 设置 API Key 环境变量
if [[ -n "$API_KEY" ]]; then
    export OPENAI_API_KEY="$API_KEY"
fi

# 检查透传参数中是否已包含 --seed
HAS_SEED=false
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [[ "$arg" == "--seed" ]]; then
        HAS_SEED=true
        break
    fi
done

# ── 多轮对话测试模式 (bench_multiturn.py) ────────────────────────────────
if [[ "$MODE" == "multi-turn" ]]; then
    BENCH_SCRIPT="bench_multiturn.py"

    # 工具只支持 --host/--port：base_url 模式解析出 host/port
    if [[ -n "$BASE_URL" ]]; then
        PARSED=$(python3 -c "
import sys, urllib.parse
u = urllib.parse.urlparse('$BASE_URL')
host = u.hostname or ''
port = u.port or ('443' if u.scheme == 'https' else '80')
print(f'{host} {port}')
")
        HOST=$(echo "$PARSED" | awk '{print $1}')
        PORT=$(echo "$PARSED" | awk '{print $2}')
    fi

    BASE_CMD=(
        python3 "$BENCH_SCRIPT"
        --host "$HOST"
        --port "$PORT"
        --model-path "$MODEL_PATH"
        --dataset-path "$DATASET_PATH"
    )
    if [[ -n "$REQUEST_MODEL" ]]; then
        BASE_CMD+=(--api-model-name "$REQUEST_MODEL")
    fi
    if [[ -n "$API_KEY" ]]; then
        BASE_CMD+=(--api-key "$API_KEY")
    fi
    if [[ "$HAS_SEED" == "false" ]]; then
        BASE_CMD+=(--seed $(date +%s))
    fi

    echo "=============================================="
    echo "Running multi-turn benchmark with:"
    echo " --script: $BENCH_SCRIPT"
    echo " --target: ${HOST}:${PORT}"
    echo "=============================================="
    CMD=("${BASE_CMD[@]}" "${PASSTHROUGH_ARGS[@]}")
    echo "${CMD[@]}"
    echo "----------------------------------------------"
    "${CMD[@]}"
    echo "Multi-turn benchmark completed!"
    exit 0
fi

# 构建基础命令
BASE_CMD=(
python3 "$BENCH_SCRIPT"
--backend sglang-oai-chat
--dataset-path "$DATASET_PATH"
--dataset-name random
--random-input-len "$RANDOM_INPUT_LEN"
--random-output-len "$RANDOM_OUTPUT_LEN"
--random-range-ratio 1
--extra-request-body '{"stream_options": {"include_usage": true}}'
)

# 如果透传参数中没有 --seed，则添加随机时间戳
if [[ "$HAS_SEED" == "false" ]]; then
    BASE_CMD+=(--seed $(date +%s))
fi

# 根据是否使用网关来添加不同的参数
if [[ -n "$BASE_URL" ]]; then
    # 使用网关地址
    BASE_CMD+=(--base-url "$BASE_URL")
else
    # 使用 host:port 方式
    BASE_CMD+=(--host "$HOST" --port "$PORT")
fi

# 设置模型参数: --model 用于获取 tokenizer，--model-name 用于请求时的模型名
# MODEL_PATH: 用于获取 tokenizer
# REQUEST_MODEL: 请求时使用的模型名 (如 TokenService/GLM-5)
BASE_CMD+=(--model "$MODEL_PATH")
if [[ -n "$REQUEST_MODEL" ]]; then
    BASE_CMD+=(--model-name "$REQUEST_MODEL")
fi



# 检查透传参数中是否已包含 --max-concurrency 或 --num-prompts
# 如果已包含（如 generated-shared-prefix 模式），则不再重复添加
HAS_MAX_CONCURRENCY=false
HAS_NUM_PROMPTS=false
for ((i=0; i<${#PASSTHROUGH_ARGS[@]}; i++)); do
    if [[ "${PASSTHROUGH_ARGS[$i]}" == "--max-concurrency" ]]; then
        HAS_MAX_CONCURRENCY=true
    fi
    if [[ "${PASSTHROUGH_ARGS[$i]}" == "--num-prompts" ]]; then
        HAS_NUM_PROMPTS=true
    fi
done

# 使用可配置的并发数列表
for MAX_CONCURRENCY in "${CONCURRENCIES[@]}"; do
# 使用可配置的乘数计算NUM_PROMPTS
NUM_PROMPTS=$((PROMPT_MULTIPLIER * MAX_CONCURRENCY))

# 确定实际生效的并发数和请求数（透传参数优先）
if [[ "$HAS_MAX_CONCURRENCY" == "true" ]]; then
    EFFECTIVE_MAX_CONCURRENCY="(from passthrough args)"
else
    EFFECTIVE_MAX_CONCURRENCY="$MAX_CONCURRENCY"
fi
if [[ "$HAS_NUM_PROMPTS" == "true" ]]; then
    EFFECTIVE_NUM_PROMPTS="(from passthrough args)"
else
    EFFECTIVE_NUM_PROMPTS="$NUM_PROMPTS"
fi

echo "=============================================="
echo "Running benchmark with:"
echo " --max-concurrency: $EFFECTIVE_MAX_CONCURRENCY"
echo " --num-prompts: $EFFECTIVE_NUM_PROMPTS"
echo "=============================================="

# complete cmd: 只在透传参数未提供时才添加 bench.sh 的默认值
CMD=("${BASE_CMD[@]}")
if [[ "$HAS_MAX_CONCURRENCY" == "false" ]]; then
    CMD+=(--max-concurrency "$MAX_CONCURRENCY")
fi
if [[ "$HAS_NUM_PROMPTS" == "false" ]]; then
    CMD+=(--num-prompts "$NUM_PROMPTS")
fi
CMD+=("${PASSTHROUGH_ARGS[@]}")

echo "${CMD[@]}"
echo "----------------------------------------------"
"${CMD[@]}"

sleep 1
done

echo "All benchmark tests completed!"
