"""OpenAI API 测试用例定义

将 customer_api.sh 中的测试用例转换为 Python 结构。
每个测试用例定义请求参数和验证规则。
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TestCase:
    """单个测试用例"""
    category: str  # 分类: generate, completions, chat, function_call, reasoning, health, bad_case
    name: str  # 测试名称
    endpoint: str  # API 端点
    method: str = "POST"  # HTTP 方法
    request_body: dict = field(default_factory=dict)  # 请求体
    check_type: str = "http_code"  # 验证类型: http_code, json_field, custom
    check_field: Optional[str] = None  # 需要检查的字段
    enabled: bool = True  # 是否启用
    description: str = ""  # 测试描述
    requires_fc: bool = False  # 是否需要启用 FC 测试
    requires_n: bool = False  # 是否需要启用 N 参数测试

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "endpoint": self.endpoint,
            "method": self.method,
            "request_body": self.request_body,
            "check_type": self.check_type,
            "check_field": self.check_field,
            "enabled": self.enabled,
            "description": self.description,
            "requires_fc": self.requires_fc,
            "requires_n": self.requires_n,
        }


# 工具定义
TOOLS_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "date": {"type": "string", "description": "The date in format YYYY-mm-dd"}
                },
                "required": ["location", "date"]
            }
        }
    }
]

TOOL_DRIVER_MATCH = [{
    "type": "function",
    "function": {
        "name": "driver_match",
        "description": "发起行程规划，进行司机匹配",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
    }
}]

TOOL_SEARCH_WEB = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "如果用户消息超过你的知识范围或实时性要求较高。",
        "parameters": {
            "additionalProperties": False,
            "properties": {
                "query_input_lst": {
                    "description": "不同侧重点的关键搜索词列表。",
                    "items": {"type": "string"},
                    "title": "Query Input Lst",
                    "type": "array"
                }
            },
            "required": ["query_input_lst"],
            "title": "search_web_args",
            "type": "object"
        }
    }
}]


# 所有测试用例定义
DEFAULT_TEST_CASES: list[TestCase] = [
    # ============ /generate 接口测试 ============
    TestCase(
        category="generate",
        name="/generate",
        endpoint="/generate",
        request_body={"text": "Hello, world", "max_tokens": 2048},
    ),
    TestCase(
        category="generate",
        name="/generate stream",
        endpoint="/generate",
        request_body={"text": "Hello, world", "max_tokens": 2048, "stream": True, "stream_options": {"include_usage": True}},
    ),
    TestCase(
        category="generate",
        name="/generate logprobs",
        endpoint="/generate",
        request_body={"text": "Hello, world", "max_tokens": 2048, "stream": True, "stream_options": {"include_usage": True}, "logprobs": 3},
    ),

    # ============ /v1/completions 接口测试 ============
    TestCase(
        category="completions",
        name="/v1/completions",
        endpoint="/v1/completions",
        request_body={"model": "{model}", "prompt": "What is the capital of France?", "max_tokens": 2048},
    ),
    TestCase(
        category="completions",
        name="/v1/completions stream",
        endpoint="/v1/completions",
        request_body={"model": "{model}", "prompt": "What is the capital of France?", "max_tokens": 2048, "stream": True, "stream_options": {"include_usage": True}},
    ),
    TestCase(
        category="completions",
        name="/v1/completions logprobs",
        endpoint="/v1/completions",
        request_body={"model": "{model}", "prompt": "What is the capital of France?", "max_tokens": 2048, "stream": True, "stream_options": {"include_usage": True}, "logprobs": 3},
    ),

    # ============ /v1/chat/completions 接口测试 ============
    TestCase(
        category="chat",
        name="/v1/chat/completions",
        endpoint="/v1/chat/completions",
        request_body={"model": "{model}", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
        check_type="json_field",
        check_field="choices[0].message.content",
    ),
    TestCase(
        category="chat",
        name="/v1/chat/completions stream",
        endpoint="/v1/chat/completions",
        request_body={"model": "{model}", "messages": [{"role": "user", "content": "What is the capital of France?"}], "stream": True, "stream_options": {"include_usage": True}},
    ),
    TestCase(
        category="chat",
        name="/v1/chat/completions logprobs",
        endpoint="/v1/chat/completions",
        request_body={"model": "{model}", "messages": [{"role": "user", "content": "What is the capital of France?"}], "stream": True, "stream_options": {"include_usage": True}, "logprobs": True},
    ),
    TestCase(
        category="chat",
        name="/v1/chat/completions 多轮对话",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{model}",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "And what is the population of that city?"}
            ]
        },
        check_type="json_field",
        check_field="choices[0].message.content",
    ),

    # ============ Function Call 测试 ============
    TestCase(
        category="function_call",
        name="FC 输入兼容1 - 无content",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{fc_model}",
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "# 你是一个专注于打车服务与回复打车价格、距离和时长的助手"},
                {"role": "assistant", "tool_calls": [{"id": "0", "type": "function", "function": {"name": "driver_match", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "0", "content": "小滴正在为您寻找司机，请稍等一会。起点：万科天空之城B座东北侧，终点：安狮钻石大厦(软件园三号路)。"}
            ],
            "tools": TOOL_DRIVER_MATCH
        },
        enabled=False,
        requires_fc=True,
    ),
    TestCase(
        category="function_call",
        name="FC 输入兼容2 - 空字符串content",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{fc_model}",
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "# 你是一个专注于打车服务与回复打车价格、距离和时长的助手"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "0", "type": "function", "function": {"name": "driver_match", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "0", "content": "小滴正在为您寻找司机，请稍等一会。起点：万科天空之城B座东北侧，终点：安狮钻石大厦(软件园三号路)。"}
            ],
            "tools": TOOL_DRIVER_MATCH
        },
        enabled=False,
        requires_fc=True,
    ),
    TestCase(
        category="function_call",
        name="FC 输入兼容3 - null content",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{fc_model}",
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "# 你是一个专注于打车服务与回复打车价格、距离和时长的助手"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "0", "type": "function", "function": {"name": "driver_match", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "0", "content": "小滴正在为您寻找司机，请稍等一会。起点：万科天空之城B座东北侧，终点：安狮钻石大厦(软件园三号路)。"}
            ],
            "tools": TOOL_DRIVER_MATCH
        },
        enabled=False,
        requires_fc=True,
    ),
    TestCase(
        category="function_call",
        name="FC 天气查询工具调用",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{fc_model}",
            "messages": [{"role": "user", "content": "How's the weather in Hangzhou Tomorrow"}],
            "tools": TOOLS_WEATHER,
            "temperature": 0.0
        },
        check_type="json_field",
        check_field="choices[0].message.tool_calls",
        enabled=False,
        requires_fc=True,
    ),

    # ============ Reasoning 测试 ============
    TestCase(
        category="reasoning",
        name="Reasoning way1 - reasoning object",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{reasoning_model}",
            "messages": [{"role": "user", "content": "你是谁"}],
            "reasoning": {"enabled": True, "effort": "high"},
            "max_tokens": 2048,
            "temperature": 0.7
        },
        check_type="json_field",
        check_field="choices[0].message.reasoning_content",
    ),
    TestCase(
        category="reasoning",
        name="Reasoning way2 - reasoning_effort",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{reasoning_model}",
            "messages": [{"role": "user", "content": "你是谁"}],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking": True},
            "max_tokens": 2048,
            "temperature": 0.7
        },
        check_type="json_field",
        check_field="choices[0].message.reasoning_content",
    ),
    TestCase(
        category="reasoning",
        name="Reasoning string think",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{reasoning_model}",
            "messages": [{"role": "user", "content": "你是谁"}],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking": "true"},
            "max_tokens": 2048,
            "temperature": 0.7
        },
        check_type="custom",
        check_field="reasoning_string_think",
    ),
    TestCase(
        category="reasoning",
        name="Reasoning way3 - stream",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{reasoning_model}",
            "messages": [{"role": "user", "content": "你是谁"}],
            "reasoning": {"enabled": True, "effort": "medium"},
            "max_tokens": 2048,
            "temperature": 0.7,
            "stream": True,
            "stream_reasoning": True
        },
    ),

    # ============ Response Format 测试 ============
    TestCase(
        category="response_format",
        name="response_format json_object",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{reasoning_model}",
            "messages": [{"role": "user", "content": "1+1"}],
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "stream": False
        },
        check_type="custom",
        check_field="json_content",
    ),

    # ============ N 参数测试 ============
    TestCase(
        category="n_parameter",
        name=f"n parameter test",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{model}",
            "messages": [{"role": "user", "content": "What is 1+1? Please answer briefly."}],
            "max_tokens": 2048,
            "temperature": 0.7,
            "n": "{n_value}",
            "stream": False
        },
        check_type="custom",
        check_field="n_choices",
        enabled=False,
        requires_n=True,
    ),

    # ============ 服务状态接口测试 ============
    TestCase(category="health", name="/liveness", endpoint="/liveness", method="GET"),
    TestCase(category="health", name="/readiness", endpoint="/readiness", method="GET"),
    TestCase(category="health", name="/health", endpoint="/health", method="GET"),
    TestCase(category="health", name="/health_generate", endpoint="/health_generate", method="GET"),
    TestCase(category="health", name="/v1/models", endpoint="/v1/models", method="GET"),
    TestCase(category="health", name="/get_model_info", endpoint="/get_model_info", method="GET"),
    TestCase(category="health", name="/get_server_info", endpoint="/get_server_info", method="GET"),
    TestCase(category="health", name="/get_loads", endpoint="/get_loads", method="GET"),
    TestCase(category="health", name="/list_workers", endpoint="/list_workers", method="GET"),
    TestCase(category="health", name="/flush_cache", endpoint="/flush_cache", method="GET"),

    # ============ Bad Case 测试 ============
    TestCase(
        category="bad_case",
        name="high logprobs boundary",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "default",
            "messages": [{"role": "user", "content": "how are you"}],
            "logprobs": True,
            "max_tokens": 1,
            "top_logprobs": 200
        },
    ),
    TestCase(
        category="bad_case",
        name="safe function call",
        endpoint="/v1/chat/completions",
        request_body={
            "model": "{model}",
            "messages": [
                {"role": "user", "content": "## 你是一个有用打车助手"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "0",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query_input_lst":["杭州东站 西溪谷 地址","北海 到 杭州东站 西溪谷 打车路","杭州东站 西溪谷 滴滴打车 最近车辆"]}'
                        }
                    }]
                },
                {"role": "tool", "content": "发现一个黑色专车,车牌号是浙A00001,价格100元", "tool_call_id": "0"},
                {
                    "role": "system",
                    "content": "You are a helpful assistant with tool calling capabilities. When a tool call is needed, you MUST use the following format to issue the call:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n```json\n{\"param1\": \"value1\", \"param2\": \"value2\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\nMake sure the JSON is valid.\n## Tools\n\n### Function\n\nYou have the following functions available:\n\n- `search_web`:\n```json\n{\"type\":\"function\",\"function\":{\"name\":\"search_web\",\"description\":\"如果用户消息超过你的知识范围或实时性要求较高，。\",\"parameters\":{\"additionalProperties\":false,\"properties\":{\"query_input_lst\":{\"description\":\"不同侧重点的关键搜索词列表。\",\"items\":{\"type\":\"string\"},\"title\":\"Query Input Lst\",\"type\":\"array\"}},\"required\":[\"query_input_lst\"],\"title\":\"search_web_args\",\"type\":\"object\"}}}\n```\n"
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
            "stream": False,
            "stop": [],
            "tool_choice": "none",
            "stream_options": {"include_usage": True},
            "store": True
        },
    ),
]


def get_test_cases(enable_fc: bool = False, enable_n: bool = False) -> list[TestCase]:
    """获取测试用例列表，根据配置过滤"""
    cases = []
    for tc in DEFAULT_TEST_CASES:
        if tc.requires_fc and not enable_fc:
            continue
        if tc.requires_n and not enable_n:
            continue
        cases.append(tc)
    return cases


def get_test_case_by_name(name: str) -> Optional[TestCase]:
    """根据名称获取测试用例"""
    for tc in DEFAULT_TEST_CASES:
        if tc.name == name:
            return tc
    return None
