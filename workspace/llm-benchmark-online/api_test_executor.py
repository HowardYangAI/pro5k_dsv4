"""OpenAI API 测试执行器

执行 API 测试用例并返回结果。
"""

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

import httpx

from api_test_cases import TestCase, get_test_cases


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    category: str
    status: str  # pass, fail, error
    http_code: int
    response_time_ms: int
    response_body: str
    error_message: str
    assertion_details: str

    def to_dict(self) -> dict:
        return asdict(self)


class APITestExecutor:
    """OpenAI API 测试执行器"""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        reasoning_model: str = "",
        fc_model: str = "",
        api_key: str = "",
        enable_fc: bool = False,
        enable_n: bool = False,
        n_value: int = 3,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_model = reasoning_model or model
        self.fc_model = fc_model or model
        self.api_key = api_key
        self.enable_fc = enable_fc
        self.enable_n = enable_n
        self.n_value = n_value
        self.timeout = timeout

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _render_request_body(self, body: dict) -> dict:
        """渲染请求体中的模板变量"""
        body_str = json.dumps(body)
        body_str = body_str.replace("{model}", self.model)
        body_str = body_str.replace("{reasoning_model}", self.reasoning_model)
        body_str = body_str.replace("{fc_model}", self.fc_model)
        body_str = body_str.replace("{n_value}", str(self.n_value))
        return json.loads(body_str)

    async def run_single_test(self, test_case: TestCase) -> TestResult:
        """执行单个测试用例"""
        url = self.base_url + test_case.endpoint
        headers = self._get_headers()
        method = test_case.method.upper()

        # 准备请求体
        body = None
        if test_case.request_body and method == "POST":
            body = self._render_request_body(test_case.request_body)

        start_time = time.time()
        http_code = 0
        response_body = ""
        error_message = ""
        assertion_details = ""
        status = "error"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method == "GET":
                    resp = await client.get(url, headers=headers)
                else:
                    resp = await client.post(url, headers=headers, json=body)

                http_code = resp.status_code
                response_body = resp.text
                elapsed_ms = int((time.time() - start_time) * 1000)

                # 验证响应
                success, details = self._validate_response(test_case, http_code, response_body)
                status = "pass" if success else "fail"
                assertion_details = details

        except httpx.TimeoutException:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_message = "请求超时"
            status = "error"
        except httpx.RequestError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_message = f"请求失败: {str(e)}"
            status = "error"
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_message = f"未知错误: {str(e)}"
            status = "error"

        return TestResult(
            test_name=test_case.name,
            category=test_case.category,
            status=status,
            http_code=http_code,
            response_time_ms=elapsed_ms,
            response_body=response_body,
            error_message=error_message,
            assertion_details=assertion_details,
        )

    def _validate_response(self, test_case: TestCase, http_code: int, response_body: str) -> tuple[bool, str]:
        """验证响应"""
        # 首先检查 HTTP 状态码
        if http_code != 200:
            return False, f"HTTP 状态码错误: {http_code} (期望: 200)"

        if test_case.check_type == "http_code":
            return True, "HTTP 200"

        if test_case.check_type == "json_field":
            return self._check_json_field(test_case.check_field, response_body)

        if test_case.check_type == "custom":
            return self._check_custom(test_case.check_field, response_body)

        return True, "默认验证通过"

    def _check_json_field(self, field_path: Optional[str], response_body: str) -> tuple[bool, str]:
        """检查 JSON 字段是否存在"""
        if not field_path:
            return True, "无字段检查要求"

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            return False, "响应不是有效的 JSON 格式"

        # 解析字段路径 (如 "choices[0].message.content")
        parts = re.split(r'\.|\[|\]', field_path)
        parts = [p for p in parts if p]

        current = data
        path_so_far = []
        for part in parts:
            path_so_far.append(part)
            if part.isdigit():
                idx = int(part)
                if not isinstance(current, list) or idx >= len(current):
                    return False, f"字段 {'.'.join(path_so_far)} 不存在"
                current = current[idx]
            else:
                if not isinstance(current, dict) or part not in current:
                    return False, f"字段 {'.'.join(path_so_far)} 不存在"
                current = current[part]

        # 检查值是否有效
        if current is None:
            return False, f"字段 {field_path} 值为 null"
        if isinstance(current, list) and len(current) == 0:
            return False, f"字段 {field_path} 为空数组"

        return True, f"字段 {field_path} 存在且有效"

    def _check_custom(self, check_field: Optional[str], response_body: str) -> tuple[bool, str]:
        """自定义验证逻辑"""
        if not check_field:
            return True, "无自定义检查"

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError:
            return False, "响应不是有效的 JSON 格式"

        if check_field == "reasoning_string_think":
            # 验证 reasoning_content/context 为 null 且 content 中无 <think> 标签
            msg = data.get("choices", [{}])[0].get("message", {})
            rc = msg.get("reasoning_content") or msg.get("reasoning_context")
            content = msg.get("content", "") or ""
            has_think = "<think" in content.lower()

            if rc is None and not has_think:
                return True, "reasoning_content/context 为 null 且无 <think> 标签"
            else:
                details = []
                if rc is not None:
                    details.append(f"reasoning_content/context 不为 null: {type(rc).__name__}")
                if has_think:
                    details.append("content 中包含 <think> 标签")
                return False, "; ".join(details)

        if check_field == "json_content":
            # 验证 content 是有效的 JSON
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                json.loads(content)
                return True, f"响应内容为有效 JSON: {content[:50]}..."
            except json.JSONDecodeError:
                return False, f"响应内容不是有效 JSON: {content[:100]}"

        if check_field == "n_choices":
            # 验证 choices 数组长度等于 n
            choices = data.get("choices", [])
            n = self.n_value
            if len(choices) == n:
                # 验证每个 choice 都有必要字段
                for i, c in enumerate(choices):
                    if "index" not in c or "message" not in c:
                        return False, f"choice {i} 缺少 index 或 message 字段"
                return True, f"choices 数组长度 = {n}，所有 choice 包含必要字段"
            else:
                return False, f"choices 数组长度 = {len(choices)} (期望: {n})"

        return True, f"未知检查类型: {check_field}"

    async def run_all_tests(self) -> list[TestResult]:
        """执行所有测试用例"""
        test_cases = self.get_test_cases_list()
        results = []

        for tc in test_cases:
            result = await self.run_single_test(tc)
            results.append(result)

        return results

    def get_test_cases_list(self) -> list[TestCase]:
        """获取需要执行的测试用例列表"""
        test_cases = get_test_cases(enable_fc=self.enable_fc, enable_n=self.enable_n)
        filtered = []
        for tc in test_cases:
            if not tc.enabled and not (tc.requires_fc and self.enable_fc) and not (tc.requires_n and self.enable_n):
                continue
            filtered.append(tc)
        return filtered

    def get_summary(self, results: list[TestResult]) -> dict:
        """生成测试摘要"""
        total = len(results)
        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        error = sum(1 for r in results if r.status == "error")

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "pass_rate": f"{passed / total * 100:.1f}%" if total > 0 else "0%",
            "results": [r.to_dict() for r in results],
        }
