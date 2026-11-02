#!/usr/bin/env python3
"""
Benchmark Results Parser
解析benchmark结果文件并生成CSV或Markdown文件

使用方法:
python benchmark_result_parse.py input_file.txt output_file.csv --format csv
python benchmark_result_parse.py input_file.txt output_file.md --format markdown
"""

import argparse
import csv
import os
import re
import sys
from typing import Any, Dict, List

# 默认要提取的字段列表
DEFAULT_FIELDS = [
    "Traffic request rate",
    "Successful requests",
    "Max request concurrency",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P95 TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P95 TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean E2E Latency (ms)",
    "Request throughput (req/s)",
    "Input token throughput (tok/s)",
    "Output token throughput (tok/s)",
    "Concurrency",
    "Cache hit rate",
    "Accept length",
    "Total input tokens",
    "Total generated tokens",
    # Multi-turn conversation test fields
    "Completed conversations",
    "Conversation throughput (conv/s)",
    "Mean conversation duration (s)",
    "Mean turns per conversation",
]


def generate_output_filename(input_file: str, format_type: str) -> str:
    """
    根据输入文件名和格式生成输出文件名

    Args:
        input_file: 输入文件路径
        format_type: 输出格式

    Returns:
        生成的输出文件名
    """
    # 获取输入文件的基础名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # 根据格式确定扩展名
    if format_type.lower() in ["markdown", "md"]:
        extension = ".md"
    else:
        extension = ".csv"

    return base_name + extension


def check_file_exists(file_path: str) -> bool:
    """
    检查文件是否存在

    Args:
        file_path: 文件路径

    Returns:
        True如果文件存在，False如果不存在
    """
    return os.path.exists(file_path)


def parse_benchmark_file(file_path: str) -> List[Dict[str, Any]]:
    """
    解析benchmark结果文件，提取所有测试结果

    Args:
        file_path: 输入文件路径

    Returns:
        包含所有测试结果的列表
    """

    # 使用统一的字段定义
    fields_to_extract = DEFAULT_FIELDS[:]

    results = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 使用正则表达式找到所有的benchmark结果块
    pattern = r"============ Serving Benchmark Result ============\s*\n(.*?)\s*\n=================================================="
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        result = {}
        lines = match.strip().split("\n")

        for line in lines:
            # 每行的格式是 "字段名:值"
            if ":" in line:
                # 分割字段名和值
                parts = line.split(":", 1)
                if len(parts) == 2:
                    field_name = parts[0].strip()
                    field_value = parts[1].strip()

                    # 如果这个字段是我们需要提取的
                    if field_name in fields_to_extract:
                        # 尝试转换为数值，如果失败就保持字符串
                        # 特殊处理 inf/INF 值
                        if field_value.lower() == "inf":
                            result[field_name] = "INF"
                        else:
                            try:
                                # 移除可能的逗号（千位分隔符）
                                clean_value = field_value.replace(",", "")
                                if "." in clean_value:
                                    result[field_name] = float(clean_value)
                                else:
                                    result[field_name] = int(clean_value)
                            except ValueError:
                                result[field_name] = field_value

        # 只添加非空的结果
        if result:
            results.append(result)

    return results


def write_to_csv(results: List[Dict[str, Any]], output_file: str, fields: List[str]):
    """
    将结果写入CSV文件

    Args:
        results: 解析结果列表
        output_file: 输出CSV文件路径
        fields: 字段列表（CSV的列标题）
    """

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # 写入表头
        writer.writeheader()

        # 写入每一行数据
        for result in results:
            row = {}
            for field in fields:
                row[field] = result.get(field, "")
            writer.writerow(row)


def write_to_markdown(
    results: List[Dict[str, Any]],
    output_file: str,
    fields: List[str],
    title: str = "Benchmark Results",
):
    """
    将结果写入Markdown文件

    Args:
        results: 解析结果列表
        output_file: 输出Markdown文件路径
        fields: 字段列表（Markdown表格的列标题）
        title: Markdown文档标题
    """

    with open(output_file, "w", encoding="utf-8") as mdfile:
        # 写入标题
        mdfile.write(f"# {title}\n\n")

        # 创建表格
        # 写入表头
        header = "| " + " | ".join(fields) + " |\n"
        mdfile.write(header)

        # 写入分隔线
        separator = "| " + " | ".join(["---"] * len(fields)) + " |\n"
        mdfile.write(separator)

        # 写入数据行
        for i, result in enumerate(results, 1):
            row_values = []
            for field in fields:
                value = result.get(field, "")
                # 格式化数值显示
                if isinstance(value, float):
                    row_values.append(f"{value:.2f}")
                else:
                    row_values.append(str(value))

            row = "| " + " | ".join(row_values) + " |\n"
            mdfile.write(row)


def write_results(
    results: List[Dict[str, Any]],
    output_file: str,
    fields: List[str],
    format_type: str,
    title: str = "Benchmark Results",
):
    """
    根据格式类型写入结果文件

    Args:
        results: 解析结果列表
        output_file: 输出文件路径
        fields: 字段列表
        format_type: 输出格式 ('csv' 或 'markdown')
        title: Markdown文档标题（仅在markdown格式时使用）
    """
    if format_type.lower() == "csv":
        write_to_csv(results, output_file, fields)
    elif format_type.lower() in ["markdown", "md"]:
        write_to_markdown(results, output_file, fields, title)
    else:
        raise ValueError(f"不支持的输出格式: {format_type}. 支持的格式: csv, markdown")


def main():
    parser = argparse.ArgumentParser(
        description="解析benchmark结果文件并生成CSV或Markdown文件"
    )
    parser.add_argument(
        "-i", "--input-file", required=True, help="输入的benchmark结果文件路径"
    )
    parser.add_argument(
        "-o", "--output-file", help="输出文件路径 (可选，默认为输入文件同名格式)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown", "md"],
        default="markdown",
        help="输出格式 (默认: markdown)",
    )
    parser.add_argument(
        "--title",
        default="Benchmark Results",
        help="Markdown文档标题 (仅在markdown格式时使用)",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS[:],
        help="要提取的字段列表",
    )

    args = parser.parse_args()

    try:
        # 解析benchmark文件
        print(f"正在解析文件: {args.input_file}")
        results = parse_benchmark_file(args.input_file)

        if not results:
            print("警告: 没有找到任何benchmark结果!")
            return

        print(f"找到 {len(results)} 个benchmark结果")

        # 确定输出文件名
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = generate_output_filename(args.input_file, args.format)
            print(f"自动生成输出文件名: {output_file}")

        # 检查文件是否已存在
        if check_file_exists(output_file):
            print(f"错误: 输出文件 '{output_file}' 已存在，拒绝覆盖！")
            print("请指定不同的输出文件名或删除已存在的文件。")
            sys.exit(1)

        # 根据格式写入文件
        format_name = "CSV" if args.format == "csv" else "Markdown"
        print(f"正在生成{format_name}文件: {output_file}")
        write_results(results, output_file, args.fields, args.format, args.title)

        print(f"{format_name}文件生成完成!")

        # 显示结果摘要
        print("\n结果摘要:")
        for i, result in enumerate(results, 1):
            concurrency = result.get("Max request concurrency", "N/A")
            throughput = result.get("Request throughput (req/s)", "N/A")
            print(f"  测试 {i}: 并发数={concurrency}, 吞吐量={throughput} req/s")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()