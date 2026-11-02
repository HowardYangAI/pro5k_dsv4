# llm测试规划平台
主要设计内容
1.测试不同最该并发，不同请求率，不同输入输出长度下的性能
2.使用网页界面进行任务编排。支持批量化操作
3.基于现有的系统，套上一层网页界面。


# 现有命令行系统
参考
/home/luban/benchmark/all_start_dsv4.sh 可输出结果
/home/luban/benchmark/bench_result_parser.py 可解析出结构化结果
可以通过查/home/luban/benchmark/bk_hw的一对txt和md来确认结果

# web系统
使用fastapi。
1.支持历史结果回看。包括历史配置和历史结果
2.支持url port和base_api + tokens 两种方式


