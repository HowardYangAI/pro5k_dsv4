# 第一阶段 KV 回收回归测试历史快照

[test_retraction_cpu_copy_capability.py](test_retraction_cpu_copy_capability.py) 从已归档的 2026-08-11 实际 `nl -ba` 工具输出恢复，包含不支持 CPU copy 时请求级 abort、支持 CPU copy 时恢复路径、DSV4 能力声明和 Scheduler 单请求终止这四项测试。

这是旧实现对应的可读源码，保留在独立目录；未应用到当前 SGLang，也未重新运行历史测试。原对话已同步，本次仅增加可直接阅读的文件。恢复时只去掉行号，105 行连续且 Python AST 校验通过。原件没有记录独立哈希，恢复文件的哈希、原始会话位置、JSONL 行号和调用 ID 见 [provenance.json](provenance.json)。
