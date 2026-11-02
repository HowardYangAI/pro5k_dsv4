# Perfetto 轻量 Trace

这两份文件用于浏览器端快速查看 GPU 时间线：

- `dsv4_ep8_dspark_c8_prefill_tp0_perfetto_lite.trace.json.gz`
- `dsv4_ep8_dspark_c8_decode_tp0_perfetto_lite.trace.json.gz`

使用方式：先把单个 `.trace.json.gz` 文件下载到本机，再打开
<https://ui.perfetto.dev/>，点击 **Open trace file** 选择该文件。

不要把外层 `.tar.gz` 交付包直接交给 Perfetto。

轻量版保留 GPU kernel、CUDA runtime、Profiler 标注及必要的 track 元数据；
完整 CPU 调用栈、Shape 和源码分析请使用原始 Trace 与交付 CSV。
