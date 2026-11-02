# Official main vs PR #36185 DSV4 retraction A/B test

## Versions

- Baseline: official `main` at `e51a3ae65e3401b21de860c90a64102133d2d6a6`.
- Candidate: baseline plus community PR #36185 (`fix(pd): make DSV4 retraction safe`).
- Candidate worktree: `/sgl-workspace/sglang-official-pr36185`.

## Goal

Use identical DeepSeek-V4-Flash DSPARK PD/fake pressure workloads to compare:

1. Baseline: DSV4 selects the `cpu_tensor` retraction backend, reaches an
   unsupported `get_cpu_copy`, and may crash the Decode scheduler.
2. Candidate: an unsupported backup returns failure, only the affected request
   is aborted, and the Decode scheduler remains healthy.

## Configuration

- Prefill: `10.207.193.156:31001`, TP8/DP8/EP4, MF 0.70.
- Decode: `10.207.193.166:30002`, TP8/DP8/EP4, MF 0.55.
- Router: `10.207.193.156:30000`.
- Model: DeepSeek-V4-Flash-0731, DSPARK, 6 draft tokens.
- Page size: 256; radix cache disabled.

## Acceptance evidence

- Pressure reaches the Decode retraction path.
- Baseline and candidate logs are archived separately.
- Candidate has request-level aborts with no `NotImplementedError`, scheduler
  exception, or subprocess crash, and `/health` remains 200.

