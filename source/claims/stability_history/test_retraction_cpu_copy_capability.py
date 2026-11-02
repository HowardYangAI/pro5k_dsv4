from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


def _req(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1],
        output_ids=[2],
        priority=None,
        to_finish=None,
    )


def _batch(cpu_copy_supported: bool):
    batch = ScheduleBatch(reqs=[_req("0"), _req("1")])
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        supports_cpu_copy=Mock(return_value=cpu_copy_supported)
    )
    batch._get_decode_retraction_order = Mock(return_value=[0, 1])
    batch.check_decode_mem = Mock(return_value=True)
    batch.release_req = Mock()
    batch.filter_batch = Mock()
    return batch


@patch(
    "sglang.srt.managers.schedule_batch."
    "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
    return_value=0.5,
)
def test_unsupported_pool_aborts_retracted_request_without_cpu_copy(_estimate):
    batch = _batch(cpu_copy_supported=False)
    args = SimpleNamespace(disaggregation_mode="decode")

    retracted, ratio, aborted = batch.retract_decode(args)

    assert retracted == []
    assert ratio == 0.5
    assert [req.rid for req in aborted] == ["1"]
    assert isinstance(aborted[0].to_finish, FINISH_ABORT)
    assert "does not support CPU save/restore" in aborted[0].to_finish.message
    batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)


@patch(
    "sglang.srt.managers.schedule_batch."
    "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
    return_value=0.5,
)
def test_supported_pool_keeps_existing_cpu_resume_path(_estimate):
    batch = _batch(cpu_copy_supported=True)
    args = SimpleNamespace(disaggregation_mode="decode")

    retracted, ratio, aborted = batch.retract_decode(args)

    assert [req.rid for req in retracted] == ["1"]
    assert ratio == 0.5
    assert aborted == []
    batch.release_req.assert_called_once_with(1, 1, args, offload_kv=True)


def test_deepseek_v4_pool_reports_cpu_copy_unsupported():
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    assert pool.supports_cpu_copy() is False


def test_scheduler_finalizes_only_the_unrecoverable_request():
    req = _req("abort")
    abort_reason = FINISH_ABORT(
        "KV cache cannot be restored",
        status_code=500,
        err_type="InternalServerError",
    )
    req.to_finish = abort_reason
    req.return_logprob = False
    req.time_stats = SimpleNamespace(
        trace_ctx=SimpleNamespace(abort=Mock()),
        set_completion_time=Mock(),
    )

    scheduler = object.__new__(Scheduler)
    scheduler.decode_offload_manager = Mock()
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=Mock())
    )

    scheduler._finalize_retraction_abort(req)

    assert req.to_finish is None
    assert req.finished_reason is abort_reason
    req.time_stats.trace_ctx.abort.assert_called_once_with(abort_info=abort_reason)
    req.time_stats.set_completion_time.assert_called_once_with()
    scheduler.decode_offload_manager.finalize_release_on_finish.assert_called_once_with(
        req
    )
    abort_req, sent_req = (
        scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args
    )
    assert sent_req is req
    assert abort_req.finished_reason == abort_reason.to_json()
