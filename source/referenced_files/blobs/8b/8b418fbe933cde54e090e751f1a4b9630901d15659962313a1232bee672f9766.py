import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch, release_req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMXFP8,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedMHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1],
        output_ids=[2],
        priority=None,
        to_finish=None,
        return_logprob=False,
        time_stats=SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=Mock()),
            set_completion_time=Mock(),
        ),
    )


def _server_args():
    return SimpleNamespace(
        disaggregation_mode="decode",
        retraction_policy="decode_length",
        schedule_low_priority_values_first=False,
    )


def _batch(cpu_copy_supported: bool, req_count: int = 2):
    batch = ScheduleBatch(reqs=[_req(str(i)) for i in range(req_count)])
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        supports_cpu_copy=Mock(return_value=cpu_copy_supported)
    )
    batch._get_decode_retraction_order = Mock(return_value=list(range(req_count)))
    batch.check_decode_mem = Mock(return_value=True)
    batch.release_req = Mock()
    batch.filter_batch = Mock()
    return batch


class TestKVCacheCPUCopyCapability(CustomTestCase):
    def test_mha_support_requires_unscaled_nhd_layout(self):
        pool = object.__new__(MHATokenToKVPool)
        pool.kv_cache_layout = "nhd"
        pool.k_scale_buffer = None
        pool.v_scale_buffer = None
        self.assertTrue(pool.supports_cpu_copy())

        pool.kv_cache_layout = "hnd"
        self.assertFalse(pool.supports_cpu_copy())

        pool.kv_cache_layout = "nhd"
        pool.k_scale_buffer = object()
        self.assertFalse(pool.supports_cpu_copy())

    def test_incomplete_mha_variants_report_unsupported(self):
        pools = (
            object.__new__(NoOpMHATokenToKVPool),
            object.__new__(MHATokenToKVPoolFP4),
            object.__new__(MHATokenToKVPoolMXFP8),
            object.__new__(PageMajorMHATokenToKVPool),
            object.__new__(UnifiedMHATokenToKVPool),
        )
        for pool in pools:
            with self.subTest(pool=type(pool).__name__):
                self.assertFalse(pool.supports_cpu_copy())

    def test_mla_and_deepseek_v4_capabilities(self):
        self.assertTrue(object.__new__(MLATokenToKVPool).supports_cpu_copy())
        self.assertFalse(object.__new__(MLATokenToKVPoolFP4).supports_cpu_copy())
        self.assertFalse(object.__new__(DeepSeekV4TokenToKVPool).supports_cpu_copy())

    def test_swa_requires_both_subpools(self):
        pool = object.__new__(SWAKVPool)
        pool.full_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        pool.swa_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        self.assertTrue(pool.supports_cpu_copy())

        pool.swa_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: False)
        self.assertFalse(pool.supports_cpu_copy())


class TestPDDecodeRetractionCPUCopy(CustomTestCase):
    def test_runtime_unsupported_backup_aborts_safely(self):
        req = Mock(rid="unsupported-at-runtime")
        req.finished.return_value = False
        req.offload_kv_cache.side_effect = NotImplementedError
        args = _server_args()

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache") as release,
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            backup_succeeded = release_req(
                req=req,
                remaing_req_count=1,
                server_args=args,
                req_to_token_pool=Mock(),
                token_to_kv_pool_allocator=Mock(),
                tree_cache=Mock(),
                hisparse_coordinator=None,
                abort_on_unsupported_backup=True,
            )

        self.assertFalse(backup_succeeded)
        self.assertIsNone(req.kv_cache_cpu)
        release.assert_called_once()
        req.reset_for_retract.assert_called_once_with()

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_supported_pool_keeps_resume_path(self, _estimate):
        batch = _batch(cpu_copy_supported=True)
        args = _server_args()

        retracted, ratio, aborted = batch.retract_decode(args)

        self.assertEqual([req.rid for req in retracted], ["1"])
        self.assertEqual(aborted, [])
        self.assertEqual(ratio, 0.5)
        batch.release_req.assert_called_once_with(
            1,
            1,
            args,
            offload_kv=True,
            abort_on_unsupported_backup=True,
        )

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_unsupported_pool_aborts_without_cpu_copy(self, _estimate):
        batch = _batch(cpu_copy_supported=False)
        args = _server_args()

        retracted, ratio, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in aborted], ["1"])
        self.assertEqual(ratio, 0.5)
        self.assertIsInstance(aborted[0].to_finish, FINISH_ABORT)
        self.assertIn("does not support", aborted[0].to_finish.message)
        batch.release_req.assert_called_once_with(
            1,
            1,
            args,
            offload_kv=False,
            abort_on_unsupported_backup=False,
        )

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_last_request_abort_never_attempts_cpu_copy(self, _estimate):
        batch = _batch(cpu_copy_supported=True, req_count=1)
        batch.check_decode_mem = Mock(return_value=False)
        args = _server_args()

        retracted, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in aborted], ["0"])
        batch.release_req.assert_called_once_with(0, 0, args, offload_kv=False)


class TestSchedulerRetractionAbort(CustomTestCase):
    def test_update_running_batch_aborts_only_victim_and_continues(self):
        victim = _req("victim")
        victim.to_finish = FINISH_ABORT("KV cache cannot be restored")
        batch = Mock()
        batch.batch_size.return_value = 2
        batch.is_empty.return_value = False
        batch.check_decode_mem.return_value = False
        batch.retract_decode.return_value = ([], 0.5, [victim])

        scheduler = object.__new__(Scheduler)
        scheduler.forward_ct = 1
        scheduler.enable_hierarchical_cache = False
        scheduler.server_args = _server_args()
        scheduler.token_to_kv_pool_allocator = Mock()
        scheduler.token_to_kv_pool_allocator.available_size.side_effect = [0, 1]
        scheduler.tree_cache = SimpleNamespace(req_to_token_pool=SimpleNamespace())
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.8)
        scheduler.metrics_reporter = SimpleNamespace(
            num_retracted_reqs=0,
            enable_metrics=False,
        )
        scheduler._finalize_retraction_abort = Mock()
        scheduler._add_request_to_queue = Mock()

        result = scheduler.update_running_batch(batch)

        self.assertIs(result, batch)
        scheduler._finalize_retraction_abort.assert_called_once_with(victim)
        scheduler._add_request_to_queue.assert_not_called()
        self.assertEqual(scheduler.metrics_reporter.num_retracted_reqs, 1)
        batch.prepare_for_decode.assert_called_once_with()

    def test_abort_finalization_keeps_scheduler_path_local(self):
        req = _req("abort")
        abort_reason = FINISH_ABORT(
            "KV cache cannot be restored",
            status_code=500,
            err_type="InternalServerError",
        )
        req.to_finish = abort_reason

        scheduler = object.__new__(Scheduler)
        scheduler.decode_offload_manager = Mock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )

        scheduler._finalize_retraction_abort(req)

        self.assertIsNone(req.to_finish)
        self.assertIs(req.finished_reason, abort_reason)
        req.time_stats.trace_ctx.abort.assert_called_once_with(abort_info=abort_reason)
        req.time_stats.set_completion_time.assert_called_once_with()
        scheduler.decode_offload_manager.finalize_release_on_finish.assert_called_once_with(
            req
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        send_output.assert_called_once()
        abort_req, sent_req = send_output.call_args.args
        self.assertIs(sent_req, req)
        self.assertEqual(abort_req.finished_reason, abort_reason.to_json())


if __name__ == "__main__":
    unittest.main()
