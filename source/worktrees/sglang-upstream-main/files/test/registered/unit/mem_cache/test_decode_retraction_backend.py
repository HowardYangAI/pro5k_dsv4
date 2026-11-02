import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    prepare_pd_rebootstrap,
)
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv_cache_builder import _supports_host_pool_retraction
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeRetractionBackend(unittest.TestCase):
    def test_deepseek_v4_supports_host_pool_retraction(self):
        kv_cache = MagicMock(spec=DeepSeekV4TokenToKVPool)

        self.assertTrue(_supports_host_pool_retraction(kv_cache, 1))

    def test_mha_and_hybrid_swa_support_host_pool_retraction(self):
        self.assertTrue(
            _supports_host_pool_retraction(MagicMock(spec=MHATokenToKVPool), None)
        )
        self.assertTrue(_supports_host_pool_retraction(MagicMock(spec=SWAKVPool), 1))

    def test_pure_swa_and_unknown_pool_do_not_support_host_pool_retraction(self):
        self.assertFalse(_supports_host_pool_retraction(MagicMock(spec=SWAKVPool), 0))
        self.assertFalse(_supports_host_pool_retraction(MagicMock(spec=KVCache), None))

    def test_unsupported_cpu_tensor_backup_returns_false(self):
        req = MagicMock()
        req.offload_kv_cache.side_effect = NotImplementedError
        allocator = MagicMock()
        allocator.get_kvcache.return_value = MagicMock(spec=DeepSeekV4TokenToKVPool)

        self.assertFalse(
            retraction_backup(
                req,
                tree_cache=MagicMock(),
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=allocator,
                backend="cpu_tensor",
            )
        )

    def test_backup_failure_falls_back_to_pd_rebootstrap(self):
        kept_req = MagicMock()
        fallback_req = MagicMock()
        fallback_req.supports_pd_rebootstrap.return_value = True

        batch = MagicMock()
        batch.reqs = [kept_req, fallback_req]
        batch._get_decode_retraction_order.return_value = [0, 1]
        batch.check_decode_mem.return_value = True
        batch.release_req.return_value = False

        def filter_batch(*, keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch.side_effect = filter_batch
        with patch.object(
            NewTokenRatioTracker,
            "estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, rebootstrap, ratio, aborted = ScheduleBatch.retract_decode(
                batch,
                SimpleNamespace(),
            )

        self.assertEqual(retracted, [])
        self.assertEqual(rebootstrap, [fallback_req])
        self.assertEqual(ratio, 0.5)
        self.assertEqual(aborted, [])
        batch.release_req.assert_called_once_with(1, 1, unittest.mock.ANY)

    def test_unrecomputable_backup_failure_has_serializable_abort_type(self):
        kept_req = MagicMock()
        aborted_req = MagicMock(rid="aborted")
        aborted_req.supports_pd_rebootstrap.return_value = False

        batch = MagicMock()
        batch.reqs = [kept_req, aborted_req]
        batch._get_decode_retraction_order.return_value = [0, 1]
        batch.check_decode_mem.return_value = True
        batch.release_req.return_value = False
        batch.filter_batch.side_effect = lambda *, keep_indices: setattr(
            batch, "reqs", [batch.reqs[i] for i in keep_indices]
        )

        with patch.object(
            NewTokenRatioTracker,
            "estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            _, rebootstrap, _, aborted = ScheduleBatch.retract_decode(
                batch,
                SimpleNamespace(),
            )

        self.assertEqual(rebootstrap, [])
        self.assertEqual(aborted, [aborted_req])
        self.assertEqual(aborted_req.to_finish.err_type, "InternalServerError")

    def test_prepare_pd_rebootstrap_preserves_boundary_token(self):
        req = SimpleNamespace(
            output_ids=array("q", [10, 11, 12]),
            pd_rebootstrap_forced_output_id=None,
            pd_rebootstrap_in_progress=False,
            time_stats=MagicMock(),
        )

        prepare_pd_rebootstrap(req)

        self.assertEqual(req.output_ids.tolist(), [10, 11])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 12)
        self.assertTrue(req.pd_rebootstrap_in_progress)
        req.time_stats.set_retract_time.assert_called_once_with()

    def test_req_rebootstrap_support_is_token_only_and_real_transfer(self):
        req = SimpleNamespace(
            bootstrap_host="prefill-host",
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            token_type_ids=None,
            lora_id=None,
            session=None,
            session_id=None,
        )
        server_args = SimpleNamespace(disaggregation_transfer_backend="mooncake")

        self.assertTrue(Req.supports_pd_rebootstrap(req, server_args))
        req.input_embeds = [[0.0]]
        self.assertFalse(Req.supports_pd_rebootstrap(req, server_args))

    def test_fake_transfer_rebootstrap_aborts_before_dispatch(self):
        receiver = MagicMock()
        decode_req = SimpleNamespace(
            req=SimpleNamespace(rid="fake-rid", bootstrap_host="prefill-host"),
            kv_receiver=receiver,
        )
        queue = SimpleNamespace(
            transfer_backend=TransferBackend.FAKE,
            kv_manager=MagicMock(),
        )

        DecodePreallocQueue._submit_prefill_recompute(queue, decode_req)

        receiver.abort.assert_called_once_with()
        queue.kv_manager.submit_prefill_recompute.assert_not_called()

    def test_scheduler_enqueues_rebootstrap_with_boundary_replay(self):
        req = SimpleNamespace(
            output_ids=array("q", [10, 11, 12]),
            pd_rebootstrap_forced_output_id=None,
            pd_rebootstrap_in_progress=False,
            time_stats=MagicMock(),
        )
        scheduler = object.__new__(Scheduler)
        scheduler.disagg_decode_prealloc_queue = MagicMock()

        scheduler._enqueue_pd_rebootstrap(req)

        self.assertEqual(req.output_ids.tolist(), [10, 11])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 12)
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_rebootstrap=True
        )

    def test_memory_pressure_routes_backup_failure_to_rebootstrap(self):
        req = SimpleNamespace(origin_input_ids=[1], output_ids=[2])
        batch = MagicMock()
        batch.batch_size.return_value = 1
        batch.is_empty.return_value = False
        batch.check_decode_mem.return_value = False
        batch.retract_decode.return_value = ([], [req], 0.5, [])

        scheduler = object.__new__(Scheduler)
        scheduler.forward_ct = 1
        scheduler.server_args = SimpleNamespace()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.side_effect = [0, 1]
        scheduler.tree_cache = SimpleNamespace(req_to_token_pool=SimpleNamespace())
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.8)
        scheduler.metrics_reporter = SimpleNamespace(
            num_retracted_reqs=0,
            enable_metrics=False,
        )
        scheduler._enqueue_pd_rebootstrap = MagicMock()

        result = scheduler.update_running_batch(batch)

        self.assertIs(result, batch)
        scheduler._enqueue_pd_rebootstrap.assert_called_once_with(req)
        self.assertEqual(scheduler.metrics_reporter.num_retracted_reqs, 1)
        batch.prepare_for_decode.assert_called_once_with()

    @patch("sglang.srt.managers.scheduler._make_abort_req")
    @patch("sglang.srt.managers.scheduler.prepare_abort")
    def test_retraction_abort_is_fully_finalized(self, mock_prepare, mock_make_abort):
        abort_reason = FINISH_ABORT(
            "cannot resume",
            status_code=500,
            err_type="InternalServerError",
        )
        req = SimpleNamespace(
            to_finish=abort_reason,
            finished_reason=None,
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=MagicMock()),
                set_completion_time=MagicMock(),
            ),
        )
        abort_req = object()
        mock_make_abort.return_value = abort_req
        scheduler = object.__new__(Scheduler)
        scheduler.decode_offload_manager = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )

        scheduler._finalize_retraction_abort(req)

        mock_prepare.assert_called_once_with(req, "cannot resume", status_code=500)
        self.assertIs(req.finished_reason, abort_reason)
        self.assertIsNone(req.to_finish)
        req.time_stats.trace_ctx.abort.assert_called_once_with(abort_info=abort_reason)
        req.time_stats.set_completion_time.assert_called_once_with()
        scheduler.decode_offload_manager.finalize_release_on_finish.assert_called_once_with(
            req
        )
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
            abort_req, req
        )


if __name__ == "__main__":
    unittest.main()
