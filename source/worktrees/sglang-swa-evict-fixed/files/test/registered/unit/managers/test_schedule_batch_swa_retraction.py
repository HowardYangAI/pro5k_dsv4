import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestScheduleBatchSWARetraction(CustomTestCase):
    def test_check_decode_mem_forces_swa_reclamation_before_retraction(self):
        batch = ScheduleBatch(reqs=[])
        batch.forward_mode = Mock()
        batch.forward_mode.is_decode.return_value = True
        batch.tree_cache = Mock()
        batch.tree_cache.supports_swa.return_value = True
        batch.token_to_kv_pool_allocator = Mock()
        batch.token_to_kv_pool_allocator.available_size.side_effect = [3, 8]
        batch.new_tokens_required_next_decode = Mock(return_value=8)
        batch.maybe_evict_swa = Mock()

        with patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache") as evict:
            self.assertTrue(batch.check_decode_mem())

        batch.maybe_evict_swa.assert_called_once_with(force=True)
        self.assertEqual(evict.call_count, 2)

    def test_force_swa_reclamation_ignores_periodic_interval(self):
        req = SimpleNamespace(decode_batch_idx=1, seqlen=32)
        batch = ScheduleBatch(reqs=[req])
        batch.forward_mode = Mock()
        batch.forward_mode.is_decode.return_value = True
        batch.tree_cache = Mock(sliding_window_size=16)
        batch.tree_cache.supports_swa.return_value = True
        batch.forward_iter = 1
        batch._evict_swa = Mock()

        with (
            patch(
                "sglang.srt.managers.schedule_batch.envs."
                "SGLANG_SWA_EVICTION_INTERVAL.get",
                return_value=8,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.envs."
                "SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get",
                return_value=False,
            ),
            patch("sglang.srt.managers.schedule_batch.get_server_args"),
            patch("sglang.srt.managers.schedule_batch.maybe_evict_dsv4_state"),
        ):
            batch.maybe_evict_swa()
            batch._evict_swa.assert_not_called()

            batch.maybe_evict_swa(force=True)

        batch._evict_swa.assert_called_once_with(req, req.seqlen - 1)


if __name__ == "__main__":
    unittest.main()
