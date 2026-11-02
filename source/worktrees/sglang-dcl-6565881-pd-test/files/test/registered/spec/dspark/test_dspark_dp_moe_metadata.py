import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsparkDpMoeMetadata(CustomTestCase):
    def test_real_token_count_is_set_for_hand_built_draft_batch(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=3,
        )
        proposer.draft_model_runner = SimpleNamespace(device="cpu")

        forward_batch = SimpleNamespace(input_ids=torch.arange(12))
        schedule_batch = SimpleNamespace(
            global_num_tokens=[3, 1],
            global_num_tokens_for_logprob=[2, 1],
            can_run_dp_cuda_graph=True,
        )

        proposer._fill_dp_moe_sync_metadata(forward_batch, schedule_batch)

        self.assertEqual(forward_batch.num_token_non_padded.item(), 12)
        self.assertEqual(forward_batch.num_token_non_padded.dtype, torch.int32)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 12)
        self.assertEqual(forward_batch.global_num_tokens_cpu, [12, 4])
        self.assertEqual(forward_batch.global_num_tokens_for_logprob_cpu, [6, 3])
        self.assertTrue(forward_batch.can_run_dp_cuda_graph)


if __name__ == "__main__":
    unittest.main()
