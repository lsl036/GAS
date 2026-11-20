import unittest
from pathlib import Path

import torch

from bench_twilight_upstream_six import author_budgets, synthetic_masks, top_p_mask


class SyntheticSupportTest(unittest.TestCase):
    def test_exact_top_p(self):
        scores = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        self.assertEqual(top_p_mask(scores, 0.8).tolist(), [[True, True, False]])

    def test_support_relationships(self):
        upstream = Path(__file__).resolve().parents[3] / 'Twilight_Upstream_20261113'
        budgets = author_budgets(upstream / 'benchmark/efficiency/bench_breakdown.py', 256)
        self.assertEqual(len(budgets), 32)
        for length in (16384, 32768):
            masks, less_k, shared = synthetic_masks(length, int(length * .08), budgets, 42 + length)
            self.assertTrue(torch.all(masks['twilight'] <= masks['quest']))
            union = masks['twilight'].view(8, 4, length).any(1).repeat_interleave(4, 0)
            self.assertTrue(torch.all(masks['gas'] <= union))
            self.assertTrue(torch.all(shared.repeat_interleave(4, 0) <= masks['gas']))
            self.assertLessEqual(int((masks['gas'] & ~shared.repeat_interleave(4, 0)).sum(-1).max()), 16)
            self.assertTrue(torch.all(masks['lessismore_style'].sum(-1) == less_k))


if __name__ == '__main__':
    unittest.main()
