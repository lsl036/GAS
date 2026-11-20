import unittest

import torch

from bench_lessismore_e6_matched import less_indices


class LessIsMoreMatchedTest(unittest.TestCase):
    def test_global_indices_are_unique_and_shared(self):
        group, head = less_indices(128, 3, 2, 8, 17, 0.25, 7, torch.device("cpu"))
        self.assertEqual(group.shape, (3, 2, 17))
        self.assertEqual(head.shape, (3, 8, 17))
        for batch in range(3):
            values = group[batch, 0].tolist()
            self.assertEqual(len(set(values)), 17)
            self.assertEqual(values[-5:], list(range(123, 128)))
            for group_id in range(2):
                self.assertTrue(torch.equal(group[batch, group_id], group[batch, 0]))
            for head_id in range(8):
                self.assertTrue(torch.equal(head[batch, head_id], group[batch, 0]))


if __name__ == "__main__":
    unittest.main()
