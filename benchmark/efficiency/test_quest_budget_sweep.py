"""CPU checks for GQA addressing in the budget audit (no CUDA/JIT needed)."""

import unittest

import torch

from bench_quest_budget_sweep import explicit_gather, paged_metadata


class QuestAddressingTest(unittest.TestCase):
    def test_gather_and_paged_metadata_preserve_each_head_support(self):
        for kv_heads, heads in ((1, 1), (2, 8), (4, 4)):
            with self.subTest(kv_heads=kv_heads, heads=heads):
                batch, length, dim, budget = 2, 11, 3, 5
                k = torch.arange(batch * kv_heads * length * dim).reshape(batch, kv_heads, length, dim).float()
                v = -k - 1
                ids = (torch.arange(batch * heads * budget).reshape(batch, heads, budget) * 3) % length
                out_k = torch.empty(batch, heads, budget, dim)
                out_v = torch.empty_like(out_k)
                actual_k, actual_v = explicit_gather(k, v, ids, out_k, out_v)
                indptr, pages, last = paged_metadata(ids.int(), kv_heads, length)
                self.assertEqual(pages.dtype, torch.int32)
                self.assertEqual(indptr.tolist(), list(range(0, (batch * heads + 1) * budget, budget)))
                self.assertEqual(last.tolist(), [1] * (batch * heads))
                self.assertEqual(actual_k.data_ptr(), out_k.data_ptr())
                self.assertEqual(actual_v.data_ptr(), out_v.data_ptr())
                for b in range(batch):
                    for h in range(heads):
                        kvh = h // (heads // kv_heads)
                        expected_k, expected_v = k[b, kvh, ids[b, h]], v[b, kvh, ids[b, h]]
                        self.assertTrue(torch.equal(actual_k[b, h], expected_k))
                        self.assertTrue(torch.equal(actual_v[b, h], expected_v))
                        request = b * heads + h
                        selected = pages[indptr[request]:indptr[request + 1]].long()
                        self.assertTrue(torch.equal(k.view(-1, dim)[selected], expected_k))
                        self.assertTrue(torch.equal(v.view(-1, dim)[selected], expected_v))


if __name__ == "__main__":
    unittest.main()
