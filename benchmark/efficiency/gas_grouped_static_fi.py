"""Experimental reusable FI grouped plan; fixed per-group capacity with overflow checks."""
import math

import flashinfer
import torch


class StaticGroupedAttention:
    def __init__(self, q, k, v, initial_mask, quantum=128):
        if quantum < 2 or quantum % 2:
            raise ValueError('capacity quantum must be a positive even integer')
        batch, heads, dim = q.shape
        length = k.shape[0]
        if heads != 32 or k.shape != (length, 8, dim) or v.shape != k.shape or dim != 128:
            raise ValueError('expected Q[B,32,128] and shared K/V[L,8,128]')
        if initial_mask.shape != (batch, heads, length):
            raise ValueError('expected mask[B,32,L]')
        self.batch, self.length, self.device = batch, length, q.device
        counts = initial_mask.reshape(batch * 8, 4, length).any(1).sum(-1, dtype=torch.int32)
        capacities = ((counts + quantum - 1) // quantum * quantum).clamp(min=quantum)
        self.capacities = capacities.to(q.device)
        cpu_caps = capacities.cpu().tolist()
        ptr = [0]
        for value in cpu_caps:
            ptr.append(ptr[-1] + value)
        self.kvptr = torch.tensor(ptr, dtype=torch.int32, device=q.device)
        self.maskptr = self.kvptr * 4
        self.qptr = torch.arange(batch * 8 + 1, dtype=torch.int32, device=q.device) * 4
        self.last = torch.ones(batch * 8, dtype=torch.int32, device=q.device)
        self.indices = torch.zeros(ptr[-1], dtype=torch.int32, device=q.device)
        self.mask_bits = torch.zeros(ptr[-1] * 4, dtype=torch.bool, device=q.device)
        self.packed = torch.zeros(ptr[-1] // 2, dtype=torch.uint8, device=q.device)
        self.mask_indptr = torch.empty(batch * 8 + 1, dtype=torch.int32, device=q.device)
        work = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            work, 'NHD', use_cuda_graph=True,
            qo_indptr_buf=torch.empty_like(self.qptr), paged_kv_indptr_buf=torch.empty_like(self.kvptr),
            paged_kv_indices_buf=self.indices, paged_kv_last_page_len_buf=torch.empty_like(self.last),
            custom_mask_buf=self.packed, mask_indptr_buf=self.mask_indptr)
        self.wrapper.plan(self.qptr, self.kvptr, self.indices, self.last, 1, 1, dim, 1,
                          custom_mask=self.mask_bits, causal=False, pos_encoding_mode='NONE',
                          q_data_type=q.dtype, kv_data_type=k.dtype, sm_scale=1 / math.sqrt(dim))
        self.pages = (k.view(-1, 1, 1, dim), v.view(-1, 1, 1, dim))

    def update(self, mask):
        """Rebuild indices and packed masks; fails before writing on capacity overflow."""
        if mask.shape != (self.batch, 32, self.length) or mask.device != self.device or mask.dtype != torch.bool:
            raise ValueError('mask must be CUDA bool [B,32,L] on the planned device')
        grouped = mask.reshape(self.batch * 8, 4, self.length)
        union = grouped.any(1)
        counts = union.sum(-1, dtype=torch.int32)
        if bool((counts > self.capacities).any().item()):
            raise ValueError('group support exceeds fixed plan capacity; replan required')
        rows, tokens = union.nonzero(as_tuple=True)
        actual_ptr = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
        rank = torch.arange(rows.numel(), device=self.device) - actual_ptr[rows]
        self.indices.zero_()
        self.mask_bits.zero_()
        self.indices[(self.kvptr[rows] + rank).long()] = (tokens * 8 + rows % 8).to(torch.int32)
        head = torch.arange(4, device=self.device)
        mask_dest = self.maskptr[rows, None] + head[None, :] * self.capacities[rows, None] + rank[:, None]
        self.mask_bits[mask_dest.long()] = grouped[rows, :, tokens]
        packed, _ = flashinfer.quantization.segment_packbits(self.mask_bits, self.maskptr, bitorder='little')
        self.packed.copy_(packed)

    def run(self, q):
        return self.wrapper.run(q.view(-1, 1, 128), self.pages).view_as(q)

    def update_triton(self, mask):
        from gas_grouped_compact_triton import update
        update(self, mask)
