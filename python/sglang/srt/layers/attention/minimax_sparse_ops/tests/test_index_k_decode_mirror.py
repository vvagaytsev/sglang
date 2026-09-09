"""Decode fp8 index-K mirror (SGLANG_MINIMAX_FP8_INDEX_K).

A missed store leaves decode scoring a stale cache while prefill stays correct.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

DEVICE = "cuda"
SIZE = 256
PAGE_SIZE = 1
HEAD_NUM = 2
HEAD_DIM = 64
IDX_HEAD_DIM = 128
DENSE_LAYER_IDS = [0]
SPARSE_LAYER_IDS = [1, 2]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU KV pool"
)


def fp8_dtype():
    return torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def build_pool(index_decode_dtype):
    return MiniMaxSparseKVPool(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        idx_head_dim=IDX_HEAD_DIM,
        dense_layer_ids=DENSE_LAYER_IDS,
        sparse_layer_ids=SPARSE_LAYER_IDS,
        disable_value_sparse_layer_ids=SPARSE_LAYER_IDS,  # K-only sparse, as on M3
        device=DEVICE,
        index_dtype=torch.bfloat16,
        index_decode_dtype=index_decode_dtype,
        start_layer=0,
        end_layer=len(DENSE_LAYER_IDS) + len(SPARSE_LAYER_IDS),
    )


def mirror_bytes(pool, layer_id, loc):
    return pool.get_index_k_decode_buffer(layer_id).view(torch.uint8)[loc]


def expected_bytes(idx_k):
    return idx_k.to(fp8_dtype()).view(torch.uint8)


def random_idx_k(num_tokens):
    return torch.randn(num_tokens, 1, IDX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)


def test_no_mirror_when_not_requested():
    pool = build_pool(None)
    layer_id = SPARSE_LAYER_IDS[0]

    assert pool.index_k_decode_pool is None
    assert (
        pool.get_index_k_decode_buffer(layer_id).data_ptr()
        == pool.get_index_k_buffer(layer_id).data_ptr()
    )


def test_set_index_k_buffer_updates_mirror():
    pool = build_pool(fp8_dtype())
    layer_id = SPARSE_LAYER_IDS[1]

    loc = torch.tensor([3, 17, 40], device=DEVICE, dtype=torch.int64)
    idx_k = random_idx_k(loc.shape[0])
    pool.set_index_k_buffer(SimpleNamespace(layer_id=layer_id), loc, idx_k)

    prefill_buf = pool.get_index_k_buffer(layer_id)
    assert prefill_buf.dtype == torch.bfloat16
    assert pool.get_index_k_decode_buffer(layer_id).dtype == fp8_dtype()

    assert torch.equal(prefill_buf[loc], idx_k)
    assert torch.equal(mirror_bytes(pool, layer_id, loc), expected_bytes(idx_k))

    untouched = torch.tensor([4], device=DEVICE, dtype=torch.int64)
    assert not mirror_bytes(pool, layer_id, untouched).any()


def test_mirror_is_per_layer():
    pool = build_pool(fp8_dtype())
    first, second = SPARSE_LAYER_IDS
    loc = torch.tensor([5], device=DEVICE, dtype=torch.int64)
    idx_k = torch.ones(1, 1, IDX_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    pool.set_index_k_buffer(SimpleNamespace(layer_id=first), loc, idx_k)

    assert mirror_bytes(pool, first, loc).any()
    assert not mirror_bytes(pool, second, loc).any()


def test_fused_store_updates_mirror():
    pool = build_pool(fp8_dtype())
    layer_id = SPARSE_LAYER_IDS[0]

    loc = torch.tensor([7, 9], device=DEVICE, dtype=torch.int64)
    k = torch.randn(
        loc.shape[0], HEAD_NUM, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    idx_k = random_idx_k(loc.shape[0])
    pool.set_fused_kv_index_buffer(
        SimpleNamespace(layer_id=layer_id), loc, k, torch.randn_like(k), idx_k, None
    )

    assert torch.equal(mirror_bytes(pool, layer_id, loc), expected_bytes(idx_k))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
