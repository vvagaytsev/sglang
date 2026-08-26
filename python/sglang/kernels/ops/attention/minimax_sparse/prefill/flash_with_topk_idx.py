# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import _bitonic_merge, get_cu_seqblocks, robust_allocator

# Score-only path (DISABLE_INDEX_VALUE=True), i.e. every sparse layer of MiniMax-M3.
# BLOCK_SIZE_Q=64 halves the workgroup count vs BLOCK_SIZE_Q=32, keeping a
# chunked-prefill-8192 step at ~128 workgroups instead of ~256. That matters because
# BLOCK_SIZE_K=256 costs 132 KB of the 160 KB LDS/CU, so only one workgroup is
# resident per CU: crossing 256 active workgroups serializes into a second wave and
# ~1.9x the latency, with no partial-occupancy cushion.
_SCORE_ONLY_CONFIG = triton.Config(
    {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2
)

# DISABLE_INDEX_VALUE=False also stages a V tile, and BLOCK_SIZE_K=256 then needs
# 305072 B > the 163840 B LDS limit. Every BLOCK_SIZE_K=256 config is unusable here,
# so this path needs its own BLOCK_SIZE_K<=128 config.
_WITH_INDEX_VALUE_CONFIG = triton.Config(
    {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
)


def _select_config(configs, named_args, **kwargs):
    """Pick the config by code path instead of letting autotune benchmark for it.

    The autotune key holds only dtype/layout constexprs, so it cannot see the token
    layout that actually decides the winner. Autotune therefore measures whichever
    shape a process happens to run first and reuses that choice for every later
    shape -- and on a short warmup shape the two configs tie, so it can settle on
    the slower one for the whole process. Forcing the mapping keeps selection
    deterministic across restarts.
    """
    disable_index_value = {**named_args, **kwargs}["DISABLE_INDEX_VALUE"]
    return [_SCORE_ONLY_CONFIG if disable_index_value else _WITH_INDEX_VALUE_CONFIG]


@triton.jit
def _topk_accumulate(
    topk_score,  # [BLOCK_SIZE_Q, BLOCK_SIZE_T] running scores, unsorted
    topk_idx,  # [BLOCK_SIZE_Q, BLOCK_SIZE_T] running block ids, -1 = empty slot
    score,  # [BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK] block scores of this K tile
    valid_blocks,  # [BLOCK_SIZE_Q] causal block count per query row
    first_block,  # K-block index of score[:, 0]
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    BLOCKS_PER_K_BLOCK: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Fold one K tile's block scores into the running top-k (FUSE_TOPK epilogue).

    Selection is an argmin-insertion rather than the bitonic merge
    ``_topk_index_kernel`` uses: a K tile carries only BLOCKS_PER_K_BLOCK (1-2)
    new candidates, and a merge network costs the same for 2 candidates as for
    32, so merging per tile would be ~12x the standalone's per-candidate cost.
    Insertion is ~4 passes over a BLOCK_SIZE_Q x BLOCK_SIZE_T tile per candidate.
    The array is left unsorted here and sorted once by ``_topk_sort_and_store``.
    """
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = first_block + off_bpk
    # NaN breaks the epilogue's bitonic sort; -inf is the sentinel that can never
    # win a slot below. Mirrors the scrub in _topk_index_kernel.
    score = tl.where(score != score, float("-inf"), score)
    # Force the init and local blocks into the selection, matching
    # _topk_index_kernel under the shipped MASK_INIT=False / MASK_LOCAL=False.
    # causal_mask keeps a block outside the causal window from being boosted.
    causal_mask = blk[None, :] < valid_blocks[:, None]
    score = tl.where(causal_mask & (blk[None, :] < init_blocks), 1e30, score)
    local_lo = tl.maximum(valid_blocks - local_blocks, 0)
    score = tl.where(causal_mask & (blk[None, :] >= local_lo[:, None]), 1e29, score)
    for p in tl.static_range(BLOCKS_PER_K_BLOCK):
        cand = tl.sum(tl.where(off_bpk[None, :] == p, score, 0.0), axis=1)
        # Any slot holding the running minimum is a valid eviction target, so
        # argmin's tie-break does not affect the selected set. A -inf candidate
        # never beats the -1e30 init, so masked blocks can never be inserted.
        # The second term compares against topk_score rather than its min: at the
        # argmin slot the two are equal, so this is the same predicate for one
        # cross-lane reduction instead of two.
        hit = (off_t[None, :] == tl.argmin(topk_score, axis=1)[:, None]) & (
            cand[:, None] > topk_score
        )
        topk_score = tl.where(hit, cand[:, None], topk_score)
        topk_idx = tl.where(hit, first_block + p, topk_idx)
    return topk_score, topk_idx


@triton.jit
def _topk_sort_and_store(
    ti_ptr,  # topk_idx base, head offset already folded in
    topk_score,
    topk_idx,
    row_idx,  # [BLOCK_SIZE_Q] destination row per query
    row_valid,  # [BLOCK_SIZE_Q] rows inside q_len
    valid_blocks,  # [BLOCK_SIZE_Q]
    topk,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Sort the running top-k by score and store it right-padded with -1.

    ``topk_sparse.py`` walks ``range(sum(topk_idx >= 0))`` and multiplies each
    entry by BLOCK_SIZE_K, so an interleaved -1 becomes a negative K position
    that its ``pos < seq_len`` mask does not reject. Front-packing is therefore
    a hard correctness requirement, not a convention: never-filled slots hold
    -1e30 and sort to the back, which produces it.
    """
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_T)
    # _bitonic_merge is row-independent for 2-D input (n_outer = numel >> n_dims),
    # so the whole q-tile sorts in one call per stage.
    for j in tl.static_range(1, n_dims):
        topk_score, topk_idx = _bitonic_merge(
            topk_score, topk_idx.to(tl.int32), j, 2, n_dims
        )
    topk_score, topk_idx = _bitonic_merge(
        topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
    )
    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_ptrs = ti_ptr + row_idx[:, None] * stride_ti_n + off_t[None, :] * stride_ti_t
    store_mask = row_valid[:, None] & (
        off_t[None, :] < tl.minimum(topk, valid_blocks)[:, None]
    )
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        # _select_config forces one of these two per code path, so nothing here is
        # benchmarked against anything else.
        _SCORE_ONLY_CONFIG,
        _WITH_INDEX_VALUE_CONFIG,
    ],
    key=[
        "qk_head_dim",
        "v_head_dim",
        "block_size",
        "use_gumbel_topk",
        "SCORE_TYPE",
        "DISABLE_INDEX_VALUE",
        "FUSE_TOPK",
        "BLOCK_SIZE_T",
    ],
    prune_configs_by={"early_config_prune": _select_config},
)
@triton.jit
def _flash_attn_fwd_with_block_score_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    o_ptr,  # O: n x h x d
    score_ptr,  # Score: h x n x max_seqblock
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    block_size: tl.constexpr,
    # sm_scale
    sm_scale,
    # gumbel topk
    use_gumbel_topk: tl.constexpr,
    gumbel_seed,
    # stride
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    DISABLE_INDEX_VALUE: tl.constexpr,
    # Fused block top-k (SILOTIGER-790). Everything below is defaulted so callers
    # of the unfused path -- including the pinned launches in tmp/ that bypass
    # @heuristics -- keep working with the original argument list.
    topk_idx_ptr=None,  # topk_idx: h x n x topk
    cu_seqblocks_q=None,
    topk=0,  # not constexpr, matching _topk_index_kernel
    stride_ti_h=0,
    stride_ti_n=0,
    stride_ti_t=0,
    init_blocks: tl.constexpr = 1,
    local_blocks: tl.constexpr = 2,
    FUSE_TOPK: tl.constexpr = False,
    BLOCK_SIZE_T: tl.constexpr = 1,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size
    # get batch id and head id
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    block_num = (seq_len + block_size - 1) // block_size
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    if not FUSE_TOPK:
        s_ptrs = tl.make_block_ptr(
            base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
            shape=(q_len, block_num),
            strides=(stride_s_q, stride_s_k),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
            order=(1, 0),
        )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    if HAS_SINK:
        m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        lse_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        qsink = tl.sum(q * sink[None, :], axis=1) * sm_scale_log2e  # (BLOCK_SIZE_Q,)
        m_i += qsink
        lse_i += qsink
    else:
        m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    if FUSE_TOPK:
        # Per-row causal block count. The scalar form in _topk_index_kernel is
        # (prefix_len + pid_q * sample_interval + block_size) // block_size; here
        # off_q already carries prefix_len + the row's position in the q-tile,
        # which is the same thing at sample_interval == 1.
        valid_blocks = (off_q + block_size) // block_size
        topk_score = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_T), -1e30, dtype=tl.float32)
        topk_idx = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_T), -1, dtype=tl.int32)
    # attention
    diag_start = (prefix_len + pid_q * BLOCK_SIZE_Q) // BLOCK_SIZE_K * BLOCK_SIZE_K
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        # paged load K via req_to_token: pos -> slot -> k_cache
        pos = i + off_k
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots  # safety against negative
        # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_kd[:, None] * stride_k_d,
            mask=kd_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        # compute qk
        qk = tl.dot(q, k) * sm_scale_log2e
        if i >= diag_start:
            qk = tl.where(off_q[:, None] >= (i + off_k)[None, :], qk, float("-inf"))
        # K boundary mask: positions beyond seq_len contribute -inf
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        # save score
        score = tl.reshape(
            qk, (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False
        )
        sub_max = tl.max(score, axis=2)
        if SCORE_TYPE == "max":
            score = sub_max
        else:  # "lse"
            # fully-masked sub-blocks produce NaN via -inf - (-inf); clamp
            # back to -inf so downstream bitonic sort sees a clean sentinel.
            score = sub_max + tl.log2(
                tl.sum(tl.exp2(score - sub_max[:, :, None]), axis=2)
            )
            score = tl.where(score != score, float("-inf"), score)
        if use_gumbel_topk:
            # generate non-conflicting offset for random generation
            # noise_offset shape: (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK)
            # random seed include head id, batch id and gumbel seed
            # (Head low 7 bits | Batch middle 12 bits | Other high bits)
            local_seed = (pid_h | (pid_b << 7) | (gumbel_seed << 19)).to(tl.int32)
            # noise offset include q index and k block index
            # [31-13: Q (19bits)] | [12-0: K_Block (13bits)]
            noise_offset = (off_q << 13)[:, None] | (off_bpk + i // block_size)[None, :]
            # gumbel noise (scaled to log2 scale to match sm_scale_log2e)
            noise = tl.rand(local_seed, offset=noise_offset)
            noise = tl.clamp(noise, min=1e-9, max=1 - 1e-9)  # avoid log(0)
            noise = -tl.log(-tl.log(noise)) * 1.4426950409
            score += noise
        if FUSE_TOPK:
            topk_score, topk_idx = _topk_accumulate(
                topk_score,
                topk_idx,
                score,
                valid_blocks,
                i // block_size,
                init_blocks,
                local_blocks,
                BLOCKS_PER_K_BLOCK,
                BLOCK_SIZE_T,
            )
        else:
            tl.store(
                s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1)
            )
        if not DISABLE_INDEX_VALUE:
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # paged load V
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_v_s
                + pid_kh * stride_v_h
                + off_vd[None, :] * stride_v_d,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        if not FUSE_TOPK:
            s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
    if FUSE_TOPK:
        # The K loop covered the full causal range of this q-tile, so the running
        # top-k is final here -- no cross-CTA reduction needed.
        off_r = tl.arange(0, BLOCK_SIZE_Q)
        block_start = tl.load(cu_seqblocks_q + pid_b)
        _topk_sort_and_store(
            topk_idx_ptr + pid_h * stride_ti_h,
            topk_score,
            topk_idx,
            block_start + pid_q * BLOCK_SIZE_Q + off_r,
            off_r + pid_q * BLOCK_SIZE_Q < q_len,
            valid_blocks,
            topk,
            stride_ti_n,
            stride_ti_t,
            BLOCK_SIZE_T,
        )
    if not DISABLE_INDEX_VALUE:
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + seq_start * stride_o_n + pid_h * stride_o_h,
            shape=(q_len, v_head_dim),
            strides=(stride_o_n, stride_o_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        # Large configs for H200/B200 to support larger topk
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=[
        "BLOCK_SIZE_T"
    ],  # use BLOCK_SIZE_T instead of topk to reduce autotune frequency
)
@triton.jit
def _topk_index_kernel(
    s_ptr,  # Score: h x n x max_seqblock
    ti_ptr,  # topk_idx: h x n x topk
    # size
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    # seqlens
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    # shape
    topk,  # not constexpr to avoid recompilation when topk changes
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    # stride
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(
        BLOCK_SIZE_K > BLOCK_SIZE_T
    )  # use BLOCK_SIZE_T instead of topk (stricter but safe)
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    # offsets
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # init qkv pointer
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    # init statistics
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    # compute topk
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        # masks
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        # load score
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        # handle NaN: NaN inputs cause bitonic sort to fail, resulting in invalid indices (-2)
        # appearing in the topk list. We replace NaN with -inf to maintain sort order.
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # fill init and local part, make sure init part is always in topk
        # and at the first position. Note: must use causal_mask to protect
        # init_mask to avoid selecting blocks outside causal window
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        # bitonic merge
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # get topk, shape: [BLOCK_SIZE_T,]
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    # save topk
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    topk_mask = tl.arange(0, BLOCK_SIZE_T) < min(topk, valid_blocks)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=topk_mask)


@torch.no_grad()
def flash_prefill_with_topk_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,  # paged
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
    use_tma: bool = False,
    score_type: str = "max",
    disable_index_value: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    fuse_topk: Optional[bool] = None,
):
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    triton.set_allocator(robust_allocator)
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k_cache.dtype == q.dtype
    assert cu_seqlens.dtype == torch.int32
    # shape
    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    if disable_index_value:
        # placeholder for BLOCK_SIZE_VD; V is never loaded
        v_head_dim = qk_head_dim
    else:
        assert v_cache is not None and v_cache.dtype == q.dtype
        assert v_cache.shape[1] == k_cache.shape[1]
        v_head_dim = v_cache.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    if sink is not None:
        assert sink.shape[0] == num_heads and sink.shape[1] == qk_head_dim
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    if fuse_topk is None:
        # Off by default: the fused epilogue is correct but slower on gfx950. It
        # deletes the 21 MiB score round trip, its -inf memset and the standalone
        # launch, buying a ~138 us budget at chunk 8192 / context 82050. Best fused
        # config over a 108-point BQ x BK x warps x stages sweep is 0.78x the
        # two-kernel path, see tmp/bench_prefill_fused_topk.
        # The epilogue itself can fit that budget -- at BQ 32 / BK 256 / w4 it costs
        # +125 us -- but only at a config where the *base* score kernel is 933 us
        # against its own 689 us optimum (BQ 32 / BK 256 / w8), where the epilogue
        # instead costs +391 us. The cost is a per-candidate cross-warp argmin over
        # BLOCK_SIZE_T plus an MFMA-slice -> blocked layout conversion, once for
        # each of the ~640 K-blocks a q-tile scans; it scales with num_warps
        # (+172 us at w2 vs +391 us at w8, same tile), which is exactly why it
        # cannot be paid for -- the score kernel needs w8 to reach 689 us.
        # Callers must opt in explicitly.
        # block_size_q == 1 remains the correctness gate whenever they do: only
        # then is all_seqblock_q == total_q and cu_seqblocks_q == cu_seqlens, which
        # is what makes the fused row index block_start + pid_q * BLOCK_SIZE_Q + r
        # valid.
        fuse_topk = False
    assert not fuse_topk or block_size_q == 1, "fused top-k requires block_size_q == 1"
    if disable_index_value:
        o = None
    else:
        o = torch.empty(total_q, num_heads, v_head_dim, dtype=q.dtype, device=q.device)
    topk_idx = torch.full(
        (num_heads, all_seqblock_q, topk),
        fill_value=-1,
        device=q.device,
        dtype=torch.int32,
    )
    if fuse_topk:
        # SILOTIGER-790: the score kernel selects the top-k in its epilogue, so the
        # [num_heads, total_q, max_seqblock_k] staging tensor and its ~2x HBM round
        # trip disappear along with the standalone launch.
        score = None
    else:
        score = torch.full(
            (num_heads, total_q, max_seqblock_k),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

    # launch kernel
    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        False,
        1,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        o.stride(0) if o is not None else 0,
        o.stride(1) if o is not None else 0,
        o.stride(2) if o is not None else 0,
        score.stride(0) if score is not None else 0,
        score.stride(1) if score is not None else 0,
        score.stride(2) if score is not None else 0,
        req_to_token.stride(0),
        topk_idx_ptr=topk_idx,
        cu_seqblocks_q=cu_seqblocks_q,
        topk=topk,
        stride_ti_h=topk_idx.stride(0),
        stride_ti_n=topk_idx.stride(1),
        stride_ti_t=topk_idx.stride(2),
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
        FUSE_TOPK=fuse_topk,
        # Passed rather than derived by @triton.heuristics: a heuristic reading
        # args["topk"] only sees positional args plus explicit kwargs, so it
        # KeyErrors on any caller that invokes the kernel with the original
        # positional argument list.
        BLOCK_SIZE_T=triton.next_power_of_2(max(topk, 1)),
    )
    if fuse_topk:
        return o, topk_idx

    # topk extraction kernel
    grid = (max_seqblock_q, batch_size, num_heads)
    _topk_index_kernel[grid](
        score,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return o, topk_idx
