# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import _bitonic_merge, get_cu_seqblocks, robust_allocator

# Score-only path (DISABLE_INDEX_VALUE=True), i.e. every sparse layer of MiniMax-M3.
#
# Occupancy decides this one, and the limit is VGPRs, not LDS: num_warps=4 puts one
# wave per SIMD so a CU holds 2 workgroups, where num_warps=8 fits only 1 and a
# launch crossing 256 workgroups then serializes into a second wave for ~1.9x the
# latency -- which a chunked-prefill-8192 step sits right on. Measured on gfx950
# (MI355X, 256 CU) over the production shape mix vs the previous 64x256 w8 s2;
# num_stages=3 over 2 is worth ~1.2x. The CDNA launch knobs (waves_per_eu,
# matrix_instr_nonkdim) moved nothing outside noise, so they are left unset.
# BLOCK_SIZE_Q is a heuristic rather than a config field; see _score_block_size_q.
_SCORE_ONLY_CONFIG = triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=3)

# DISABLE_INDEX_VALUE=False also stages a V tile, and BLOCK_SIZE_K=256 then needs
# 305072 B > the 163840 B LDS limit. Every BLOCK_SIZE_K=256 config is unusable here,
# so this path needs its own BLOCK_SIZE_K<=128 config.
_WITH_INDEX_VALUE_CONFIG = triton.Config(
    {"BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
)


# The two BLOCK_SIZE_Q the score path chooses between, and the workgroups of each
# that gfx950's 256 CUs hold at once: at BLOCK_SIZE_K=256 / num_warps=4 the 32-row
# tile fits two per CU and the 128-row tile one.
_SCORE_SMALL_TILE = 32
_SCORE_LARGE_TILE = 128
_SCORE_SMALL_TILE_WAVE = 512
_SCORE_LARGE_TILE_WAVE = 256

# What one wave of 128-row tiles costs in units of a full 32-row launch. Measured
# per-wave it is 1113 vs 785 ns (1.42), but that overstates the large tile: its
# waves are a ceiling, so a partly-filled last wave is charged whole. Fitting the
# crossover puts the effective ratio at 1.25, and the optimum is flat -- anything
# in 1.1..1.35 gets at most four more of the 114 shapes wrong.
_SCORE_LARGE_TILE_WAVE_COST = 1.25


def _score_block_size_q(args):
    """Pick BLOCK_SIZE_Q from the launch shape, which no autotune config can see.

    The autotune key holds only dtypes and layout constexprs, so a Config cannot
    dispatch on the token and head counts that decide this. Both tiles are
    occupancy-bound but retire workgroups differently: at 1 wg/CU the 128-row tile
    is a staircase in workgroup count (1128 -> 2222 -> 3336 ns across 256 / 512 /
    768), while at 2 wg/CU the 32-row tile rises smoothly (785 -> 1234 -> 1533 ns)
    because its tail pipelines into the wave ahead. Comparing a step function to a
    line makes the crossover non-monotone -- at num_heads=16 the large tile wins at
    total_q 1536, loses at 2048 for every batch but 1, wins again at 3072 -- so
    both costs are estimated rather than thresholded on either count.

    Measured over 114 shapes (num_heads 16/4/1 x total_q 512..16384 x batch 1..16,
    plus a held-out ragged/off-power-of-two set). Picks the slower tile on 5, worst
    1.53x, landing within 1.007x of always choosing the measured winner; a
    num_heads rule gets 28 wrong (1.033x), a fixed 128-row tile 38 (1.051x), a
    fixed 32-row tile 74 (1.299x). Both halves agree on the constant.
    """
    if not args["DISABLE_INDEX_VALUE"]:
        return 64
    total_q = args["q_ptr"].shape[0]
    num_heads, batch_size = args["num_heads"], args["batch_size"]

    def workgroups(tile):
        # The grid gives each sequence its own tiles, so a batch costs batch-1
        # blocks over the flat token count; those are resident too.
        return (triton.cdiv(total_q, tile) + batch_size - 1) * num_heads

    small_cost = workgroups(_SCORE_SMALL_TILE) / _SCORE_SMALL_TILE_WAVE
    large_cost = triton.cdiv(workgroups(_SCORE_LARGE_TILE), _SCORE_LARGE_TILE_WAVE)
    if small_cost > _SCORE_LARGE_TILE_WAVE_COST * large_cost:
        return _SCORE_LARGE_TILE
    return _SCORE_SMALL_TILE


# Top-k index kernel. The grid is one workgroup per q-row per head -- ~131k of them
# for a chunked-prefill-8192 step at 16 heads -- so this is bound by how many
# workgroups a CU keeps resident, not by per-workgroup width, and num_warps=1 (a
# single 64-thread wave) maximizes that. Measured on gfx950 (MI355X, 256 CU) at
# topk=16 over the production shape mix: 64w1s2 = 5686.7 us vs 10683.2 us for the
# 256w4s2 the old autotune space selected (1.88x), winning every shape; the
# narrowest legal tile at num_warps=1 also won at every topk from 8 to 512.
#
# BLOCK_SIZE_K must stay > BLOCK_SIZE_T (static_assert below), so 64 is only legal
# up to topk=32 and _select_topk_config walks up from there. Width is not an
# accuracy trade: the bitonic merge carries BLOCK_SIZE_K/2 survivors across each
# chunk boundary, so every legal tile selects the same block set (measured 0.00%
# drift).
_TOPK_INDEX_CONFIGS = [
    triton.Config({"BLOCK_SIZE_K": block_size_k}, num_warps=1, num_stages=2)
    for block_size_k in (64, 128, 256, 512, 1024, 2048)
]


def _select_topk_config(configs, named_args, **kwargs):
    """Pin the narrowest single-wave tile this topk allows, instead of measuring.

    Forced rather than autotuned for the same reason as _select_config: the key is
    BLOCK_SIZE_T alone, so one config is baked per topk bucket for the whole
    process, chosen on whichever shape ran first. The ranking does not depend on
    the shape (see _TOPK_INDEX_CONFIGS), so there is nothing for a benchmark to
    discover and pinning removes the warmup-order dependence.
    """
    block_size_t = {**named_args, **kwargs}["BLOCK_SIZE_T"]
    return [
        min(
            (c for c in configs if c.kwargs["BLOCK_SIZE_K"] > block_size_t),
            key=lambda c: c.kwargs["BLOCK_SIZE_K"],
        )
    ]


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


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
        # A heuristic, not a config field: it depends on runtime token/batch/head
        # counts, invisible to the autotune key. @heuristics runs before the grid
        # lambda, so the launch below sees the chosen value.
        "BLOCK_SIZE_Q": _score_block_size_q,
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
    batch_size,
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
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size
    # get batch id and head id
    pid_q_global, pid_h = tl.program_id(0), tl.program_id(1)
    pid_kh = pid_h // gqa_group_size

    # Map the flat q-tile id to (batch, tile-within-batch). The grid is now the sum
    # of cdiv(q_len_b, BLOCK_SIZE_Q) rather than max q-tiles per sequence, so a
    # ragged batch no longer launches max_seqlen_q worth of workgroups for every
    # short sequence just to have them return immediately. Branchless because
    # Triton has no `break`, and the trip count is the batch size.
    pid_b = 0
    tile_start = 0  # q-tiles owned by the batches before pid_b
    tiles_seen = 0
    seq_end_prev = tl.load(cu_seqlens)
    for b in tl.range(0, batch_size):
        seq_end = tl.load(cu_seqlens + b + 1)
        n_tiles = tl.cdiv(seq_end - seq_end_prev, BLOCK_SIZE_Q)
        seq_end_prev = seq_end
        # This batch lies entirely before pid_q_global.
        passed = pid_q_global >= tiles_seen + n_tiles
        pid_b += tl.where(passed, 1, 0)
        tile_start += tl.where(passed, n_tiles, 0)
        tiles_seen += n_tiles
    # The grid is rounded up (see flash_prefill_with_topk_index), so the trailing
    # workgroups own no q-tile at all.
    if pid_b >= batch_size:
        return
    pid_q = pid_q_global - tile_start

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
        tl.store(s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
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
        s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
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
    configs=_TOPK_INDEX_CONFIGS,
    key=[
        "BLOCK_SIZE_T"
    ],  # use BLOCK_SIZE_T instead of topk to reduce autotune frequency
    prune_configs_by={"early_config_prune": _select_topk_config},
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
    batch_size,
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
    pid_q_global, pid_h = tl.program_id(0), tl.program_id(1)

    # Map the flat q-block id to (batch, block-within-batch) as the score kernel
    # does, except cu_seqblocks_q is already that prefix sum so the scan need only
    # count batches ending at or before this id. Dim 0 was (max q-blocks) over a
    # batch dimension, sizing every sequence by the longest: a [8060, 1 x 14] batch
    # launched 8060 blocks fifteen times, fourteen of which returned immediately.
    pid_b = 0
    for b in tl.range(0, batch_size):
        pid_b += tl.where(pid_q_global >= tl.load(cu_seqblocks_q + b + 1), 1, 0)
    # all_seqblock_q is exact, so this only fires if a caller rounds the grid up.
    if pid_b >= batch_size:
        return
    block_start = tl.load(cu_seqblocks_q + pid_b)
    pid_q = pid_q_global - block_start
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
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
    # block_start + pid_q is pid_q_global by construction: the flat id already
    # indexes topk_idx's [all_seqblock_q] row dimension.
    ti_ptrs = (
        ti_ptr + pid_q_global * stride_ti_n + pid_h * stride_ti_h + off_t * stride_ti_t
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
    # max_seqblock_q is accepted for call compatibility but no longer read: it
    # sized the old dense top-k grid, which the flat q-block grid replaced.
    del max_seqblock_q
    if cu_seqblocks_q is None or all_seqblock_q is None:
        cu_seqblocks_q, _, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
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
    score = torch.full(
        (num_heads, total_q, max_seqblock_k),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # launch kernel
    def grid(META):
        # The kernel walks a flat q-tile id, so the grid is sum_b cdiv(q_len_b,
        # BLOCK_SIZE_Q) -- one dimension instead of the old (max q-tiles) x batch.
        # cdiv is subadditive by at most 1 per sequence, so this is an exact upper
        # bound on that sum needing no device sync; the kernel returns early on the
        # few trailing workgroups that own no tile.
        return (
            triton.cdiv(total_q, META["BLOCK_SIZE_Q"]) + batch_size - 1,
            num_heads,
        )

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        batch_size,
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
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
    )

    # topk extraction kernel, on the same flat grid: one workgroup per real q-block
    # per head, not max_seqblock_q per sequence. all_seqblock_q is the exact sum,
    # so unlike the score grid this needs no batch-1 slack.
    grid = (all_seqblock_q, num_heads)
    _topk_index_kernel[grid](
        score,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        batch_size,
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
