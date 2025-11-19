import torch
import triton
import triton.language as tl


# ============================================================
# Triton kernel (unified for global/rowwise/groupwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_fp16_int4_unified_kernel(
    a_ptr,  # *const fp16             - A [M, K]
    b_packed_ptr,  # *const i32       - B_packed [N, K/8]
    meta_ptr,  # *const f16/f32       - метаданные (global_max / absmaxs / scales)
    c_ptr,  # *mut fp16               - C [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    global_scale,  # f32, global_max/8 (используется только если mode==0)
    mode,  # i32: 0=global, 1=rowwise, 2=groupwise
    group_size,  # i32: для groupwise
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # lane-ы для распаковки 8×int4
    lanes = tl.arange(0, 8)
    shifts = (lanes * 4).to(tl.int32)

    # helper для groupwise
    num_groups = K // group_size if group_size > 0 else 1

    for k0 in range(0, K, BLOCK_SIZE_K):
        rk = k0 + tl.arange(0, BLOCK_SIZE_K)
        mask_k = rk < K

        # ----- A -----
        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a_block = a_block.to(tl.float32)  # [BM, BK]

        # ----- B (packed int4 -> unpack) -----
        packed_k = rk // 8
        mask_packed_k = packed_k < (K // 8)

        b_packed_ptrs = b_packed_ptr + rn[:, None] * stride_bn + packed_k[None, :] * stride_bk
        b_packed = tl.load(
            b_packed_ptrs,
            mask=mask_n[:, None] & mask_packed_k[None, :],
            other=0,
        )  # [BN, BK], каждый элемент хранит 8 int4

        b_packed_i32 = b_packed.to(tl.int32)[:, :, None]  # [BN, BK, 1]
        q_u_all = (b_packed_i32 >> shifts[None, None, :]) & 0xF  # [BN, BK, 8], 0..15
        q_all = q_u_all.to(tl.int32) - 8  # signed [-8..7]

        lane_index = (rk % 8).to(tl.int32)  # [BK]
        lane_index_b = lane_index[None, :, None]  # [1, BK, 1]
        lane_index_b = tl.broadcast_to(lane_index_b, q_all.shape)  # [BN, BK, 8]

        mask_lane = lanes[None, None, :] == lane_index_b
        q_selected = tl.where(mask_lane, q_all, 0)
        b_block = tl.sum(q_selected, axis=2)  # [BN, BK]
        b_block = b_block.to(tl.float32)
        b_block = tl.where(mask_k[None, :], b_block, 0.0)

        # ----- Применяем scale в зависимости от режима -----
        # mode: 0=global, 1=rowwise, 2=groupwise
        if mode == 0:
            # GLOBAL: один scale для всех
            b_block = b_block * global_scale

        elif mode == 1:
            # ROWWISE: meta_ptr[n], absmax per row
            # scale_row = absmax[n] / 8
            absmaxs_ptrs = meta_ptr + rn
            absmaxs = tl.load(absmaxs_ptrs, mask=mask_n, other=1.0)  # [BN]
            scale_rows = absmaxs.to(tl.float32) / 8.0  # [BN]
            b_block = b_block * scale_rows[:, None]

        elif mode == 2:
            # GROUPWISE: meta_ptr[n, group_id], group_size along K
            # scale[n, group_id] / 8. group_id = rk // group_size
            group_ids = (rk // group_size).to(tl.int32)  # [BK]
            # base ptr for row n: meta_ptr + n * num_groups
            base_row_ptrs = meta_ptr + rn * num_groups  # [BN]

            # [BN, BK] индексы по meta
            scales_ptrs = base_row_ptrs[:, None] + group_ids[None, :]  # [BN, BK]
            scales = tl.load(scales_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=1.0)
            scale_vals = scales.to(tl.float32) / 8.0  # [BN, BK]
            b_block = b_block * scale_vals

        # ----- Dot -----
        b_block_t = tl.trans(b_block)  # [BK, BN]
        acc += tl.dot(a_block, b_block_t, allow_tf32=False)

    # ----- Store C -----
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ============================================================
# High-level: единый matmul для 3 схем квантования весов
# ============================================================
def matmul_fp16_int4_unified(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    meta: torch.Tensor,
    mode: str,
    group_size: int | None = None,
) -> torch.Tensor:
    """
    C = A @ B^T, где B квантована в int4->int32 одной из схем:
      - mode = "global":   meta -> global_max (scalar)
      - mode = "rowwise":  meta -> absmaxs [N]
      - mode = "groupwise": meta -> scales [N, num_groups], group_size обязателен

    Форматы:
      A: [M, K], float16
      B: [N, K] (логически), хранится как:
        b_packed: [N, K/8], int32 (8 signed int4 в одном int32)

    Параметры:
      mode:
        "global"   -> global symmetric per-tensor
        "rowwise"  -> symmetric per-row
        "groupwise"-> symmetric per-group (по K, с заданным group_size)

      meta:
        - "global":   scalar tensor (global_max)
        - "rowwise":  [N], absmaxs
        - "groupwise":[N, K/group_size], scales
    """

    assert a.is_cuda and b_packed.is_cuda and meta.is_cuda, "All tensors must be CUDA"
    assert a.dtype == torch.float16, f"A must be fp16, got {a.dtype}"
    assert b_packed.dtype == torch.int32, f"b_packed must be int32, got {b_packed.dtype}"

    M, K = a.shape
    N, packed_K = b_packed.shape
    assert K % 8 == 0, "K must be divisible by 8 for int4->int32 packing"
    assert packed_K == K // 8, f"b_packed.shape[1] must be K/8, got {packed_K}, expected {K // 8}"

    mode_map = {"global": 0, "rowwise": 1, "groupwise": 2}
    assert mode in mode_map, f"mode must be one of {list(mode_map)}, got {mode}"
    mode_int = mode_map[mode]

    # Проверяем метаданные под конкретный режим
    if mode == "global":
        assert meta.numel() == 1, "global mode: meta must be scalar global_max"
        scale = (meta.to(torch.float32) / 8.0).item()  # scalar f32
        meta_ptr = meta
        group_size_val = 0  # не используется
    elif mode == "rowwise":
        assert meta.shape == (N,), f"rowwise mode: meta must be [N], got {meta.shape}"
        scale = 1.0  # rowwise scale вычисляем внутри ядра
        meta_ptr = meta
        group_size_val = 0
    elif mode == "groupwise":
        assert group_size is not None, "groupwise mode: group_size must be provided"
        assert K % group_size == 0, "K must be divisible by group_size"
        num_groups = K // group_size
        assert meta.shape == (N, num_groups), (
            f"groupwise mode: meta must be [N, K/group_size]={N, num_groups}, got {meta.shape}"
        )
        scale = 1.0  # groupwise scale внутри ядра
        meta_ptr = meta
        group_size_val = group_size
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = (
        triton.cdiv(M, 64),
        triton.cdiv(N, 64),
    )

    _matmul_fp16_int4_unified_kernel[grid](
        a,
        b_packed,
        meta_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        c.stride(0),
        c.stride(1),
        scale,  # global scale (используется только в global-режиме)
        mode_int,  # 0=global,1=rowwise,2=groupwise
        group_size_val,  # для groupwise
    )

    return c
