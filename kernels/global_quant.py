import torch
import triton
import triton.language as tl


# ============================================================
# Global symmetric int4 -> int32 quantization kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    ],
    key=["n_cols"],
)
@triton.jit
def _quant_global_int4_int32(
    x_ptr,  # *const f32        - входной тензор (n_rows, n_cols), построчное хранение
    output_ptr,  # *mut i32     - выходной тензор (n_rows, n_cols/8), 8 int4 в одном int32
    scale,  # f32               - глобальный scale = 8 / global_max
    n_rows,  # i32              - количество строк
    n_cols,  # i32              - количество столбцов
    BLOCK_SIZE: tl.constexpr,
):
    """
    Глобальное симметричное квантование float32 -> signed int4 [-8, 7]
    с упаковкой 8 значений в один int32.

    GLOBAL (per-tensor) означает:
      - один scale для всех элементов: scale = 8 / global_max
      - global_max вычисляется снаружи (в Python) и передаётся сюда.

    Квантование:
      q = clamp(round(x * scale), -8, 7)
      упаковываем (q+8) в 4-битные поля одного int32.
    """

    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols

    # n_cols должно быть кратно 8 (проверяется в Python-обёртке)
    packed_row_stride = n_cols // 8
    packed_row_start = row_idx * packed_row_stride

    # Внутренние индексы 0..7 — позиции внутри одного int32
    lanes = tl.arange(0, 8)  # [8]
    shifts = (lanes * 4).to(tl.int32)  # [8], битовые сдвиги для полубайт

    for packed_start in range(0, packed_row_stride, BLOCK_SIZE):
        packed_offs = packed_start + tl.arange(0, BLOCK_SIZE)
        mask_packed = packed_offs < packed_row_stride

        # Для каждого packed_offs нужен блок из 8 элементов
        base_col = packed_offs[:, None] * 8  # [BLOCK_SIZE, 1]
        offs = base_col + lanes[None, :]  # [BLOCK_SIZE, 8]

        mask_cols = (offs < n_cols) & mask_packed[:, None]  # [BLOCK_SIZE, 8]
        global_offs = row_start + offs  # [BLOCK_SIZE, 8]

        vals = tl.load(x_ptr + global_offs, mask=mask_cols, other=0.0)

        # Квантование: q = clamp(round(x * scale), -8, 7)
        q = tl.extra.cuda.libdevice.rint(vals * scale)
        q = tl.minimum(tl.maximum(q, -8.0), 7.0)
        q = q.to(tl.int32)  # [BLOCK_SIZE, 8]

        # Диапазон -8..7 сдвигаем в 0..15 для упаковки в 4 бита
        q_u = (q + 8).to(tl.int32)  # [BLOCK_SIZE, 8], 0..15

        # Сдвигаем каждый lane к своей 4-битной позиции
        q_shifted = q_u << shifts[None, :]  # [BLOCK_SIZE, 8]

        # Сумма по оси 1 даёт одно int32 на каждый packed_offs
        packed = tl.sum(q_shifted, axis=1)  # [BLOCK_SIZE], i32

        tl.store(output_ptr + packed_row_start + packed_offs, packed, mask=mask_packed)


# ============================================================
# Python wrappers: quantize / dequantize (global symmetric)
# ============================================================
def quant_global_int4_int32(x: torch.Tensor):
    """
    Глобальное (per-tensor) симметричное квантование float32 -> int4 [-8, 7]
    с упаковкой 8 значений в один int32.

    Параметры:
      x : torch.Tensor [n_rows, n_cols], CUDA, dtype=float32 (или приводится к float32)

    Возвращает:
      q_packed   : [n_rows, n_cols/8], dtype=torch.int32  — упакованные int4
      global_max : scalar torch.float16                    — absmax по всему тензору
    """
    assert x.is_cuda and x.dim() == 2, "x должен быть CUDA 2D-тензором"
    n_rows, n_cols = x.shape
    assert n_cols % 8 == 0, "n_cols must be divisible by 8 for int4->int32 packing"

    x = x.to(torch.float32)

    # Глобальный absmax по всему тензору
    global_max = x.abs().max()
    global_max_safe = torch.clamp(global_max, min=1e-12)
    scale = (8.0 / global_max_safe).item()  # scalar float для Triton

    packed_n_cols = n_cols // 8
    q_packed = torch.empty((n_rows, packed_n_cols), device=x.device, dtype=torch.int32)

    grid_q = (n_rows,)
    _quant_global_int4_int32[grid_q](x, q_packed, scale, n_rows, n_cols)

    return q_packed, global_max_safe.to(torch.float16)


def dequant_global_int4_int32(
    packed: torch.Tensor,
    global_max: torch.Tensor,
):
    """
    Деквантование из формата global symmetric int4 ([-8,7]), упакованного в int32.

    Параметры:
      packed     : [n_rows, n_packed], dtype=torch.int32
                   каждый элемент хранит 8 значений int4.
      global_max : scalar tensor (float16 или float32) — absmax по всему тензору.

    Формула:
      scale = global_max / 8
      q ∈ [-8,7] -> x ≈ q * scale
    """
    assert packed.is_cuda, "packed должен быть CUDA-тензором"
    assert packed.dtype == torch.int32, "packed должен быть dtype=torch.int32"
    assert packed.dim() == 2, "packed должен быть 2D"

    if global_max.numel() != 1:
        raise ValueError("global_max должен быть скаляром (тензор с 1 элементом)")

    n_rows, packed_n_cols = packed.shape
    n_cols = packed_n_cols * 8

    # Распаковка 8 значений из одного int32
    packed_i64 = packed.to(torch.int64)
    mask_4bit = 0xF

    parts = []
    for k in range(8):
        shift = 4 * k
        # 0..15
        qk_u = ((packed_i64 >> shift) & mask_4bit).to(torch.int32)
        # -8..7
        qk = (qk_u - 8).to(torch.int32)
        parts.append(qk.unsqueeze(-1))

    q_all = torch.cat(parts, dim=-1).reshape(n_rows, n_cols)  # [n_rows, n_cols], int32

    # scale = global_max / 8
    global_max = global_max.to(torch.float16)
    scale = (global_max / 8.0).to(torch.float16)  # scalar

    dequantized = q_all.to(torch.float16) * scale
    return dequantized
