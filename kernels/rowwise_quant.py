import torch
import triton
import triton.language as tl


# ===========================================
# Triton kernel: rowwise int4 -> int32 packing (signed [-8, 7])
# ===========================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=3),
    ],
    key=["n_cols"],
)
@triton.jit
def _quant_rowwise_int4_int32(
    x_ptr,  # *const f32         - входной тензор (n_rows, n_cols), построчное хранение
    output_ptr,  # *mut i32      - выходной тензор (n_rows, n_cols/8), 8 int4 в одном int32
    output_maxs,  # *mut f16     - массив длиной n_rows, row_max (absmax) для каждой строки
    n_rows,  # i32               - количество строк
    n_cols,  # i32               - количество столбцов
    BLOCK_SIZE: tl.constexpr,  # - размер блока по колонкам
):
    """
    Построчное симметричное квантование float32 -> signed int4 [-8, 7]
    с упаковкой 8 значений в один int32.

    Для каждой строки:
      1) вычисляется row_max = max(|x_j|) по всем элементам строки
      2) scale = 8 / max(row_max, 1e-12)
      3) квантованное значение q_j = clamp(round(x_j * scale), -8, 7)
      4) сохраняем row_max_safe (для деквантования: x ≈ q * (row_max/8))
      5) 8 значений q_j пакуются в одно 32-битное int32:
           (q0+8) в биты [0..3], (q1+8) в [4..7], ..., (q7+8) в [28..31]
    """

    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols

    # --- 1. Поиск row_max = max |x| по строке (одним блоком) ---

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)

    abs_row = tl.abs(row_vals)
    row_max = tl.max(tl.where(mask, abs_row, 0.0))

    row_max_safe = tl.maximum(row_max, 1e-12)
    scale = 8.0 / row_max_safe

    # Сохраняем absmax (row_max_safe) для строки
    tl.store(output_maxs + row_idx, row_max_safe.to(tl.float16))

    # --- 2. Квантование и упаковка 8×int4 -> int32 ---

    # n_cols должно быть кратно 8 (проверяется в Python-обёртке)
    packed_row_stride = n_cols // 8
    packed_row_start = row_idx * packed_row_stride

    # Внутренние "lane"-индексы 0..7 — позиции внутри одного int32
    lanes = tl.arange(0, 8)  # [8]
    shifts = (lanes * 4).to(tl.int32)  # [8], сдвиг для каждого полубайта

    for packed_start in range(0, packed_row_stride, BLOCK_SIZE):
        packed_offs = packed_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        mask_packed = packed_offs < packed_row_stride  # [BLOCK_SIZE]

        # Для каждого packed_offs нужно 8 элементов строки
        base_col = packed_offs[:, None] * 8  # [BLOCK_SIZE, 1]
        offs = base_col + lanes[None, :]  # [BLOCK_SIZE, 8]

        mask_cols = (offs < n_cols) & mask_packed[:, None]  # [BLOCK_SIZE, 8]
        global_offs = row_start + offs  # [BLOCK_SIZE, 8]

        vals = tl.load(x_ptr + global_offs, mask=mask_cols, other=0.0)  # [BLOCK_SIZE, 8]

        # Квантование: q = clamp(round(x * scale), -8, 7)
        q = tl.extra.cuda.libdevice.rint(vals * scale)
        q = tl.minimum(tl.maximum(q, -8.0), 7.0)
        q = q.to(tl.int32)  # [BLOCK_SIZE, 8]

        # Сдвигаем диапазон -8..7 -> 0..15 для упаковки в 4 бита
        q_u = (q + 8).to(tl.int32)  # [BLOCK_SIZE, 8]

        # Сдвигаем каждый lane к своей 4-битной позиции
        q_shifted = q_u << shifts[None, :]  # [BLOCK_SIZE, 8]

        # Сумма по оси 1 даёт одно int32 на каждый packed_offs
        packed = tl.sum(q_shifted, axis=1)  # [BLOCK_SIZE], i32

        tl.store(output_ptr + packed_row_start + packed_offs, packed, mask=mask_packed)


# ===========================================
# Python wrapper: quantize rowwise int4 -> int32
# ===========================================
def quant_rowwise_int4_int32(x: torch.Tensor):
    """
    Построчное симметричное квантование float32 -> int4 [-8, 7]
    с упаковкой 8 значений в один int32.

    Параметры:
      x : torch.Tensor [n_rows, n_cols], CUDA, dtype=float32 (или приводится к float32)

    Возвращает:
      q_packed : torch.Tensor [n_rows, n_cols/8], dtype=torch.int32
                 В каждом int32 упаковано 8 значений int4.
      absmaxs  : torch.Tensor [n_rows], dtype=torch.float16
                 Абсолютный максимум по строке (row_max).
    """
    assert x.is_cuda and x.dim() == 2, "x must be CUDA-tensor 2D"
    n_rows, n_cols = x.shape

    # Для упаковки 8×4-bit в 32-bit
    assert n_cols % 8 == 0, "n_cols must be divisible by 8 for int4->int32 packing"

    x = x.to(torch.float32)

    packed_n_cols = n_cols // 8
    q_packed = torch.empty((n_rows, packed_n_cols), device=x.device, dtype=torch.int32)
    absmaxs = torch.empty(n_rows, device=x.device, dtype=torch.float16)

    grid = (n_rows,)
    _quant_rowwise_int4_int32[grid](x, q_packed, absmaxs, n_rows, n_cols)

    return q_packed, absmaxs


# ===========================================
# Python: dequantize rowwise int4 from int32
# ===========================================
def dequant_rowwise_int4_int32(
    packed: torch.Tensor,
    absmaxs: torch.Tensor,
):
    """
    Деквантование из формата 8×int4 [-8,7], упакованных в int32.

    Формат:
      packed : [n_rows, n_packed], dtype=torch.int32
               каждый элемент содержит 8 квантованных значений int4.
      absmaxs: [n_rows], dtype=torch.float16
               row_max (absmax) по каждой строке.

    Деквантование:
      q_j ∈ [-8,7] -> x_j ≈ q_j * (row_max / 8),
      где row_max берётся из absmaxs для соответствующей строки.
    """
    assert packed.is_cuda and absmaxs.is_cuda, "packed и absmaxs должны быть CUDA-тензорами"
    assert packed.dtype == torch.int32, "packed должен быть dtype=torch.int32"
    assert absmaxs.dtype == torch.float16, "absmaxs должен быть dtype=torch.float16"
    assert packed.dim() == 2 and absmaxs.dim() == 1, "размерности: packed [2D], absmaxs [1D]"
    assert packed.size(0) == absmaxs.size(0), "число строк в packed и absmaxs должно совпадать"

    n_rows, packed_n_cols = packed.shape
    n_cols = packed_n_cols * 8

    # Распаковка 8 значений из одного int32
    packed_i64 = packed.to(torch.int64)
    mask_4bit = 0xF

    parts = []
    for k in range(8):
        shift = 4 * k
        # Достаём 4 бита и приводим к int32
        qk_u = ((packed_i64 >> shift) & mask_4bit).to(torch.int32)  # [n_rows, packed_n_cols]
        # Возвращаемся к signed int4: 0..15 -> -8..7
        qk = (qk_u - 8).to(torch.int32)
        parts.append(qk.unsqueeze(-1))  # [n_rows, packed_n_cols, 1]

    # [n_rows, packed_n_cols, 8] -> [n_rows, n_cols]
    q_all = torch.cat(parts, dim=-1).reshape(n_rows, n_cols)  # int32, значения в [-8, 7]

    # Применяем row_max / 8 для каждой строки
    dequantized = torch.empty((n_rows, n_cols), device=packed.device, dtype=torch.float16)

    # scale_row = row_max / 8, shape: [n_rows, 1]
    scale_row = (absmaxs.to(torch.float16) / 8.0).unsqueeze(1)  # [n_rows, 1]
    dequantized[:] = q_all.to(torch.float16) * scale_row  # broadcasting по колонкам

    return dequantized
