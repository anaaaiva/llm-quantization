import torch
import triton
import triton.language as tl


# ===========================================
# Triton kernel: groupwise int4 -> int32 packing
# ===========================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
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
    key=["group_size"],
)
@triton.jit
def _quant_groupwise_int4_int32(
    x_ptr,  # *const f32           - входной тензор (n_rows, n_cols), построчное хранение
    output_ptr,  # *mut i32        - выходной тензор (n_rows, n_cols/8), 8 int4 в одном int32
    output_scales,  # *mut f16     - массив длиной (n_rows * num_groups_per_row), scale на группу
    n_rows,  # i32                 - количество строк
    n_cols,  # i32                 - количество столбцов
    group_size,  # i32             - размер группы по столбцам
    BLOCK_SIZE: tl.constexpr,  #   - размер блока для прохода по группе
):
    """
    Групповое (groupwise) симметричное квантование float32 -> signed int4 [-8, 7]
    с упаковкой 8 значений в один int32.

    Для каждой группы (в пределах строки):
      1) вычисляется group_max = max(|x_j|) по элементам группы
      2) scale = 8 / max(group_max, 1e-12)
      3) квантованное значение q_j = clamp(round(x_j * scale), -8, 7)
      4) сохраним (для деквантования):
            group_max_safe в output_scales
         (фактический scale для деквантования = 8 / group_max_safe)
      5) 8 значений q_j пакуются в один int32:
            (q0+8) в биты [0..3], (q1+8) в [4..7], ... (q7+8) в [28..31]

    Сетка Triton:
      - 1 program_id (pid) обрабатывает одну группу в одной строке.
      - pid = row_idx * num_groups_per_row + group_idx
    """

    pid = tl.program_id(0)

    # Сколько групп в одной строке (ceil-div)
    num_groups_per_row = tl.cdiv(n_cols, group_size)

    # Восстанавливаем индексы строки и группы по pid
    row_idx = pid // num_groups_per_row
    group_idx = pid % num_groups_per_row

    # Если выходим за число строк — выходим
    if row_idx >= n_rows:
        return

    # Смещения по строке
    row_start = row_idx * n_cols
    group_start = group_idx * group_size
    group_end = tl.minimum(group_start + group_size, n_cols)
    actual_group_size = group_end - group_start

    # --- 1. Поиск group_max = max |x| по группе ---

    group_max = 0.0
    for start in range(0, actual_group_size, BLOCK_SIZE):
        col_offsets = group_start + start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < group_end

        global_offs = row_start + col_offsets
        block = tl.load(x_ptr + global_offs, mask=mask, other=0.0)

        abs_block = tl.abs(block)
        block_max = tl.max(tl.where(mask, abs_block, 0.0))
        group_max = tl.maximum(group_max, block_max)

    # --- 2. Вычисление scale и запись group_max (для деквантования) ---

    group_max_safe = tl.maximum(group_max, 1e-12)
    # scale для квантования: q = x * (8 / group_max_safe)
    scale = 8.0 / group_max_safe

    # Индекс группы в массиве scales (построчно)
    scale_idx = row_idx * num_groups_per_row + group_idx
    tl.store(output_scales + scale_idx, group_max_safe.to(tl.float16))

    # --- 3. Квантование и упаковка 8×int4 -> int32 ---

    # Для паковки 8 int4 в 1 int32 нужно, чтобы group_size по строке в сумме был кратен 8.
    # Внешняя обёртка проверяет n_cols % 8 == 0 и group_size % 8 == 0.
    # Здесь работаем с "реальным" размером группы (последняя может быть короче).
    packed_group_size = (actual_group_size + 7) // 8  # количество int32 на группу

    # Всего int32 на строку:
    packed_row_stride = n_cols // 8
    # Смещение начала упакованных данных для строки и группы
    packed_row_start = row_idx * packed_row_stride
    packed_group_start = packed_row_start + (group_start // 8)

    # Внутренние "lane"-индексы 0..7 — позиции внутри одного int32
    lanes = tl.arange(0, 8)  # [8]
    shifts = (lanes * 4).to(tl.int32)  # [8], сдвиг для каждого полубайта

    for packed_start in range(0, packed_group_size, BLOCK_SIZE):
        packed_offs = packed_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        mask_packed = packed_offs < packed_group_size  # [BLOCK_SIZE]

        # Для каждого packed_offs нужен блок из 8 исходных значений
        base_col = group_start + packed_offs[:, None] * 8  # [BLOCK_SIZE, 1]
        offs = base_col + lanes[None, :]  # [BLOCK_SIZE, 8]

        # Маска по колонкам и по границам группы
        mask_cols = (offs < group_end) & mask_packed[:, None]  # [BLOCK_SIZE, 8]

        global_offs = row_start + offs  # [BLOCK_SIZE, 8]
        vals = tl.load(x_ptr + global_offs, mask=mask_cols, other=0.0)  # [BLOCK_SIZE, 8]

        # Квантование: q = clamp(round(x * scale), -8, 7)
        q = tl.extra.cuda.libdevice.rint(vals * scale)
        q = tl.minimum(tl.maximum(q, -8.0), 7.0)
        q = q.to(tl.int32)  # [BLOCK_SIZE, 8]

        # Сдвигаем диапазон -8..7 -> 0..15 для упаковки в 4 бита
        q_u = (q + 8).to(tl.int32)  # [BLOCK_SIZE, 8], 0..15

        # Сдвигаем каждый lane к своей 4-битной позиции
        q_shifted = q_u << shifts[None, :]  # [BLOCK_SIZE, 8]

        # Сумма по оси 1 даёт одно int32 на каждый packed_offs
        packed = tl.sum(q_shifted, axis=1)  # [BLOCK_SIZE], i32

        # Глобальные индексы в упакованном выходе
        global_packed_offs = packed_group_start + packed_offs
        # Маска: 1) внутри группы, 2) не выходить за пределы строчного stride
        mask_store = mask_packed & (global_packed_offs < packed_row_start + packed_row_stride)

        tl.store(output_ptr + global_packed_offs, packed, mask=mask_store)


# ===========================================
# Python wrapper: quantize (groupwise, int4->int32)
# ===========================================
def quant_groupwise_int4_int32(x: torch.Tensor, group_size: int = 128):
    """
    Групповое симметричное квантование float32 -> int4 [-8, 7] с упаковкой
    8 значений в один int32.

    Параметры:
      x          : torch.Tensor [n_rows, n_cols], dtype=float32 (или приводится к float32)
      group_size : размер группы по колонкам (должен быть кратен 8)

    Возвращает:
      q_packed : torch.Tensor [n_rows, n_cols/8], dtype=torch.int32
      scales   : torch.Tensor [n_rows, num_groups_per_row], dtype=torch.float16
                 (хранит group_max для каждой группы)
    """
    assert x.ndim == 2, "ожидается двумерный тензор"
    n_rows, n_cols = x.shape

    # Для упаковки 8×4-bit в 32-bit нужна кратность 8
    assert n_cols % 8 == 0, "n_cols must be divisible by 8 for int4->int32 packing"
    assert group_size % 8 == 0, "group_size must be divisible by 8 for int4->int32 packing"

    x = x.to(torch.float32)

    packed_n_cols = n_cols // 8
    q_packed = torch.empty((n_rows, packed_n_cols), device=x.device, dtype=torch.int32)

    num_groups_per_row = (n_cols + group_size - 1) // group_size
    scales = torch.empty(n_rows * num_groups_per_row, device=x.device, dtype=torch.float16)

    total_blocks = n_rows * num_groups_per_row

    grid = (total_blocks,)

    _quant_groupwise_int4_int32[grid](x, q_packed, scales, n_rows, n_cols, group_size)

    scales = scales.view(n_rows, num_groups_per_row)
    return q_packed, scales


# ===========================================
# Python: dequantize (groupwise, int4 from int32)
# ===========================================
def dequant_groupwise_int4_int32(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
):
    """
    Деквантование из формата 8×int4 [-8,7], упакованных в int32.

    Формат:
      packed : [n_rows, n_packed], dtype=torch.int32
               каждый элемент содержит 8 квантованных значений int4.
      scales : [n_rows, num_groups_per_row], dtype=torch.float16
               хранит group_max для каждой группы.
      group_size : размер группы по колонкам (тот же, что в квантовании)

    Деквантование:
      q_j ∈ [-8,7] -> x_j ≈ q_j * (group_max / 8),
      где group_max берётся из соответствующей группе.
    """
    assert packed.dtype == torch.int32, "packed должен быть dtype=torch.int32"

    n_rows, packed_n_cols = packed.shape
    n_cols = packed_n_cols * 8
    num_groups_per_row = scales.shape[1]

    # Распаковка 8 значений из одного int32
    packed_i64 = packed.to(torch.int64)
    mask_4bit = 0xF

    parts = []
    for k in range(8):
        shift = 4 * k
        # Достаём 4 бита и приводим к uint8
        qk_u = ((packed_i64 >> shift) & mask_4bit).to(torch.int32)  # [n_rows, packed_n_cols]
        # Возвращаемся к signed int4: 0..15 -> -8..7
        qk = (qk_u - 8).to(torch.int32)
        parts.append(qk.unsqueeze(-1))  # [n_rows, packed_n_cols, 1]

    # [n_rows, packed_n_cols, 8] -> [n_rows, n_cols]
    q_all = torch.cat(parts, dim=-1).reshape(n_rows, n_cols)  # int32, значения в [-8, 7]

    # Теперь надо применить group_max / 8 для каждой группы
    dequantized = torch.empty((n_rows, n_cols), device=packed.device, dtype=torch.float16)

    for group_idx in range(num_groups_per_row):
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, n_cols)
        if group_start >= n_cols:
            break

        # group_max для этой группы, shape: [n_rows, 1]
        group_max = scales[:, group_idx : group_idx + 1].to(torch.float16)
        # фактический scale для деквантования: group_max / 8
        group_scale = group_max / 8.0

        dequantized[:, group_start:group_end] = q_all[:, group_start:group_end].to(torch.float16) * group_scale

    return dequantized
