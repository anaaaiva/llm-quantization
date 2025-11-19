from typing import List, Optional

import torch
import torch.nn as nn
from global_quant import quant_global_int4_int32
from groupwise_quant import quant_groupwise_int4_int32
from rowwise_quant import quant_rowwise_int4_int32
from unified_matmul import matmul_fp16_int4_unified


# ------------------------------------------------------------
# Квантованный Linear на int4 (global/rowwise/groupwise)
# ------------------------------------------------------------
class QuantLinearInt4(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "rowwise",
        group_size: Optional[int] = None,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.bias = None

        self.register_buffer("b_packed", None, persistent=False)
        self.register_buffer("meta", None, persistent=False)

    @torch.no_grad()
    def quantize_from_linear(self, linear: nn.Linear):
        W = linear.weight.to(torch.float16)  # [out_features, in_features]
        assert W.shape == (self.out_features, self.in_features)
        mode = self.mode

        if mode == "global":
            b_packed, meta = quant_global_int4_int32(W)
        elif mode == "rowwise":
            b_packed, meta = quant_rowwise_int4_int32(W)
        elif mode == "groupwise":
            assert self.group_size is not None, "group_size must be set for groupwise mode"
            assert self.in_features % self.group_size == 0, (
                f"in_features={self.in_features} must be divisible by group_size={self.group_size}"
            )
            b_packed, meta = quant_groupwise_int4_int32(W, group_size=self.group_size)
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")

        self.b_packed = b_packed
        self.meta = meta

        if self.bias is not None and linear.bias is not None:
            self.bias.data.copy_(linear.bias.to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.b_packed is not None and self.meta is not None, "Layer is not quantized yet"
        orig_shape = x.shape
        assert orig_shape[-1] == self.in_features

        x_2d = x.reshape(-1, self.in_features).to(torch.float16).contiguous()

        if self.mode == "global":
            y_2d = matmul_fp16_int4_unified(x_2d, self.b_packed, self.meta, mode="global")
        elif self.mode == "rowwise":
            y_2d = matmul_fp16_int4_unified(x_2d, self.b_packed, self.meta, mode="rowwise")
        elif self.mode == "groupwise":
            y_2d = matmul_fp16_int4_unified(
                x_2d, self.b_packed, self.meta, mode="groupwise", group_size=self.group_size
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.bias is not None:
            y_2d = y_2d + self.bias.to(y_2d.dtype)

        y = y_2d.reshape(*orig_shape[:-1], self.out_features)
        return y


# ------------------------------------------------------------
# Замена всех nn.Linear в модели на QuantLinearInt4
# ------------------------------------------------------------
def quantize_all_linears(
    model: nn.Module,
    mode: str = "rowwise",
    group_size: Optional[int] = None,
    not_quant_module_names: Optional[List[str]] = [],
):
    for name, module in model.named_children():
        quantize_all_linears(module, mode=mode, group_size=group_size)

        if isinstance(module, nn.Linear) and name not in not_quant_module_names:
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            qlinear = QuantLinearInt4(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                mode=mode,
                group_size=group_size,
                device=module.weight.device,
                dtype=torch.float16,
            )
            qlinear.quantize_from_linear(module)
            setattr(model, name, qlinear)
