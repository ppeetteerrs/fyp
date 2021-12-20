import os
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function
from torch.functional import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(
        ctx: Any,
        grad_output: Tensor,
        out: Tensor,
        bias: Tensor,
        negative_slope: float,
        scale: float,
    ):
        # Save context
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        ctx.save_for_backward(out)

        # Dummy
        empty = grad_output.new_empty(0)

        grad_input: Tensor = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = None

        return grad_input, grad_bias

    @staticmethod
    def backward(
        ctx: Any, gradgrad_input: Tensor, gradgrad_bias: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        (out,) = ctx.saved_tensors
        gradgrad_out: Tensor = fused.fused_bias_act(
            gradgrad_input.contiguous(),
            gradgrad_bias,
            out,
            ctx.negative_slope,
            ctx.scale,
        )

        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        bias: Optional[Tensor],
        negative_slope: float,
        scale: float,
    ) -> Tensor:
        # Dummy tensor for unused inputs to C++
        empty = input.new_empty(0)

        # Save bias context
        ctx.bias = bias is not None

        bias = empty if bias is None else bias

        output: Tensor = fused.fused_bias_act(input, bias, empty, negative_slope, scale)

        # Save settings and output to context
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        (output,) = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, output, ctx.bias, ctx.negative_slope, ctx.scale
        )
        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(
        self,
        n_channels: int,
        bias: bool = True,
        negative_slope: float = 0.2,
        scale: float = 2 ** 0.5,
    ):
        """
        Leaky ReLU with / without learnable biases

            Purpose of scale factor (StyleGAN2 Paper):
                "To avoid having to account for the activation function in Equation 3, we scale our
                activation functions so that they retain the expected signal variance
                (instead of simply initializing with variance-preserving weights)"
        """
        super().__init__()
        self.bias = Parameter(torch.zeros(n_channels)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(
    input: Tensor,
    bias: Optional[Tensor] = None,
    negative_slope: float = 0.2,
    scale: float = 2 ** 0.5,
) -> Any:
    return FusedLeakyReLUFunction.apply(input.contiguous(), bias, negative_slope, scale)
