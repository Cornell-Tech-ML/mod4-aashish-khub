from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw
    kernel_size = kh * kw
    input = input.contiguous().view(
        batch, channel, new_height, kh, new_width, kw
    )  # shape is batch x channel x new_height x kh x new_width x kw
    input = input.permute(
        0, 1, 2, 4, 3, 5
    )  # NOW shape is batch x channel x new_height x new_width x kh x kw
    new_tensor = input.contiguous().view(
        batch, channel, new_height, new_width, kernel_size
    )
    return new_tensor, new_height, new_width


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to a tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    orig_batch, orig_channel, orig_height, orig_width = input.shape
    new_tensor, new_height, new_width = tile(
        input, kernel
    )  # shape is batch x channel x new_height x new_width x (kernel_height * kernel_width)
    # print("New tensor shape inside maxpool2d: ", new_tensor.shape)
    new_tensor_pooled = new_tensor.mean(
        4
    )  # shape is batch x channel x new_height x new_width
    output = new_tensor_pooled.contiguous().view(
        orig_batch, orig_channel, new_height, new_width
    )  # shape is batch x channel x new_height x new_width
    return output


# - argmax: Compute the argmax as a 1-hot tensor


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax of a tensor

    Args:
    ----
        input: tensor to compute
        dim: dimension to compute the argmax along

    Returns:
    -------
        Tensor with 1 at the maximum value and 0 elsewhere

    """
    max_vals = input.f.max_reduce(input, dim)
    argmax_mask = input.f.eq_zip(input, max_vals)
    return argmax_mask


# - Max: New Function for max operator


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max function"""
        max_vals = a.f.max_reduce(a, int(dim.item()))
        argmax_mask = a.f.eq_zip(a, max_vals)
        ctx.save_for_backward(argmax_mask)
        return max_vals

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the max"""
        (argmax_mask,) = ctx.saved_values
        masked_product = grad_output.f.mul_zip(argmax_mask, grad_output)
        return masked_product, 0.0


# - max: Apply max reduction


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of a tensor

    Args:
    ----
        input: tensor to compute
        dim: dimension to compute the max along

    Returns:
    -------
        Tensor with the maximum value

    """
    if dim < 0:
        tensor_maxxing = input.contiguous().view(input.size)
        dim_to_max_over = input._ensure_tensor(0)
    else:
        tensor_maxxing = input
        dim_to_max_over = input._ensure_tensor(dim)
    tensor_maxxing = Max.apply(tensor_maxxing, dim_to_max_over)
    return tensor_maxxing


# - softmax: Compute the softmax as a tensor
def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor

    Args:
    ----
        input: tensor to compute
        dim: dimension to compute the softmax along

    Returns:
    -------
        Tensor with the softmax value

    """
    exp_tensor = input.exp()
    sum_exp = exp_tensor.sum(dim)
    normalized = exp_tensor / sum_exp
    return normalized


# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax of a tensor

    Args:
    ----
        input: tensor to compute
        dim: dimension to compute the log softmax along

    Returns:
    -------
        Tensor with the log softmax value

    """
    max_vals = max(input, dim)
    input_shifted = input - max_vals
    exp_tensor = input_shifted.exp()
    sum_exp = exp_tensor.sum(dim)
    log_sum_exp = sum_exp.log()
    return input_shifted - log_sum_exp


# - maxpool2d: Tiled max pooling 2D
def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to a tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    orig_batch, orig_channel, orig_height, orig_width = input.shape
    new_tensor, new_height, new_width = tile(
        input, kernel
    )  # shape is batch x channel x new_height x new_width x (kernel_height * kernel_width)
    print("New tensor shape inside maxpool2d: ", new_tensor.shape)
    # raise("Error")
    new_tensor_pooled = max(
        new_tensor, 4
    )  # shape is batch x channel x new_height x new_width
    output = new_tensor_pooled.contiguous().view(
        orig_batch, orig_channel, new_height, new_width
    )  # shape is batch x channel x new_height x new_width
    return output


# - dropout: Dropout positions based on random noise, include an argument to turn off
def dropout(input: Tensor, dropout_prob: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor

    Args:
    ----
        input: tensor to apply dropout
        dropout_prob: probability of dropout
        ignore: whether to disable dropout or not e.g. when evaluating

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore or dropout_prob == 0.0:
        return input
    dropout_prob_tensor = input.zeros(input.shape) + dropout_prob
    dropout_mask = (
        rand(input.shape, backend=input.backend, requires_grad=False)
        > dropout_prob_tensor
    )
    return input * dropout_mask
