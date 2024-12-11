# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time compilation for CUDA"""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Just-in-time compilation for CUDA"""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Handles zip operation"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Handles reduce operation"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Handles matrix multiplication"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    #  /$$      /$$
    # | $$$    /$$$
    # | $$$$  /$$$$  /$$$$$$   /$$$$$$
    # | $$ $$/$$ $$ |____  $$ /$$__  $$
    # | $$  $$$| $$  /$$$$$$$| $$  \ $$
    # | $$\  $ | $$ /$$__  $$| $$  | $$
    # | $$ \/  | $$|  $$$$$$$| $$$$$$$/
    # |__/     |__/ \_______/| $$____/
    #                        | $$
    #                        | $$
    #                        |__/
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)
            in_value = in_storage[in_position]
            out[out_position] = fn(in_value)

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    #  /$$$$$$$$ /$$
    # |_____ $$ |__/
    #      /$$/  /$$  /$$$$$$
    #     /$$/  | $$ /$$__  $$
    #    /$$/   | $$| $$  \ $$
    #   /$$/    | $$| $$  | $$
    #  /$$$$$$$$| $$| $$$$$$$/
    # |________/|__/| $$____/
    #               | $$
    #               | $$
    #               |__/
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            a_value = a_storage[a_position]
            b_value = b_storage[b_position]

            out_position = index_to_position(out_index, out_strides)
            out[out_position] = fn(a_value, b_value)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Behold ye a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()
    # reduce in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sum practice function"""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """
    #  /$$$$$$$                  /$$
    # | $$__  $$                | $$
    # | $$  \ $$  /$$$$$$   /$$$$$$$ /$$   /$$  /$$$$$$$  /$$$$$$
    # | $$$$$$$/ /$$__  $$ /$$__  $$| $$  | $$ /$$_____/ /$$__  $$
    # | $$__  $$| $$$$$$$$| $$  | $$| $$  | $$| $$      | $$$$$$$$
    # | $$  \ $$| $$_____/| $$  | $$| $$  | $$| $$      | $$_____/
    # | $$  | $$|  $$$$$$$|  $$$$$$$|  $$$$$$/|  $$$$$$$|  $$$$$$$
    # |__/  |__/ \_______/ \_______/ \______/  \_______/ \_______/

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            out_position = index_to_position(out_index, out_strides)
            if pos < a_shape[reduce_dim]:
                read_from = (
                    index_to_position(out_index, a_strides)
                    + pos * a_strides[reduce_dim]
                )
                cache[pos] = a_storage[read_from]
            else:
                cache[pos] = reduce_value
            cuda.syncthreads()  # wait for all threads in the block to finish copying data to shared memory
            # reduce in parallel in shared memory
            stride = (
                BLOCK_DIM // 2
            )  # combine pos & pos + 512, then pos & pos + 256, then pos & pos + 128, etc.
            while stride > 0:
                if (
                    pos < stride
                ):  # only threads with pos < stride will participate in the reduction
                    cache[pos] = fn(
                        cache[pos], cache[pos + stride]
                    )  # reduce the values at pos and pos + stride and store the result in pos
                stride //= 2  # halve the stride for the next iteration until we reach 0
                cuda.syncthreads()
            # write the result to global memory
            if pos == 0:
                out[out_position] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Behold ye a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    thread_x = cuda.threadIdx.x
    thread_y = cuda.threadIdx.y
    out_row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    out_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    summa = 0.0

    for t in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        if out_row < size and t * BLOCK_DIM + thread_x < size:
            a_shared[thread_y, thread_x] = a[out_row * size + t * BLOCK_DIM + thread_x]
        else:
            a_shared[thread_y, thread_x] = 0.0  # Load 0 if out of bounds

        if out_col < size and t * BLOCK_DIM + thread_y < size:
            b_shared[thread_y, thread_x] = b[
                (t * BLOCK_DIM + thread_y) * size + out_col
            ]
        else:
            b_shared[thread_y, thread_x] = 0.0

        cuda.syncthreads()
        for k in range(BLOCK_DIM):
            summa += a_shared[thread_y, k] * b_shared[k, thread_x]
        # wait for all threads in the block to finish computing the sum
        cuda.syncthreads()
    # PLEASE WORK THIS TIME PLEASE I BEG YOU
    if out_row < size and out_col < size:
        out[out_row * size + out_col] = summa


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Matrix multiply practice function"""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


#  /$$      /$$             /$$     /$$      /$$           /$$
# | $$$    /$$$            | $$    | $$$    /$$$          | $$
# | $$$$  /$$$$  /$$$$$$  /$$$$$$  | $$$$  /$$$$ /$$   /$$| $$
# | $$ $$/$$ $$ |____  $$|_  $$_/  | $$ $$/$$ $$| $$  | $$| $$
# | $$  $$$| $$  /$$$$$$$  | $$    | $$  $$$| $$| $$  | $$| $$
# | $$\  $ | $$ /$$__  $$  | $$ /$$| $$\  $ | $$| $$  | $$| $$
# | $$ \/  | $$|  $$$$$$$  |  $$$$/| $$ \/  | $$|  $$$$$$/| $$
# |__/     |__/ \_______/   \___/  |__/     |__/ \______/ |__/
def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    iter_dim_length = a_shape[
        -1
    ]  # length of the inner dimension of a and b over which we must iterate
    # The local position in the block.
    thread_i = cuda.threadIdx.x
    thread_j = cuda.threadIdx.y

    summa = 0.0
    tile_count = (
        iter_dim_length + BLOCK_DIM - 1
    ) // BLOCK_DIM  # number of tiles needed to span the entire row of a and col of b
    for t in range(tile_count):
        if (i < a_shape[-2]) and (
            (t * BLOCK_DIM + thread_j) < a_shape[-1]
        ):  # make sure we are in bounds for a's shape
            a_shared[thread_i, thread_j] = a_storage[
                batch * a_batch_stride  # batch dimension's contribution to position
                + i * a_strides[-2]  # row dimension's contribution to position
                + (t * BLOCK_DIM + thread_j)
                * a_strides[-1]  # col dimension's contribution to position
            ]
        else:
            a_shared[thread_i, thread_j] = 0.0  # pad with 0s if out of bounds

        if (thread_j < b_shape[-1]) and (
            (t * BLOCK_DIM + thread_i) < b_shape[-2]
        ):  # make sure we are in bounds for b's shape
            b_shared[thread_i, thread_j] = b_storage[
                batch * b_batch_stride  # batch dimension's contribution to position
                + (t * BLOCK_DIM + thread_i)
                * b_strides[-2]  # row dimension's contribution to position
                + j * b_strides[-1]  # col dimension's contribution to position
            ]
        else:
            b_shared[thread_i, thread_j] = 0.0  # pad with 0s if out of bounds
        cuda.syncthreads()  # wait for all threads in the block to finish copying data to shared memory

        for k in range(BLOCK_DIM):  # iterate over the shared memory to compute the sum
            summa += a_shared[thread_i, k] * b_shared[k, thread_j]
        cuda.syncthreads()  # wait for all threads in the block to finish computing the sum

    if i < out_shape[-2] and j < out_shape[-1]:  # check if we are in bounds for out
        out_position = (
            batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        )  # since we wish to write to (batch, i, j) position in out
        out[out_position] = summa


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
