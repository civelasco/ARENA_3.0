# %% 
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"
# %%
""" Basic einops operations """

arr = np.load(section_dir / "numbers.npy")
# %%
display_array_as_img(arr[0])
# %%
arr_column = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr_column)
# %%
arr_rep = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr_rep)
# %%
arr_rep2 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr_rep2)
# %%
arr_stretch = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr_stretch)
# %%
arr_split = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr_split)
# %%
arr_stack = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1 = 2)
display_array_as_img(arr_stack)
# %%
arr_transpose = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr_transpose)
# %%
arr_normal = einops.rearrange(arr, "b c h w -> c h (b w)")
arr_shrink = einops.reduce(arr, "b c (h h1) (w w1) -> c h (b w)", "max", h1=2, w1=2)
display_array_as_img(arr_normal)
display_array_as_img(arr_shrink)
# %%
""" einops operations we will be using in practice """
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")
# %%
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    arr = t.arange(3,9)
    return einops.rearrange(arr, "(h w) -> h w", h = 3)

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)
# %%
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    arr = t.arange(1,7)
    return einops.rearrange(arr, "(h w) -> h w", h = 2)

assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

# %%
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(h 7) -> h", "mean")

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))

# %%
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg_temps = einops.reduce(temps, "(h 7) -> h", "mean")
    list1     = einops.repeat(avg_temps, "h -> (h 7)")
    list2     = temps-list1
    return list2

expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

# %%
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    avg_temps = einops.reduce(temps, "(h 7) -> h", "mean")
    avg_temps = einops.repeat(avg_temps, "h -> (h 7)")
    std_temps = einops.reduce(temps, "(h 7) -> h",  t.std)
    std_temps = einops.repeat(std_temps, "h -> (h 7)")
    ans = (temps-avg_temps)/std_temps
    return ans

expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))

# %%
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    l2norm = matrix.norm(dim = 1, keepdim=True)
    return matrix/l2norm
    
matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)
# %%
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    normalized = normalize_rows(matrix)
    return normalized @ normalized.T

matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)
# %%
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    tens_bool = t.rand(n, 1) > t.cumsum(probs, dim=0)
    return tens_bool.sum(dim=-1)


n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)
# %%
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    predicted_classes = scores.argmax(dim=1)
    right_wrong = (predicted_classes == true_classes).sum()
    return right_wrong/len(true_classes)



scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected
print("Tests passed!")
# %%
def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    return t.gather(prices,0,items).sum()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0
print("Tests passed!")
# %%
def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    maxel = t.max(matrix,dim=-1).values
    exps = t.exp(matrix-einops.rearrange(maxel, "n -> n 1"))
    return maxel + t.log(t.sum(exps,dim=-1))
    

matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)

matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)

# %%
def batched_softmax(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    logsum = einops.rearrange(batched_logsumexp(matrix), "n -> n 1")
    return t.exp(matrix-logsum)


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))  # check it's translation-invariant

matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))

# %%
def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n).

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    logsum = einops.rearrange(batched_logsumexp(matrix), "n -> n 1")
    return matrix-logsum


matrix = t.arange(1, 7).view((2, 3)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 7).view((2, 3)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-2.4076, -1.4076, -0.4076], [-2.4076, -1.4076, -0.4076]])
assert_all_close(actual, expected)

# %%
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    diagonal = np.diag(mat)
    return einops.einsum(diagonal, "i ->")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec, "i j, j -> i")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einops.einsum(mat1,mat2,"i j, j k -> i k")


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return einops.einsum(vec1,vec2, "i, i -> ")


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return einops.einsum(vec1, vec2, "i, j -> i j")


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)

# %%
