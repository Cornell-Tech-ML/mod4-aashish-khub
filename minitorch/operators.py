"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:


def mul(x: float, y: float) -> float:
    """Multiplies two numbers"""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negates a number"""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another"""
    return 1.0 if x <= y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    if y > x:
        return y
    return x


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value"""
    epsilon = 1e-2
    return abs(y - x) <= epsilon


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    if x > 0:
        return x
    else:
        return 0.0


def log(x: float) -> float:
    """Returns the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Returns the exponential function"""
    return math.exp(x)


def log_back(x: float, a: float) -> float:
    """Computes the derivative of log times a second arg. Assuming second arg is constant this is d/dx (a log x) = a/x"""
    return a / x


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x


def inv_back(x: float, a: float) -> float:
    """Computes the derivative of reciprocal times a second arg. Assuming second arg is constant this is d/dx (a/x) = -ax^-2"""
    return -a * x**-2


def relu_back(x: float, a: float) -> float:
    """Computes the derivative of ReLU times a second arg. Assuming second arg is constant this is either 0 or d/dx (ax) = a"""
    if x > 0:
        return a
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions


def map(f: Callable[[float], float], x: Iterable[float]) -> Iterable[float]:
    """Applies a function to each element of a list"""
    return [f(i) for i in x]


def zipWith(
    f: Callable[[float, float], float], x: Iterable[float], y: Iterable[float]
) -> Iterable[float]:
    """Applies a function to pairs of elements from two lists"""
    return [f(i, j) for i, j in zip(x, y)]


def reduce(
    f: Callable[[float, float], float], x: Iterable[float], init: float
) -> float:
    """Combines elements of a list using a binary function"""
    for i in x:
        init = f(init, i)
    return init


# Use these to implement


def negList(x: Iterable[float]) -> Iterable[float]:
    """Negates each element of a list"""
    return map(neg, x)


def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Adds two lists elementwise"""
    return zipWith(add, x, y)


def sum(x: Iterable[float]) -> float:
    """Sums the elements of a list"""
    return reduce(add, x, 0)


def prod(x: Iterable[float]) -> float:
    """Multiplies the elements of a list"""
    return reduce(mul, x, 1)
