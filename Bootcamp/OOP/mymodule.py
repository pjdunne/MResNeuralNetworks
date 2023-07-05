"""
An example module
C Paterson  30/9/2013
"""

import numpy as _np

_sqrt5 = _np.sqrt(5.0)
_phi = (1.0+_np.sqrt(5.0)) / 2.0


def fibo1(n):
    "Implement method 1 to calculate a Fibonacci number"
    val = int(_phi**n/_sqrt5 + 0.5)
    return val


def fibo2(n):
    "Implement method 2 to calculate a Fibonacci number"
    val = (_phi**n - (-_phi)**(-n)) / _sqrt5
    return val


def fibo(n):
    "Calculate the nth Fibonacci number"
    return __fibo_impl(n)


__fibo_impl = fibo1

__all__ = ['fibo']
