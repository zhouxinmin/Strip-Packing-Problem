# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/4/26 9:29
# @Site    : 
# @File    : tutorial.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm

import numpy
from theano import *
import theano.tensor as T
from theano import function
from theano import pp
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

# x = T.dscalar('x')
# y = T.dscalar('y')
# z = x + y
# f = function([x, y], z)
# print f(2, 3)
# print numpy.allclose(f(16.3, 12.1), 28.4)
# print pp(z)

# x = T.dmatrix('x')
# y = T.dmatrix('y')
# z = x + y
# f = function([x, y], z)
# print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

# a = theano.tensor.vector()
# out = a + a ** 10
# f = theano.function([a], out)
# print f([0, 1, 2])

# x = T.dmatrix('x')
# s = 1/(1+T.exp(-x))
# logistic = theano.function([x], s)
# print logistic([[0, 1], [-1, -2]])

# a, b = T.dmatrices('a', 'b')
# diff = a - b
# abs_diff = abs(diff)
# diff_squared = diff**2
# f = theano.function([a, b], [diff, abs_diff, diff_squared])
# print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

# x, y = T.dscalars('x', 'y')
# z = x + y
# f = function([x, In(y, value=1)], z)
# print f(33)
# print f(33, 2)

# state = shared(0)
# inc = T.iscalar('inc')
# accumulator = function([inc], state, updates=[(state, state + inc)])
# print state.get_value()
# print accumulator(1)
# print state.get_value()
# print accumulator(300)
# print state.get_value()
# state.set_value(-1)
# print accumulator(3)
# print state.get_value()
#
# fn_of_state = state * 2 + inc
# foo = T.scalar(dtype=state.dtype)
# skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
# print skip_shared(1, 3)
# print state.get_value()

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    # Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
f_val0 = f()
f_val1 = f()
g_val0 = g()
g_val1 = g()
print f_val0, f_val1
print g_val0, g_val1