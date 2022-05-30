#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 10-5 Python duck typing
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-28

# https://en.wikipedia.org/wiki/Fold_(higher-order_function)

# duck typing 鸭子类型
# 解释器会调用 __getattr__ 方法

# 散列和快速等值测试
1. __hash__
2. __eq__

# 格式化
__format__
"""

from array import array
import reprlib
import math
import numbers
import functools
import operator
import itertools


class Vector():
    """A multidimensional ``Vector`` class, take 5
    A ``Vector`` is built from an iterable of numbers::
    """
    typecode = 'd'

    def __init__(self, components):
        self._components = array(self.typecode, components)
    
    def __iter__(self):
        return iter(self._components)

    def __repr__(self):
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return f'Vector({components})'

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) + bytes(self._components))

    def __eq__(self, other):
        return (len(self) == len(other) and all(a == b for a, b in zip(self, other)))

    def __hash__(self):
        hashes = (hash(x) for x in self)
        return functools.reduce(operator.xor, hashes, 0)

    def __abs__(self):
        return math.sqrt(sum(x * x for x in self))

    def __bool__(self):
        return bool(abs(self))

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)

    # 支持序列协议
    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        elif isinstance(index, numbers.Integral):
            return self._components[index]
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))

    # 动态存取属性
    shortcut_names = 'xyzt'

    def __getattr__(self, name):
        cls = type(self)
        if len(name) == 1:
            pos = cls.shortcut_names.find(name)
            if 0 <= pos < len(self._components):
                return self._components[pos]
        msg = '{.__name__!r} object has no attribute {!r}'
        raise AttributeError(msg.format(cls, name))

    def angle(self, n):
        r = math.sqrt(sum(x * x for x in self[n:]))
        a = math.atan2(r, self[n-1])
        if (n == len(self) - 1) and (self[-1] < 0):
            return math.pi * 2 - a
        else:
            return a

    def angles(self):
        return (self.angle(n) for n in range(1, len(self)))

    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('h'): # 超球面坐标
            fmt_spec = fmt_spec[:-1]
            coords = itertools.chain([abs(self)], self.angles())
            outer_fmt = '<{}>'
        else:
            coords = self
            outer_fmt = '({})'
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(', '.join(components))


# ----------------------------
if __name__ == "__main__":
    print("---------------------------")
    print(Vector([3.1, 4.2]))
    print(Vector([3, 4, 5]))
    print(Vector(range(10)))

    print("---------------------------")
    vec_1 = Vector([3, 4])
    x, y = vec_1
    print(x, y)
    print(vec_1)
    vec_1_clone = eval(repr(vec_1))
    print(vec_1 == vec_1_clone)
    print(vec_1)
    print(bytes(vec_1))
    print(abs(vec_1))
    print(bool(vec_1))

    print("---------------------------")
    v1_clone = Vector.frombytes(bytes(vec_1))
    print(v1_clone)
    print(v1_clone == vec_1)

    print("---------------------------")
    vec_2 = Vector([3, 4, 5])
    x, y, z = vec_2
    print(x, y, z)
    print(vec_2)
    vec_2_clone = eval(repr(vec_2))
    print(vec_2 == vec_2_clone)
    print(vec_2)
    print(bytes(vec_2))
    print(abs(vec_2))
    print(bool(vec_2))

    print("---------------------------")
    vec_7 = Vector(range(7))
    print(vec_7)
    print(abs(vec_7))
    print(vec_7[-1])
    print(vec_7[1:4])
    print(vec_7.x)
    print(vec_7.y)
    print(vec_7.z)
    print(vec_7.t)

    print("---------------------------")
    v1 = Vector([3, 4])
    v2 = Vector([3.1, 4.2])
    v3 = Vector([3, 4, 5])
    v6 = Vector(range(6))
    print(hash(v1), hash(v3), hash(v6))