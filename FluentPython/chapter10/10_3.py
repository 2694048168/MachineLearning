#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 10-3 Python 切片原理, 一例胜千言
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-28

# 能处理切片的 __getitem__ 方法
"""

from array import array
import reprlib
import math
import numbers


class Vector():
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
        return tuple(self) == tuple(other)

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


class MySeq():
    # 了解 __getitem__ 和切片的行为
    def __getitem__(self, index):
        return index


# ----------------------------
if __name__ == "__main__":
    seq = MySeq()
    print(seq[1])
    print(seq[1:4])
    print(seq[1:4:2])
    print(seq[1:4:2, 9])
    print(seq[1:4:2, 7:9])

    print("----------- slice -------------")
    # 交互式 (ipython) 控制台是个有价值的工具，能发现新事物
    print(slice)
    print(dir(slice))
    print("----------- slice help -------------")
    # https://docs.python.org/3/reference/datamodel.html?highlight=indices#slice.indices
    help(slice.indices)

    # 测试改进的 Vector.__getitem__ 方法
    print("----------- __getitem__ -------------")
    vec_3 = Vector(range(7))
    print(vec_3[-1])
    print(vec_3[1:4])
    print(vec_3[-1:])
    # print(vec_3[1, 2])