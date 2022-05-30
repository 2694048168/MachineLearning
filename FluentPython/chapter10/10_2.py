#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 10-2 Python Vector类: 可切片的序列(支持序列协议)
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-28
"""

from array import array
import reprlib
import math


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
        return self._components[index]


# ----------------------------
if __name__ == "__main__":
    print(Vector([3.1, 4.2]))
    print("-------------------")
    print(Vector(range(10)))
    print("-------------------")

    vec_1 = Vector([3, 4, 5])
    print(len(vec_1))
    print(vec_1[0])
    print(vec_1[-1])

    vec_2 = Vector(range(7))
    print(vec_2)
    print(vec_2[1:4])