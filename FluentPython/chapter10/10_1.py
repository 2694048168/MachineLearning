#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 10-1 Python Vector类: 用户定义的序列类型
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
    

# ----------------------------
if __name__ == "__main__":
    print(Vector([3.1, 4.2]))
    print("-------------------")
    print(Vector(range(10)))