#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 9-1 Python 对象表示形式
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

每门面向对象的语言至少都有一种获取对象的字符串表示形式的标准方式
# Python 提供了两种方式：
1. repr(): 以便于开发者理解的方式返回对象的字符串表示形式。
2. str(): 以便于用户理解的方式返回对象的字符串表示形式。
实现 __repr__ 和 __str__ 特殊方法，为 repr() 和 str() 提供支持

__bytes__ 方法与 __str__ 方法类似： bytes() 函数调用它获取对象的字节序列表示形式。
而 __format__ 方法会被内置的 format() 函数和 str.format() 方法调用，
使用特殊的格式代码显示对象的字符串表示形式。

在 Python 3 中， __repr__、 __str__ 和 __format__ 都必须返回 Unicode 字符串(str 类型)
只有 __bytes__ 方法应该返回字节序列(bytes 类型)
"""

from array import array
import math


class Vector2d():
    # type_code 类属性，在 Vector2d 实例和字节序列之间转换时使用
    type_code = 'd'

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.type_code)]) + bytes(array(self.type_code, self)))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    # 备选构造方法: 从字节序列转换成 Vector2d 实例
    # 类方法使用 classmethod 装饰器修饰
    @classmethod
    def from_bytes(cls, octal_binary):
        type_code = chr(octal_binary[0])
        memory_view = memoryview(octal_binary[1:]).cast(type_code)
        return cls(*memory_view)


# ----------------------------
if __name__ == "__main__":
    vector_1 = Vector2d(3, 4)
    print(vector_1)
    print(vector_1.x, vector_1.y)

    x, y = vector_1
    print(x, y)

    vector_1_clone = eval(repr(vector_1))
    print(vector_1 == vector_1_clone)

    octets_vector = bytes(vector_1)
    print(octets_vector)

    print(abs(vector_1))
    print(bool(vector_1))
    print(bool(Vector2d(0, 0)))

    print("-------- from bytes to str --------")
    print(Vector2d.from_bytes(octets_vector))