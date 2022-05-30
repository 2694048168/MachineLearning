#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 9-5 可散列的 Vector2d and Python 的私有属性和 “受保护的” 属性
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

把 Vector2d 实例变成可散列的，必须使用 __hash__ 方法
还需要 __eq__ 方法，还要让向量不可变

如果以 __mood 的形式（两个前导下划线，尾部没有或最多有一个下划线）命名实例属性， 
Python 会把属性名存入实例的 __dict__ 属性中，而且会在前面加上一个下划线和类名。
因此，对 Dog 类来说， __mood 会变成 _Dog__mood
Python 这个语言特性叫名称改写(name mangling)

# 约定使用一个下划线前缀编写“受保护”的属性 如 self._x
批评使用两个下划线这种改写机制的人认为，应该使用命名约定来避免意外覆盖属性
"""

from array import array
import math


class Vector2d():
    type_code = 'd'

    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

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

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    @classmethod
    def from_bytes(cls, octal_binary):
        type_code = chr(octal_binary[0])
        memory_view = memoryview(octal_binary[1:]).cast(type_code)
        return cls(*memory_view)

    def angle(self):
        return math.atan2(self.y, self.x)

    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = '<{}, {}>'
        else:
            coords = self
            outer_fmt = '<{}, {}>'
        
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)


# ----------------------------
if __name__ == "__main__":
    print("----------------------")
    v1 = Vector2d(3, 4)
    print(hash(v1))
    print(set([v1]))

    print("----------------------")
    v1 = Vector2d(3, 4)
    print(v1.__dict__)