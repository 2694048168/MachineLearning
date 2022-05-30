#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 9-6 使用__slots__类属性节省空间
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

一个特殊的属性（不是方法），它会影响对象的内部存储，对内存用量可能也有重大影响

默认情况下， Python 在各个实例中名为 __dict__ 的字典里存储实例属性。
为了使用底层的散列表提升访问速度，字典会消耗大量内存。
如果要处理数百万个属性不多的实例，通过 __slots__ 类属性，
能节省大量内存，方法是让解释器在元组中存储实例属性，而不用字典

继承自超类的 __slots__ 属性没有效果
# Python 只会使用各个类中定义的 __slots__ 属性

定义 __slots__ 的方式是，创建一个类属性，使用 __slots__ 这个名字，
并把它的值设为一个字符串构成的可迭代对象，其中各个元素表示各个实例属性

# __slots__ 的问题
1. 每个子类都要定义 __slots__ 属性，因为解释器会忽略继承的 __slots__ 属性
2. 实例只能拥有 __slots__ 中列出的属性，除非把 '__dict__' 加入 __slots__ 中（这样做就失去了节省内存的功效）
3. 如果不把 '__weakref__' 加入 __slots__,实例就不能作为弱引用的目标

# 覆盖类属性
Python 有个很独特的特性：类属性可用于为实例属性提供默认值。 
"""

from array import array
import math


class Vector2d():
    # 在类中定义 __slots__ 属性的目的是告诉解释器：
    # “这个类中的所有实例属性都在这儿了！”
    # 这样， Python 会在各个实例中使用类似元组的结构存储实例变量，
    # 从而避免使用消耗内存的 __dict__ 属性。如果有数百万个实例同时活动，这样做能节省大量内存。
    __slots__ = ("__x", "__y")
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
    print(v1.__dict__)