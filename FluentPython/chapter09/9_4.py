#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 9-4 Python 格式化显示之自定义
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

实现自己的微语言来解决这个问题。
首先，假设用户提供的格式说明符是用于格式化向量中各个浮点数分量的

# 为自定义的格式代码选择字母时，避免使用其他类型用过的字母
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

    # # 自定义格式说明
    # def __format__(self, fmt_spec=''):
    #     components = (format(c, fmt_spec) for c in self)
    #     return '({}, {})'.format(*components)

    # 极坐标
    def angle(self):
        return math.atan2(self.y, self.x)

    # 自定义格式说明 —— 增强版本
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
    v1 = Vector2d(3, 4)
    print(format(v1))
    print(format(v1, '.2f'))
    print(format(v1, '.3e'))

    print("----------------------")
    print(format(Vector2d(1, 1), 'p'))
    print(format(Vector2d(1, 1), '.3ep'))
    print(format(Vector2d(1, 1), '0.5fp'))