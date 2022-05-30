#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-2 可迭代的对象与迭代器的对比
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

# 可迭代的对象
使用 iter 内置函数可以获取迭代器的对象。
如果对象实现了能返回迭代器的 __iter__ 方法, 那么对象就是可迭代的
序列都可以迭代；实现了 __getitem__ 方法，而且其参数是从零开始的索引，这种对象也可以迭代

# 可迭代的对象和迭代器之间的关系： Python 从可迭代的对象中获取迭代器

# StopIteration 异常表明迭代器到头了
Python 语言内部会处理 for 循环和其他迭代上下文（如列表推导、元组拆包，等等）中的 StopIteration 异常

# 标准的迭代器接口(在 collections.abc.Iterator 抽象基类中制定)有两个方法:
__next__ : 返回下一个可用的元素，如果没有元素了，抛出 StopIteration 异常
__iter__ : 返回 self,以便在应该使用可迭代对象的地方使用迭代器,例如在 for 循环中

因为迭代器只需 __next__ 和 __iter__ 两个方法，所以除了调用 next() 方法，
以及捕获StopIteration 异常之外，没有办法检查是否还有遗留的元素。
"""


# ----------------------------
if __name__ == "__main__":
    str_seq = "ABC"
    # 字符串 'ABC' 是可迭代的对象。背后是有迭代器的，只不过看不到
    for char in str_seq:
        print(char)

    print("---------------------")
    # 使用可迭代的对象构建迭代器 iterator_str
    iterator_str = iter(str_seq)
    while True:
        try:
            # 在迭代器上调用 next 函数，获取下一个字符
            print(next(iterator_str))
        except StopIteration:
            #  如果没有字符了，迭代器会抛出 StopIteration 异常
            del iterator_str
            break