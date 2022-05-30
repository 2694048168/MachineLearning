#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-7 Python 3.3中新出现的句法： yield from
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

# 可迭代的归约函数
函数都接受一个可迭代的对象，然后返回单个结果。这些函数叫 “归约”函数、“合拢”函数或“累加”函数。
其实，这里列出的每个内置函数都可以使用 functools.reduce 函数实现，内置是因为使用它们便于解决常见的问题。

# 深入分析 iter 函数
iter 函数还有一个鲜为人知的用法：传入两个参数，使用常规的函数或任何可调用的对象创建迭代器。
这样使用时，第一个参数必须是可调用的对象，用于不断调用（没有参数），产出各个值；
第二个值是哨符，这是个标记值，当可调用的对象返回这个值时，触发迭代器抛出 StopIteration 异常，而不产出哨符。
"""

from random import randint


def chain(*iterables):
    for it in iterables:
        for i in it:
            yield i

def chain_new(*iterables):
    for i in iterables:
        yield from i

def d6():
    return randint(1, 6)


# ----------------------------
if __name__ == "__main__":
    str_seq = 'ABC'
    tuple_idx = tuple(range(3))
    print(list(chain(str_seq, tuple_idx)))
    print(list(chain_new(str_seq, tuple_idx)))

    print("--- 深入分析iter函数 ----")
    d6_iter = iter(d6, 1)
    print(d6_iter)
    for roll in d6_iter:
        print(roll)