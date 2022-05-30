#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-1 把一个字符串变成 Unicode 码位的列表
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-16
"""

import timeit

TIMES = 10000

SETUP = """
symbols = '$¢£¥€¤'
def non_ascii(c):
    return c > 127
"""

def clock(label, cmd):
    res = timeit.repeat(cmd, setup=SETUP, number=TIMES)
    print(label, *(f'{x:.3f}' for x in res))


# 列表推导同 filter 和 map 的比较 时间效率对比
clock('listcomp        :', '[ord(s) for s in symbols if ord(s) > 127]')
clock('listcomp + func :', '[ord(s) for s in symbols if non_ascii(ord(s))]')
clock('filter + lambda :', 'list(filter(lambda c: c > 127, map(ord, symbols)))')
clock('filter + func   :', 'list(filter(non_ascii, map(ord, symbols)))')


# ----------------------------
if __name__ == "__main__":
    symbols = "$¢£¥€¤"
    codes = []
    for symbol in symbols:
        codes.append(ord(symbol))
    print(f"the Unicode is : {codes}")

    # 列表推导式和可读性
    # 列表推导也可能被滥用,通常的原则是，只用列表推导来创建新的列表，并且尽量保持简短; 否则 for 循环
    codes_list = [ord(symbol) for symbol in symbols]
    print(f"the Unicode is : {codes}")

    # 列表推导、生成器表达式，以及集合(set)推导和字典(dict)推导，
    # 在 Python 3 中都有了自己的局部作用域，就像函数似的。
    # 表达式内部的变量和赋值只在局部起作用，表达式的上下文里的同名变量还可以被正常引用，局部变量并不会影响到它们。

    # 列表推导可以帮助把一个序列或是其他可迭代类型中的元素过滤或是加工，然后再新建一个列表。 
    # Python 内置的 filter 和 map 函数组合起来也能达到这一效果，但是可读性上打了不小的折扣
    beyond_ascii = [ord(s) for s in symbols if ord(s) > 127]
    print(f"the beyond ascii Unicode is : {beyond_ascii}")

    beyond_ascii_lambda = list(filter(lambda c: c > 127, map(ord, symbols)))
    print(f"the beyond ascii Unicode is : {beyond_ascii_lambda}")