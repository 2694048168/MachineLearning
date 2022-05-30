#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-3 使用生成表达式建立元组和数组
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-17
"""

import array


# ----------------------------
if __name__ == "__main__":
    symbols = '$¢£¥€¤'

    # 生成器表达式的语法跟列表推导差不多，只不过把方括号换成圆括号而已
    # 如果生成器表达式是一个函数调用过程中的唯一参数，那么不需要额外再用括号把它围起来。
    symbol_tuple = tuple(ord(symbol) for symbol in symbols)
    print(f"the tuple generator is: {symbol_tuple}")

    # array 的构造方法需要两个参数，因此括号是必需的。 
    # array 构造方法的第一个参数指定了数组中数字的存储方式
    symbol_array = array.array("I", (ord(symbol) for symbol in symbols))
    print(f"the array generator is: {symbol_array}")

    # 利用生成器表达式实现了一个笛卡儿积
    # 用到生成器表达式之后，内存里不会留下一个有 6 个组合的列表，
    # 因为生成器表达式会在每次 for 循环运行时才生成一个组合。
    # 生成器表达式就可以帮忙省掉运行 for 循环的开销，即一个额外的列表。
    colors = ["black", "white"]
    sizes = ["S", "M", "L"]
    # generate expression 生成表达式避免额外的内存占用
    for tshirt in (f"{c} {s}" for c in colors for s in sizes):
        print(tshirt)