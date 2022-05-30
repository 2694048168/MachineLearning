#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 8-4 del 和垃圾回收
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

# 对象绝不会自行销毁；然而，无法得到对象时，可能会被当作垃圾回收

在 CPython 中，垃圾回收使用的主要算法是引用计数。
实际上，每个对象都会统计有多少引用指向自己。当引用计数归零时，对象立即就被销毁
"""

import weakref

def bye():
    print("Gone with the wind...")


# ----------------------------
if __name__ == "__main__":
    # 没有指向对象的引用时，监视对象生命结束时的情形
    set_seq_1 = {1, 2, 3}
    set_seq_2 = set_seq_1

    ender = weakref.finalize(set_seq_1, bye)
    print(ender.alive)
    del set_seq_1
    print(ender.alive)

    set_seq_2 = 'spam'
    print(ender.alive)