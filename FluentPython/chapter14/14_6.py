#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-6 等差数列生成器
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

内置的 range 函数用于生成有穷整数等差数列 Arithmetic Progression, AP
itertools.count 函数用于生成无穷等差数列

# 构造方法的签名是 ArithmeticProgression(begin, step[, end])
# 使用itertools模块生成等差数列

# 标准库中的生成器函数
https://docs.python.org/3/library/os.html#os.walk

"""

import itertools


class ArithmeticProgression():
    def __init__(self, begin, step, end=None):
        self.begin = begin
        self.step = step
        self.end = end # None ----> 无穷数列

    def __iter__(self):
        # 强制类型转换
        result = type(self.begin + self.step)(self.begin)
        forever = self.end is None
        index = 0
        while forever or result < self.end:
            yield result
            index +=1 
            result = self.begin + self.step * index

def AritprogGen(begin, step, end=None):
    result = type(begin + step)(begin)
    forever = end is None
    index = 0
    while forever or result < end:
        yield result
        index += 1
        result = begin + step * index

def Aritprog_Gen(begin, step, end=None):
    first = type(begin + step)(begin)
    ap_gen = itertools.count(first, step)
    if end is not None:
        ap_gen = itertools.takewhile(lambda n: n < end, ap_gen)
    return ap_gen


# ----------------------------
if __name__ == "__main__":
    ap = ArithmeticProgression(0, 1, 3)
    print(list(ap))

    ap = ArithmeticProgression(1, .5, 3)
    print(list(ap))

    ap = ArithmeticProgression(0, 1/3, 1)
    print(list(ap))

    print("---- generator expression ----")
    ap = AritprogGen(0, 1, 3)
    print(list(ap))

    ap = AritprogGen(1, .5, 3)
    print(list(ap))

    ap = AritprogGen(0, 1/3, 1)
    print(list(ap))

    print("---- itertools generator ----")
    gen = itertools.count(1, .5)
    print(next(gen))
    print(next(gen))
    print(next(gen))
    # itertools.count 函数从不停止，因此，如果调用 list(count())， 
    # Python 会创建一个特别大的列表，超出可用内存，在调用失败之前，电脑会疯狂地运转
    print(next(gen))

    # itertools.takewhile 函数则不同，它会生成一个使用另一个生成器的生成器，
    # 在指定的条件计算结果为 False 时停止
    gen = itertools.takewhile(lambda n: n < 3, itertools.count(1, .5))
    print(list(gen))

    print("---- generator expression ----")
    ap = Aritprog_Gen(0, 1, 3)
    print(list(ap))

    ap = Aritprog_Gen(1, .5, 3)
    print(list(ap))

    ap = Aritprog_Gen(0, 1/3, 1)
    print(list(ap))