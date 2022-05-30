#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-6 使用functools.lru_cache做备忘
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24

functools.lru_cache 是非常实用的装饰器, 实现了备忘 memoization 功能。
这是一项优化技术，它把耗时的函数的结果保存起来，避免传入相同的参数时重复计算。
# 可以用于动态规划算法里面, 避免相同的结果重复计算！！！
LRU 三个字母是 Least Recently Used 的缩写，表明缓存不会无限制增长，一段时间不用的缓存条目会被扔掉

# 特别要注意， lru_cache 可以使用两个可选的参数来配置。它的签名是：
functools.lru_cache(maxsize=128, typed=False)
maxsize 参数指定存储多少个调用的结果。缓存满了之后，旧的结果会被扔掉，腾出空间。
为了得到最佳性能， maxsize 应该设为 2 的幂。 
typed 参数如果设为 True, 把不同参数类型得到的结果分开保存,
即把通常认为相等的浮点数和整数参数 (如 1 和 1.0) 区分开。
顺便说一下，因为 lru_cache 使用字典存储结果，而且键根据调用时传入的定位参数和关键字参数创建，
所以被 lru_cache 装饰的函数，它的所有参数都必须是可散列的。
"""

import time
import functools


def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - time_start
        name = func.__name__
        arg_list = []
        if args:
            arg_list.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = [f"{k}={w}" for k, w in sorted(kwargs.items())]
            arg_list.append(', '.join(pairs))
        arg_str = ', '.join(arg_list)
        print(f"[{elapsed:0.8f}] {name}({arg_str}) -> {result}")
        return result
    return clocked

# 生成第 n 个斐波纳契数，递归方式非常耗时
@clock
def fibonacci(num):
    if num < 2:
        return num
    return fibonacci(num - 2) + fibonacci(num - 1)

# lru_cache 可以接受配置参数，
@functools.lru_cache()
@clock
def fibonacci_lru(num):
    if num < 2:
        return num
    return fibonacci_lru(num - 2) + fibonacci_lru(num - 1)


# ----------------------------
if __name__ == "__main__":
    print("-------- fibonacci with recursive --------")
    print(fibonacci(6))
    print("-------- fibonacci with LRU --------")
    print(fibonacci_lru(6))