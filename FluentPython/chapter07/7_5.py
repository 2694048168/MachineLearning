#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-5 实现一个简单的装饰器, 在每次调用被装饰的函数时计时，
    然后把经过的时间、传入的参数和调用的结果打印出来。
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24

# 这是装饰器的典型行为：
把被装饰的函数替换成新函数，二者接受相同的参数，
而且通常返回被装饰的函数本该返回的值，同时还会做些额外操作。
"""

import time
import functools


# 装饰器一般作为一个包(模块)进行调用
# from 7_5 import clock_decorator
def clock_decorator(func):
    def clocked(*args): # 接受任意个定位参数
        time_start = time.perf_counter()
        result = func(*args) # 自由变量 func
        elapsed = time.perf_counter() - time_start
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print(f"[{elapsed:0.8f}] {name}({arg_str}) -> {result}")
        return result
    return clocked

# 上面实现的 clock 装饰器有几个缺点：
# 不支持关键字参数，而且遮盖了被装饰函数的 __name__ 和 __doc__ 属性
# 使用 functools.wraps 装饰器把相关的属性从 func 复制到 clocked 中。
# 此外，这个新版还能正确处理关键字参数。
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

@clock_decorator
def snooze(seconds):
    time.sleep(seconds)

@clock_decorator
def factorial(num):
    return 1 if num < 2 else num * factorial(num - 1)

@clock
def snooze_2(seconds):
    time.sleep(seconds)

@clock
def factorial_2(num):
    return 1 if num < 2 else num * factorial(num - 1)


# ----------------------------
if __name__ == "__main__":
    print('-------- Calling snooze(.123) --------')
    snooze(.123)

    print('-------- Calling factorial(6) --------')
    print('6! =', factorial(6))

    print('-------- Calling snooze_2(.123) --------')
    snooze_2(.123)

    print('-------- Calling factorial_2(6) --------')
    print('6! =', factorial_2(6))