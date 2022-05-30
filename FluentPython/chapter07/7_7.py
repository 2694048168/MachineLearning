#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-7 单分派泛函数
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24

# PEP 443 Single-dispatch generic functions
https://peps.python.org/pep-0443/

Python 3.4 新增的 functools.singledispatch 装饰器可以把整体方案拆分成多个模块，
甚至可以为你无法修改的类提供专门的函数。
使用 @singledispatch 装饰的普通函数会变成泛函数 generic function:
根据第一个参数的类型，以不同方式执行相同操作的一组函数(多态？)

装饰器是函数，因此可以组合起来使用（即，可以在已经被装饰的函数上应用装饰器
# 把 @d1 和 @d2 两个装饰器按顺序应用到 f 函数上，作用相当于 f = d1(d2(f))

# 参数化装饰器
解析源码中的装饰器时， Python 把被装饰的函数作为第一个参数传给装饰器函数。
那怎么让装饰器接受其他参数呢？答案是：创建一个装饰器工厂函数，把参数传给它，返回一个装饰器，
然后再把它应用到要装饰的函数上。不明白什么意思？

# 装饰器最好通过实现 __call__ 方法的类实现，不应该像示例那样通过函数实现。
# 同意使用建议的方式实现非平凡的装饰器更好，但是使用函数解说这个语言特性的基本思想更易于理解
"""

import time


# 一个参数化的注册装饰器
registry = set()

def register(active=True):
    def decorate(func):
        print(f'running register(active={active}->decorate({func})')
        if active:
            registry.add(func)
        else:
            registry.discard(func)
        return func
    return decorate

@register(active=False)
def func_1():
    print('running func_1()')

@register()
def func_2():
    print('running func_2()')

def func_3():
    print('running func_3()')

# 参数化 clock 装饰器
DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'

def clock(fmt=DEFAULT_FMT):
    def decorator(func):
        def clocked(*args): # 接受任意个定位参数
            time_start = time.perf_counter()
            result = func(*args) # 自由变量 func
            elapsed = time.perf_counter() - time_start
            name = func.__name__
            arg_str = ', '.join(repr(arg) for arg in args)
            print(f"[{elapsed:0.8f}] {name}({arg_str}) -> {result}")
            return result
        return clocked
    return decorator


# ----------------------------
if __name__ == "__main__":
    print(registry)
    print(func_3())
    print(func_2())
    print(func_1())

    print("----------------------")
    @clock()
    def snooze(seconds):
        time.sleep(seconds)

    for i in range(3):
        snooze(.123)