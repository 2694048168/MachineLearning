#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 5-3 用户定义的可调用类型;函数内省
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23
"""

import random


class BingoCage():
    def __init__(self, items):
        # 在本地构建一个副本，防止列表参数的意外副作用
        self._items = list(items)
        random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError("Pick form empty BingoCage.")
    
    def __call__(self):
        # 实现 __call__ 方法的类是创建函数类对象的简便方式，
        # 创建保有内部状态的函数，还有一种截然不同的方式——使用闭包
        return self.pick()

def factorial_func(num):
    """return n!
    Args:
        num (int): 计算整数 num 的阶乘
    Returns:
        int: 计算整数 num 的阶乘
    """
    return 1 if num < 2 else num * (factorial_func(num - 1))

# 列出常规对象没有而函数有的属性
class C():
    pass

def func():
    pass


# ----------------------------
if __name__ == "__main__":
    # 6. 可调用对象 ()
    print("-------- Callable Object --------")
    objects = [abs, str, 13]
    print([callable(obj) for obj in objects])

    # 7. 用户定义的可调用对象
    bingo = BingoCage(range(3))
    print(bingo.pick())
    print(bingo())
    print(callable(bingo))

    # 8. 函数内省
    print("-------- Function Magic Methods --------")
    print(dir(factorial_func))

    # 列出常规对象没有而函数有的属性
    print("-------- Function Special Methods --------")
    obj = C()
    print(sorted(set(dir(func)) - set(dir(obj))))