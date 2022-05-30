#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 5-2 高阶函数
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22

接受函数为参数,或者把函数作为结果返回的函数是高阶函数(higher-order function)
map 函数就是一例，内置函数 sorted 也是：可选的 key 参数用于提供一个函数，它会应用到各个元素上进行排序
"""

import functools
import operator

def reverse_word(word):
    return word[::-1]

def factorial_func(num):
    return 1 if num < 2 else num * (factorial_func(num - 1))

# ----------------------------
if __name__ == "__main__":
    # 1. 根据单词长度给一个列表排序
    fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
    print(fruits)
    # 任何单参数函数都能作为 key 参数的值
    print(sorted(fruits, key=len))

    # 2. 根据反向拼写给一个单词列表排序
    str_seq = "testing"
    print(str_seq)
    print(reverse_word(str_seq))
    print(sorted(fruits, key=reverse_word))

    # 在函数式编程范式中，最为人熟知的高阶函数有 map、 filter、 reduce 和 apply
    # apply 函数在 Python 2.3 中标记为过时，在 Python 3 中移除了，因为不再需要它了
    # 如果想使用不定量的参数调用函数，可以编写 fn(*args, **keywords)，不用再编写 apply(fn, args,kwargs)
    # map、 filter 和 reduce 这三个高阶函数还能见到，不过多数使用场景下都有更好的替代品

    # 3. 计算阶乘列表： map 和 filter 与列表推导比较
    print("-------- High-Order Function --------")
    high_order_func = list(map(factorial_func, range(6)))
    print(high_order_func)
    list_comps = [factorial_func(n) for n in range(6)]
    print(list_comps)

    high_order_func_filter = list(map(factorial_func, filter(lambda n: n % 2, range(6))))
    print(high_order_func_filter)
    list_comps_filter = [factorial_func(n) for n in range(6) if n % 2]
    print(list_comps_filter)
    # Python 3 中， map 和 filter 返回生成器(迭代器)，因此它们的直接替代品是生成器表达式

    # 4. 使用 reduce 和 sum 计算 0-100 之和
    # sum 和 reduce 的通用思想是把某个操作连续应用到序列的元素上，
    # 累计之前的结果，把一系列值归约成一个值。
    print(functools.reduce(operator.add, range(101)))
    print(sum(range(101)))

    # 5. 匿名函数 lambda expression
    # 作为参数传递给高级函数，Python 很少使用匿名函数
    # Lundh 提出的 lambda 表达式重构秘笈
    # https://docs.python.org/3/howto/functional.html
    print("-------- Lambda Expression --------")
    print(fruits)
    print(sorted(fruits, key=lambda word: word[::-1]))

    # 6. 可调用对象 ()
    print("-------- Callable Object --------")
    objects = [abs, str, 13]
    print([callable(obj) for obj in objects])