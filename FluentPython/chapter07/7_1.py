#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-1 装饰器基础知识
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23

装饰器是可调用的对象,其参数是另一个函数 (被装饰的函数)
装饰器可能会处理被装饰的函数,然后把它返回,或者将其替换成另一个函数或可调用对象

# 严格来说，装饰器只是语法糖
装饰器可以像常规的可调用对象那样调用，其参数是另一个函数
有时，这样做更方便，尤其是做元编程(在运行时改变程序的行为)时
综上，装饰器的一大特性是，能把被装饰的函数替换成其他函数
第二个特性是，装饰器在加载模块时立即执行

函数装饰器在导入模块时立即执行，而被装饰的函数只在明确调用时运行。
这突出了 Python 程序员所说的导入时和运行时之间的区别

# 使用装饰器改进 "策略" 模式
"""

def decorator_callable_function(func):
    def inner():
        print('running inner()')
    return inner

@decorator_callable_function
def target_dec():
    print('running target()')

def target():
    print('running target()')

registry = []

def register(func):
    print(f'running register({func})')
    registry.append(func)
    return func

@register
def func_1():
    print('running func_1()')

@register
def func_2():
    print('running func_2()')

@register
def func_3():
    print('running func_3()')

def main():
    print('running main()')
    print(f'registry -> {registry}')
    func_1()
    func_2()
    func_3()


# ----------------------------
if __name__ == "__main__":
    # 1. 装饰器通常把函数替换成另一个函数
    print(target())
    print(target_dec())

    # 审查对象，发现 target 现在是 inner 的引用
    print(decorator_callable_function(target()))

    # 2. Python 何时执行装饰器
    main()