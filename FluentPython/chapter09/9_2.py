#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 9-2 Python classmethod 与 staticmethod
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

classmethod: 定义操作类，而不是操作实例的方法
classmethod 改变了调用方法的方式，因此类方法的第一个参数是类本身，而不是实例
classmethod 最常见的用途是定义备选构造方法
按照约定，类方法的第一个参数名为 cls(但是 Python 不介意具体怎么命名)

staticmethod 装饰器也会改变方法的调用方式，但是第一个参数不是特殊的值
其实，静态方法就是普通的函数，只是碰巧在类的定义体中，而不是在模块层定义
"""

# 比较 classmethod 和 staticmethod 的行为
class Demo():
    @classmethod
    def klaccmethod(*args):
        return args

    @staticmethod
    def statmeth(*args):
        return args


# ----------------------------
if __name__ == "__main__":
    print("---- classmethod ----")
    print(Demo.klaccmethod())
    print(Demo.klaccmethod("spam"))

    print("---- staticmethod ----")
    print(Demo.statmeth())
    print(Demo.statmeth("spam"))