#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 8-3 Python 函数的参数传递 作为引用时
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

# Python 唯一支持的参数传递模式是共享传参 call by sharing
共享传参指函数的各个形式参数获得实参中各个引用的副本,
也就是说，函数内部的形参是实参的别名.
"""

def func(num_1, num_2):
    num_1 += num_2
    return num_1

# 一个简单的类，说明可变默认值的危险
class HauntedBus():
    """备受幽灵乘客折磨的校车"""
    def __init__(self, passengers=[]):
        self.passengers = passengers

    def pick(self, name):
        self.passengers.append(name)
    
    def drop(self, name):
        self.passengers.remove(name)


# ----------------------------
if __name__ == "__main__":
    # Step 1. 函数可能会修改接收到的任何可变对象
    x = 1
    y = 2
    print(func(x, y))
    print(x, y)

    a = [1, 2]
    b = [3, 4]
    print(func(a, b))
    print(a, b)

    t = (10, 20)
    u = (30, 40)
    print(func(t, u))
    print(t, u)

    # 不要使用可变类型作为参数的默认值
    print("-------------------------------")
    bus1 = HauntedBus(['Alice', 'Bill'])
    print(bus1.passengers)
    bus1.pick("Charlie")
    bus1.drop("Alice")
    print(bus1.passengers)
    print("-------------------------------")
    bus2 = HauntedBus()
    bus2.pick("Carrie")
    print(bus2.passengers)
    print("-------------------------------")
    bus3 = HauntedBus()
    print(bus3.passengers)
    bus3.pick("Dave")
    print(bus2.passengers)
    print("-------------------------------")
    # 没有指定初始乘客的 HauntedBus 实例会共享同一个乘客列表
    # 可变默认值导致的这个问题说明了为什么通常使用 None 作为接收可变值的参数的默认值
    print(bus2.passengers is bus3.passengers)
    print(bus1.passengers)