#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-5 不可变映射类型
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

标准库里所有的映射类型都是可变的，但有时候会有这样的需求，
比如不能让用户错误地修改某个映射。嵌入式 GPIO 的封装
Pingo.io 里有个映射的名字叫作 board.pins, 里面的数据是 GPIO 物理针脚的信息，
当然不希望用户一个疏忽就把这些信息给改了。
因为硬件方面的东西是不会受软件影响的，所以如果把这个映射里的信息改了，就跟物理上的元件对不上号了。

从 Python 3.3 开始， types 模块中引入了一个封装类名叫 MappingProxyType
如果给这个类一个映射，它会返回一个只读的映射视图。
虽然是个只读视图，但是它是动态的。
这意味着如果对原映射做出了改动，通过这个视图可以观察到，但是无法通过这个视图对原映射做出修改
"""

from types import MappingProxyType

# ----------------------------
if __name__ == "__main__":
    dict_GPIO = {1: "A"}
    dict_GPIO_proxy = MappingProxyType(dict_GPIO)
    print(dict_GPIO == dict_GPIO_proxy)
    print(dict_GPIO_proxy)
    print(dict_GPIO_proxy[1])
    
    # dict_GPIO_proxy[2] = 'x'
    dict_GPIO[2] = 'x'
    print(dict_GPIO_proxy)
    print(dict_GPIO)