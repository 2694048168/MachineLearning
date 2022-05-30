#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-1 在 Python 社区中,大多数时候都把迭代器和生成器视作同一概念
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

在 Python 3 中，生成器有广泛的用途，即使是内置的 range() 函数也返回一个类似生成器的对象，
而以前则返回完整的列表。如果一定要让 range() 函数返回列表，那么必须明确指明 list(range(100))
在 Python 中，所有集合都可以迭代。在 Python 语言内部，迭代器用于支持：
• for 循环
• 构建和扩展集合类型
• 逐行遍历文本文件
• 列表推导、字典推导和集合推导
• 元组拆包
• 调用函数时，使用 * 拆包实参

# 序列可以迭代的原因 iter 函数
解释器需要迭代对象 x 时，会自动调用 iter(x)
内置的 iter 函数有以下作用
(1) 检查对象是否实现了 __iter__ 方法，如果实现了就调用它，获取一个迭代器
(2) 如果没有实现 __iter__ 方法，但是实现了 __getitem__ 方法， Python 会创建一个迭代器，尝试按顺序（从索引 0 开始）获取元素
(3) 如果尝试失败， Python 抛出 TypeError 异常,通常会提示“C object is not iterable”(C对象不可迭代)，其中 C 是目标对象所属的类
"""

import re
import reprlib
from collections import abc


RE_WORD = re.compile('\w+')

class Sentence():
    def __init__(self, text):
        self.text = text
        # 返回一个字符串列表，里面的元素是正则表达式的全部非重叠匹配
        self.words = RE_WORD.findall(text)
    
    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        # 实用函数用于生成大型数据结构的简略字符串表示形式
        # 默认情况下, reprlib.repr 函数生成的字符串最多有 30 个字符
        return f"Sentence({reprlib.repr(self.text)})"

class Foo():
    def __iter__(self):
        pass


# ----------------------------
if __name__ == "__main__":
    str_seq = Sentence('"The time has come," the Walrus said,')
    print(str_seq)
    # 序列是可迭代的
    for word in str_seq:
        print(word)

    print(str_seq[0])
    print(str_seq[5])

    print("-------- __iter__ ---------")
    print(issubclass(Foo, abc.Iterable))
    foo = Foo()
    print(isinstance(foo, abc.Iterable))