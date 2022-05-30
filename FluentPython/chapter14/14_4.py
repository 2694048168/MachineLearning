#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-4 生成器函数
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

实现相同功能，但却符合 Python 习惯的方式是，用生成器函数代替 SentenceIterator 类

# 生成器函数的工作原理
只要 Python 函数的定义体中有 yield 关键字，该函数就是生成器函数。
调用生成器函数时，会返回一个生成器对象。也就是说，生成器函数是生成器工厂
"""

import re
import reprlib


RE_WORD = re.compile('\w+')

class Sentence():
    def __init__(self, text):
        self.text = text
        # 返回一个字符串列表，里面的元素是正则表达式的全部非重叠匹配
        self.words = RE_WORD.findall(text)
    
    def __repr__(self):
        # 实用函数用于生成大型数据结构的简略字符串表示形式
        # 默认情况下, reprlib.repr 函数生成的字符串最多有 30 个字符
        return f"Sentence({reprlib.repr(self.text)})"

    def __iter__(self):
        for word in self.words:
            # 产出当前的 word
            yield word
        # 这个 return 语句不是必要的；这个函数可以直接“落空”，自动返回。
        # 不管有没有 return 语句，生成器函数都不会抛出 StopIteration 异常，
        # 而是在生成完全部值之后会直接退出, 不用再单独定义一个迭代器类
        return

# 说明生成器的行为
def gen_123():
    yield 1
    yield 2
    yield 3

# 运行时打印消息的生成器函数
def gen_AB():
    print("---- start ----")
    yield "A"
    print("---- continue ----")
    yield "B"
    print("---- end. ----")


# ----------------------------
if __name__ == "__main__":
    str_seq = Sentence('"The time has come," the Walrus said,')
    print(str_seq)
    # 序列是可迭代的
    for word in str_seq:
        print(word)

    print("-------- 生成器函数的工作原理 --------")
    print(gen_123)  # gen_123 是函数对象
    print(gen_123())  # 调用时， gen_123() 返回一个生成器对象
    for idx in gen_123():
        print(idx)

    print("---------------------")
    gen = gen_123()
    # 生成器是迭代器，会生成传给 yield 关键字的表达式的值
    # 调用 next(gen) 会获取 yield 生成的下一个元素
    print(next(gen))
    print(next(gen))
    print(next(gen))
    # print(next(gen)) # Traceback (most recent call last): StopIteration

    print("-------- 生成器函数定义体的执行过程 --------")
    for char in gen_AB():
        print(f"----> {char}")