#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-5 生成器函数 惰性实现 and 生成器表达式
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

# 人们认为惰性是好的特质，至少在编程语言和 API 中是如此
惰性实现是指尽可能延后生成值。这样做能节省内存，而且或许还可以避免做无用的处理

设计 Iterator 接口时考虑到了惰性： next(my_iterator) 一次生成一个元素
懒惰的反义词是急迫,其实惰性求值 lazy evaluation 和及早求值 eager evaluation 是编程语言理论方面的技术术语

生成器表达式可以理解为列表推导的惰性版本：不会迫切地构建列表，而是返回一个生成器，按需惰性生成元素。
也就是说，如果列表推导是制造列表的工厂，那么生成器表达式就是制造生成器的工厂。
"""

import re
import reprlib


RE_WORD = re.compile('\w+')

class Sentence():
    def __init__(self, text):
        self.text = text
    
    def __repr__(self):
        return f"Sentence({reprlib.repr(self.text)})"

    def __iter__(self):
        for match in RE_WORD.finditer(self.text):
            yield match.group()

class SentenceGenExpression():
    def __init__(self, text):
        self.text = text
    
    def __repr__(self):
        return f"Sentence({reprlib.repr(self.text)})"

    def __iter__(self):
        return (match.group() for match in RE_WORD.finditer(self.text))

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

    # 先在列表推导中使用 gen_AB 生成器函数，然后在生成器表达式中使用
    print("------------------------")
    res_1 = [x*3 for x in gen_AB()]
    print(res_1)
    for idx in res_1:
        print(f"----> {idx}")

    print("------------------------")
    res2 = (x*3 for x in gen_AB())
    print(res2)
    for idx in res2:
        print(f"----> {idx}")

    print("---- generator expression ----")
    str_seq = SentenceGenExpression('"The time has come," the Walrus said,')
    print(str_seq)
    # 序列是可迭代的
    for word in str_seq:
        print(word)