#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 14-3 典型的迭代器
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

# 实现典型的迭代器设计模式。注意，这不符合 Python 的习惯做法
# 明确可迭代的集合和迭代器对象之间的关系

# 迭代器可以迭代，但是可迭代的对象不是迭代器
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
        return SentenceIterator(self.words)

class SentenceIterator():
    def __init__(self, words):
        self.words = words
        self.index = 0
    
    def __next__(self):
        try:
            word = self.words[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return word


# ----------------------------
if __name__ == "__main__":
    str_seq = Sentence('"The time has come," the Walrus said,')
    print(str_seq)
    # 序列是可迭代的
    for word in str_seq:
        print(word)