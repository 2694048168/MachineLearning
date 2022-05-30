#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-9 支持字符串和字节序列的双模式API
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22

标准库中的一些函数能接受字符串或字节序列为参数，然后根据类型展现不同的行为
 re 和 os 模块中就有这样的函数
"""

import os
import re


# ----------------------------
if __name__ == "__main__":
   # 1. 正则表达式中的字符串和字节序列
   re_numbers_str = re.compile(r"\d+")
   re_words_str = re.compile(r"\w+")
   re_numbers_bytes = re.compile(rb"\d+")
   re_words_bytes = re.compile(rb"\w+")

   text_str = ("Ramanujan saw \u0be7\u0bed\u0be8\u0bef" 
               " as 1729 = 1³ + 12³ = 9³ + 10³.")
   text_bytes = text_str.encode('utf_8')

   print(f"Text {repr(text_str)}", sep="\n ")
   print("Numbers")
   print(' str :', re_numbers_str.findall(text_str))
   print(' bytes:', re_numbers_bytes.findall(text_bytes))
   print('Words')
   print(' str :', re_words_str.findall(text_str))
   print(' bytes:', re_words_bytes.findall(text_bytes))

   # 2. os函数中的字符串和字节序列
   print("---------------------")
   print(os.listdir("."))
   print(os.listdir(b"."))