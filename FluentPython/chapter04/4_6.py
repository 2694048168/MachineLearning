#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-6 大小写折叠
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22

大小写折叠其实就是把所有文本变成小写，再做些其他转换。
这个功能由 str.casefold() 方法, Python 3.3 新增支持
"""

import unicodedata

# ----------------------------
if __name__ == "__main__":
   micro = 'µ'
   print(unicodedata.name(micro))
   micro_cf = micro.casefold()
   print(unicodedata.name(micro_cf))
   print(micro)
   print(micro_cf)

   eszett = 'ß'
   print(unicodedata.name(eszett))
   eszett_cf = eszett.casefold()
   print(eszett)
   print(eszett_cf)