#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-7 规范化文本匹配实用函数(工具箱 utils)
   极端“规范化”： 去掉变音符号
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22

# https://github.com/Alir3z4/python-sanitize
sanitize.py 中的函数做的事情超出了标准的规范化，
而且会对文本做进一步处理，很有可能会改变原意。
只有知道目标语言、目标用户群和转换后的用途，才能确定要不要做这么深入的规范化
"""

import unicodedata
import string

# Utility functions for normalized Unicode string comparison.
def nfc_equal(str_1, str_2):
   return unicodedata.normalize("NFC", str_1) == unicodedata.normalize("NFC", str_2)

def fold_equal(str_1, str_2):
   return unicodedata.normalize("NFC", str_1).casefold() == unicodedata.normalize("NFC", str_2).casefold()

# 去掉所有变音符号
def shave_marks(txt):
   norm_txt = unicodedata.normalize("NFD", txt)
   shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
   return unicodedata.normalize("NFC", shaved)

# 删除拉丁字母中组合记号的函数
# 把拉丁基字符中所有的变音符号删除
def shave_marks_latin(txt):
   norm_txt = unicodedata.normalize("NFD", txt)
   latin_base = False
   keepers = []
   for c in norm_txt:
      if unicodedata.combining(c) and latin_base:
         # 忽略拉丁基字符上的变音符号
         continue
      keepers.append(c)

      # 如果不是组合字符，那就是新的基字符
      if not unicodedata.combining(c):
         latin_base = c in string.ascii_letters

   shaved = ''.join(keepers)
   return unicodedata.normalize('NFC', shaved)


# ----------------------------
if __name__ == "__main__":
   print('-------- Using Normal Form C, case sensitive --------')
   str_seq_1 = 'café'
   str_seq_2 = 'cafe\u0301'
   print(str_seq_1 == str_seq_2)
   print(nfc_equal(str_seq_1, str_seq_2))
   print(nfc_equal("A", "a"))

   print('-------- Using Normal Form C with case folding --------')
   str_seq_3 = 'Straße'
   str_seq_4 = 'strasse'
   print(str_seq_3 == str_seq_4)
   print(nfc_equal(str_seq_3, str_seq_4))
   print(fold_equal(str_seq_3, str_seq_4))
   print(fold_equal(str_seq_1, str_seq_2))
   print(fold_equal("A", "a"))

   # 去掉所有变音符号
   order = ' “Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
   print(order)
   print(shave_marks(order), '\n')

   Greek = 'Zέφupoς, Zéfiro'
   print(Greek)
   print(shave_marks(Greek))