#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-5 为了正确比较而规范化 Unicode 字符串
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22
"""

import unicodedata

# ----------------------------
if __name__ == "__main__":
   # 因为 Unicode 有组合字符（变音符号和附加到前一个字符上的记号，
   # 打印时作为一个整体），所以字符串比较起来很复杂
   str_seq_1 = 'café'
   str_seq_2 = 'cafe\u0301'
   print(str_seq_1, str_seq_2)
   print(len(str_seq_1), len(str_seq_2))
   # 在 Unicode 标准中， 'é' 和 'e\u0301' 这样的序列叫“标准等价物”（canonical equivalent），
   # 应用程序应该把它们视作相同的字符。
   # 但是， Python 看到的是不同的码位序列，因此判定二者不相等
   print(str_seq_1 == str_seq_2)

   # 解决方案是使用 unicodedata.normalize 函数提供的 Unicode 规范化
   # 这个函数的第一个参数是这 4 个字符串中的一个： 'NFC'、 'NFD'、 'NFKC' 和 'NFKD'
   print('-------- unicodedata.normalize solution --------')
   print(len(unicodedata.normalize('NFC', str_seq_1)), len(unicodedata.normalize('NFC', str_seq_1)))
   print(len(unicodedata.normalize('NFD', str_seq_1)), len(unicodedata.normalize('NFD', str_seq_1)))
   print(unicodedata.normalize('NFC', str_seq_1) == unicodedata.normalize('NFC', str_seq_2))
   print(unicodedata.normalize('NFD', str_seq_1) == unicodedata.normalize('NFD', str_seq_2))

   # 西方键盘通常能输出组合字符，因此用户输入的文本默认是 NFC 形式。
   # 不过，安全起见，保存文本之前，最好使用 normalize('NFC', user_text) 清洗字符串。 
   # NFC 也是 W3C的 “Character Model for the World Wide Web: String Matching and Searching”规范
   # https://www.w3.org/TR/charmod-norm/ 推荐的规范化形式。

   # 使用 NFC 时，有些单字符会被规范成另一个单字符。
   # 例如，电阻的单位欧姆（Ω）会被规范成希腊字母大写的欧米加。
   # 这两个字符在视觉上是一样的，但是比较时并不相等，因此要规范化，防止出现意外
   ohm = '\u2126'
   print(unicodedata.name(ohm))
   ohm_c = unicodedata.normalize('NFC', ohm)
   print(unicodedata.name(ohm_c))
   print(ohm_c == ohm)

   half = '½'
   print(unicodedata.normalize('NFKC', half))

   four_squared = '4²'
   print(unicodedata.normalize('NFKC', four_squared))

   micro = 'µ'
   micro_kc = unicodedata.normalize("NFKC", micro)
   print(micro, micro_kc)
   print(ord(micro), ord(micro_kc))
   print(unicodedata.name(micro), unicodedata.name(micro_kc))