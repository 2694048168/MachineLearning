#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-1 字符的编码和解码
    把码位转换成字节序列(字符的具体表述)的过程是编码；
    把字节序列转换成码位(字符的标识)的过程是解码
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21
"""


# ----------------------------
if __name__ == "__main__":
    string_sequence = "café"
    print(len(string_sequence))
    print(type(string_sequence))
    print(string_sequence)

    # convert 'str' to 'bytes': encode 字符的编码过程
    print("-------- Encode Process --------")
    bytes_sequence = string_sequence.encode('utf8')
    print(len(bytes_sequence))
    print(type(bytes_sequence))
    print(bytes_sequence)

    # convert 'bytes' to 'str': decode 字符的解码过程
    print("-------- Decode Process --------")
    str_sequence = bytes_sequence.decode('utf8')
    print(len(str_sequence))
    print(type(str_sequence))
    print(str_sequence)