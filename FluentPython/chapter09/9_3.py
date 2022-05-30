#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 9-3 Python 格式化显示
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

内置的 format() 函数和 str.format() 方法各个类型的格式化方式委托给相应的.__format__(format_spec) 方法
format_spec 是格式说明符，它是：
- format(my_obj, format_spec) 的第二个参数，或者
- str.format() 方法的格式字符串， {} 里代换字段中冒号后面的部分

# https://docs.python.org/3/library/string.html#formatspec
格式说明符/格式规范微语言
格式规范微语言是可扩展的，因为各个类可以自行决定如何解释 format_spec 参数。
"""

from datetime import datetime


# ----------------------------
if __name__ == "__main__":
    brl = 1 / 2.43
    print(brl)
    print(format(brl, "0.4f"))
    print('1 BRL = {rate:0.2f} USD'.format(rate=brl))

    print("---- Format Specification Mini-Language ----")
    print(format(42, 'b'))
    print(format(2/3, '.1%'))

    print("---- Format Specification Mini-Language ----")
    now_time = datetime.now()
    print(format(now_time, '%H:%M:%S'))
    print("It is now {:%I:%M %p}".format(now_time))
    print(f"It is now {now_time:%I:%M %p}")
    print(f"It is now {now_time:%H:%M:%S}")