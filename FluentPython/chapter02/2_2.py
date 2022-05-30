#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-2 使用列表推导计算笛卡尔积
# 列表推导来计算笛卡儿积：两个或以上的列表中的元素对构成元组，这些元组构成的列表就是笛卡儿积。
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-17
"""


# ----------------------------
if __name__ == "__main__":
    colors = ["black", "white"]
    sizes = ["S", "M", "L"]
    t_shirts = [(color, size) for color in colors for size in sizes]
    print(f"the cartesian product is : \n{t_shirts}")

    for color in colors:
        for size in sizes:
            print((color, size))
    
    # for 从句的顺序
    # 列表推导的作用只有一个：生成列表。如果想生成其他类型的序列，生成器表达式就派上了用场
    tshirts = [(color, size) for color in colors 
                              for size in sizes]
    print(f"the cartesian product is : \n{tshirts}")