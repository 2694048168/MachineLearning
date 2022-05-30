#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-6 Python 序列的切片操作
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-17
"""


# ----------------------------
if __name__ == "__main__":
    # 切片和区间会忽略最后一个元素 Why
    # 1. 快速看出切片和区间里有几个元素
    # 2. 快速计算出切片和区间的长度，用后一个数减去第一个下标(stop - tart)即可
    # 3. 利用任意一个下标来把序列分割成不重叠的两部分
    list_sequence = list(range(12))
    print(list_sequence[:3])
    print(list_sequence[:4])
    print(list_sequence[4:])

    # 对对象进行切片
    # 用 sequence[a:b:c] 的形式对 sequence 在 a 和 b 之间以 c 为间隔取值, c的值还可以为负，负值意味着反向取值
    # 只能作为索引或者下标用在 [] 中来返回一个切片对象： slice(a, b, c)
    # 对 seq[start:stop:step] 进行求值的时候，Python 会调用 seq.__getitem__(slice(start, stop, step))
    str_sequence = 'bicycle'
    print(type(str_sequence[::3]))
    print(str_sequence[::3])
    print(str_sequence[::-1])
    print(str_sequence[::-2])
    print(str_sequence[::2])

    invoice = """
        0.....6................................40........52...55........
        1909 Pimoroni PiBrella          $17.50 3 $52.50
        1489 6mm Tactile Switch x20     $4.95  2 $9.90
        1510 Panavise Jr. - PV-201      $28.00 1 $28.00
        1601 PiTFT Mini Kit 320x240     $34.95 1 $34.95
        """
    SKU = slice(0, 6)
    DESCRIPTION = slice(6, 40)
    UNIT_PRICE = slice(40, 52)
    QUANTITY = slice(52, 55)
    ITEM_TOTAL = slice(55, None)
    line_items = invoice.split("\n")[2:]
    for item in line_items:
        print(item[UNIT_PRICE], item[DESCRIPTION])

    # 多维切片和省略
    # [] 运算符里还可以使用以逗号分开的多个索引或者是切片，
    # 外部库 NumPy 里就用到了这个特性，二维的 numpy.ndarray 就可以用 a[i, j] 这种形式来获取，
    # 抑或是用 a[m:n, k:l] 的方式来得到二维切片。
    # 要正确处理这种 [] 运算符的话，对象的特殊方法 __getitem__ 和 __setitem__ 需要以元组的形式来接收 a[i, j] 中的索引。
    # 也就是说，如果要得到 a[i, j] 的值， Python 会调用 a.__getitem__((i, j))。
    # Python 内置的序列类型都是一维的，因此它们只支持单一的索引，成对出现的索引是没有用的。
    # ---------------------------------------------------------------------------------
    # 省略 (ellipsis) 的正确书写方法是三个英语句号（...），而不是 Unicdoe 码位 U+2026 表示的半个省略号（...）。
    # 省略在 Python 解析器眼里是一个符号，而实际上它是 Ellipsis 对象的别名，而 Ellipsis 对象又是 ellipsis 类的单一实例。 
    # 它可以当作切片规范的一部分，也可以用在函数的参数清单中，比如 f(a, ..., z)，或 a[i:...]。
    # 在 NumPy 中， ... 用作多维数组切片的快捷方式。如果 x 是四维数组，那么 x[i, ...] 就是 x[i, :, :, :] 的缩写。
    # 除了用来提取序列里的内容，切片还可以用来就地修改可变序列，也就是说修改的时候不需要重新组建序列

    # 给切片赋值
    print(list_sequence)
    list_sequence[2:5] = [20, 30]
    print(list_sequence)
    del list_sequence[5:7]
    print(list_sequence)
    list_sequence[3::2] = [11, 22, 33]
    print(list_sequence)
    list_sequence[2:5] = [100]
    print(list_sequence)