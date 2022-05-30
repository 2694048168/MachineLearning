#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 1-2 一个数学和物理意义上的向量 Vector 类的实现, 利用特殊方法实现
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-14
"""

import math

class Vector():
    # 特殊方法：https://docs.python.org/3/reference/datamodel.html
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        """Python 有一个内置的函数叫 repr,它能把一个对象用字符串的形式表达出来以便辨认,这就是“字符串表示形式”。 
        repr 就是通过 __repr__ 这个特殊方法来得到一个对象的字符串表示形式的。
        如果没有实现 __repr__, 当在控制台里打印一个向量的实例时,
        得到的字符串可能会是 <Vector object at 0x10e100070>。

        # format string in Python
        # 1. % style-way
        # 2. str.format style-way
        # 3. f.string style-way (since python 3.8)

        __repr__ 和 __str__ 的区别在于，后者是在 str() 函数被使用，
        或是在用 print 函数打印一个对象的时候才被调用的，并且它返回的字符串对终端用户更友好。
        如果你只想实现这两个特殊方法中的一个， __repr__ 是更好的选择，
        因为如果一个对象没有 __str__ 函数，
        而 Python 又需要调用它的时候，解释器会用 __repr__ 作为替代。

        # https://stackoverflow.com/questions/1436703/what-is-the-difference-between-str-and-repr
        """
        return "Vector(%r, %r)" % (self.x, self.y)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        """尽管 Python 里有 bool 类型，但实际上任何对象都可以用于需要布尔值的上下文中
        比如if 或 while 语句，或者 and、 or 和 not 运算符。
        为了判定一个值 x 为真还是为假， Python 会调用 bool(x)，这个函数只能返回 True 或者 False。
        
        默认情况下，我们自己定义的类的实例总被认为是真的，除非这个类对 __bool__ 或者 __len__ 函数有自己的实现。 
        bool(x) 的背后是调用 x.__bool__() 的结果；如果不存在 __bool__ 方法，那么 bool(x) 会尝试调用 x.__len__()。
        若返回 0, 则 bool 会返回 False; 否则返回 True。

        对 __bool__ 的实现很简单，如果一个向量的模是 0, 那么就返回 False, 其他情况则返回 True。
        因为 __bool__ 函数的返回类型应该是布尔型，所以我们通过 bool(abs(self)) 把模值变成了布尔值。

        在 Python 标准库的文档中, 有一节叫作 “Built-in Types” 
        # https://docs.python.org/3/library/stdtypes.html#truth
        其中规定了真值检验的标准。通过实现 __bool__, 定义的对象就可以与这个标准保持一致。
        
        如果想让 Vector.__bool__ 更高效，可以采用这种实现：
        def __bool__(self):
            return bool(self.x or self.y)
        它不那么易读，却能省掉从 abs 到 __abs__ 到平方再到平方根这些中间步骤。
        通过 bool 把返回类型显式转换为布尔值是为了符合 __bool__ 对返回值的规定，
        因为 or 运算符可能会返回 x 或者 y 本身的值：若 x 的值等价于真,则or 返回 x 的值；否则返回 y 的值。
        """
        # return bool(abs(self))
        return bool(self.x or self.y)

    def __add__(self, other):
        """通过 __add__ 和 __mul__, 示例 1-2 为向量类带来了 + 和 * 这两个算术运算符。
        值得注意的是,这两个方法的返回值都是新创建的向量对象,
        被操作的两个向量 (self 或 other) 还是原封不动,代码里只是读取了它们的值而已。
        中缀运算符的基本原则就是不改变操作对象，而是产出一个新的值。
        """
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        """示例 1-2 只实现了数字做乘数、向量做被乘数的运算，
        乘法的交换律则被忽略了。在第 13 章里，将利用 __rmul__ 解决这个问题。

        当交换两个操作数的位置时, 就会调用反向运算符 (b * a 而不是 a * b)。
        增量赋值运算符则是一种把中缀运算符变成赋值运算的捷径 (a = a * b 就变成了 a *= b)。
        第 13 章会对这两者作出详细解释。
        """
        return Vector(self.x * scalar, self.y * scalar)


# ----------------------------
if __name__ == "__main__":
    vector_1 = Vector(2, 4)
    vector_2 = Vector(2, 1)
    # 向量的加法
    print(vector_1 + vector_2)

    vector_3 = Vector(3, 4)
    # 计算向量的模
    print(abs(vector_3))

    # 向量的标量乘法
    print(vector_3 * 3)
    print(abs(vector_3 * 3))

    # 查看 Python 解释器的实现标准 CPython, JPython, ...
    from platform import python_implementation
    print(python_implementation())