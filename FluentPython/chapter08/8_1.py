#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 8-1 Python 变量并不是存储数据的盒子(引用式变量)
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24
"""

# 创建对象之后才会把变量分配给对象
class Gizmo():
    def __init__(self):
        print(f'Gizmo id: {id(self)}')


# ----------------------------
if __name__ == "__main__":
    # Python 变量式引用式变量
    list_sequence = [1, 2, 3]
    var_reference = list_sequence
    list_sequence.append(4)
    print(var_reference)

    print("-----------------------")
    object_instance = Gizmo()
    # obj = Gizmo() * 10
    # 为了理解 Python 中的赋值语句，应该始终先读右边
    # 对象在右边创建或获取，在此之后左边的变量才会绑定到对象上，这就像为对象贴上标注
    # 因为变量只不过是标注，所以无法阻止为对象贴上多个标注。贴的多个标注，就是别名
    print(dir())

    print("-----------------------")
    # 标识、 相等性和别名
    charles = {'name': "Charles L. Dodgson", 'born': 1832}
    lewis = charles
    print(lewis is charles)
    print(id(lewis))
    print(id(charles))
    print(id(lewis) == id(charles))

    lewis['balance'] = 950
    print(charles)

    print("-----------------------")
    # alex 与 charles 比较的结果是相等，但 alex 不是 charles
    # https://docs.python.org/3/reference/datamodel.html#objects-values-and-types
    # == 运算符比较两个对象的值(对象中保存的数据), 而 is 比较对象的标识
    alex = {'name': "Charles L. Dodgson", 'born': 1832, 'balance': 950}
    print(alex == charles)
    print(alex is charles)
    print(alex is not charles)