#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-7 Python + and * operator
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-17
"""

import dis

# ----------------------------
if __name__ == "__main__":
    # Python 程序员会默认序列是支持 + 和 * 操作的
    # + 和 * 都遵循这个规律，不修改原有的操作对象，而是构建一个全新的序列
    list_sequence = [1, 2, 3]
    str_1 = "Wei"
    str_2 = "Li"
    print(list_sequence * 5)
    print(str_1 + str_2)
    print(5 * "weili")

    # 使用 * 初始化一个列表组成的列表
    board = [['_'] * 3 for i in range(3)]
    print(type(board))
    print(board)
    board[1][2] = 'X'
    print(board, "\n")

    # 含有 3 个指向同一对象的引用的列表是毫无用处的, 与想要的逻辑不一致
    # 详细理解 Python 中引用和可变对象背后的原理和陷阱
    weird_board = [['_'] * 3] * 3
    print(weird_board)
    weird_board[1][2] = 'O'
    print(weird_board, "\n")

    # 此种写法跟上述是一样的，与想要的逻辑不一致
    row = ['_'] * 3
    board  = []
    for i in range(3):
        board.append(row)
    print(board)
    board[1][2] = 'X'
    print(board, "\n")

    # 应该修定为下面
    board = []
    for i in range(3):
        row = ['_'] * 3
        board.append(row)
    print(board)
    board[1][2] = 'X'
    print(board, "\n")

    # 序列的增量赋值
    # += 特殊方法 __iadd__ (就地加法) 否则 __add__
    # *= 特殊方法 __imul__ (就地加法) 否则 __mul__
    list_sequence_variable = [1, 2, 3]
    print(f"the value of sequence at: {list_sequence_variable}")
    print(f"the logical address of sequence at: {id(list_sequence_variable)}")
    list_sequence_variable *= 2
    print(f"the value of sequence at: {list_sequence_variable}")
    print(f"the logical address of sequence at: {id(list_sequence_variable)}\n")

    # 对不可变序列进行重复拼接操作的话，效率会很低，因为每次都有一个新对象，
    # 而解释器需要把原来对象中的元素先复制到新的对象里，然后再追加新的元素
    tuple_sequence_const = (1, 2, 3)
    print(f"the value of sequence at: {tuple_sequence_const}")
    print(f"the logical address of sequence at: {id(tuple_sequence_const)}")
    tuple_sequence_const *= 2
    print(f"the value of sequence at: {tuple_sequence_const}")
    print(f"the logical address of sequence at: {id(tuple_sequence_const)}")

    # 一个关于 += 的谜题
    # 增量赋值不是一个原子操作
    # https://pythontutor.com/
    tuple_list = (1, 2, [30, 40])
    print(tuple_list)
    # tuple_list[2] += [50, 60]
    tuple_list[2].extend([50, 60])
    print(tuple_list)

    # 查看 Python 源代码执行背后的字节码
    # https://docs.python.org/3/library/dis.html
    # https://docs.python.org/3/glossary.html#term-bytecode
    print("\n-----------------------------------------------")
    print(dis.dis("s[a] += b"))