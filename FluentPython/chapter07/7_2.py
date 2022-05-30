#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-2 变量作用域规则
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24
"""

import dis

global_var = 42

def func_1(local_var):
    print(local_var)
    print(global_var)

def func_2(local_var):
    print(local_var)
    print(global_var) # UnboundLocalError: local variable 'global_var' referenced before assignment
    #  Python 不要求声明变量，但是假定在函数定义体中赋值的变量是局部变量
    global_var  = 99

def func_3(local_var):
    # 如果在函数中赋值时想让解释器把 b 当成全局变量，要使用 global 声明
    global global_var
    print(local_var)
    print(global_var)
    global_var  = 99


# ----------------------------
if __name__ == "__main__":
    print('-------- function 1 --------')
    print(f"global variable is : {global_var}")
    print(func_1(24))
    print('-------- function 2 --------')
    # print(func_2(24))
    print('-------- function 3 --------')
    print(func_3(24))
    print(f"global variable is : {global_var}")

    # 比较字节码 
    # https://docs.python.org/3/library/dis.html
    # dis 模块为反汇编 Python 函数字节码提供了简单的方式
    print('-------- Python disassembly to the bytecode --------')
    dis.dis(func_1)
    print('-------- Python disassembly to the bytecode --------')
    dis.dis(func_2)