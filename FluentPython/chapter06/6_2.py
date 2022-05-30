#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 6-2 "命令" 模式
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23

命令模式是回调机制的面向对象替代品
"""

class MacroCommand():
    """一个执行一组命令的命令"""
    def __init__(self, commands):
        self.commands = list(commands)

    def __call__(self):
        for command in self.commands:
            command()


# ----------------------------
if __name__ == "__main__":
    commands = [len, list, set, int]
    macro_command = MacroCommand(commands)
    print(macro_command)