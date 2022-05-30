#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 15-1 上下文管理器
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

with 语句会设置一个临时的上下文，交给上下文管理器对象控制，并且负责清理上下文。
这么做能避免错误并减少样板代码，因此 API 更安全,而且更易于使用。除了自动关闭文件之外,with 块还有很多用途

# 反直觉逻辑的 else 子句的行为如下：
1. for : 仅当 for 循环运行完毕时（即 for 循环没有被 break 语句中止）才运行 else 块。
2. while : 仅当 while 循环因为条件为假值而退出时（即 while 循环没有被 break 语句中止）才运行 else 块
3. try : 仅当 try 块中没有异常抛出时才运行 else 块
https://docs.python.org/3/reference/compound_stmts.html
还指出: “else 子句抛出的异常不会由前面的 except 子句处理。”

在所有情况下，如果异常或者 return、 break 或 continue 语句导致控制权跳到了复合语句的主块之外， else 子句也会被跳过。

# 上下文管理器和 with 块
with 语句的目的是简化 try/finally 模式。
这种模式用于保证一段代码运行完毕后执行某项操作，即便那段代码由于异常、 return 语句或 sys.exit() 调用而中止，也会执行指定的操作。 finally 子句中的代码通常用于释放重要的资源，或者还原临时变更的状态。

上下文管理器协议包含 __enter__ 和 __exit__ 两个方法。 
with 语句开始运行时，会在上下文管理器对象上调用 __enter__ 方法。 
with 语句运行结束后，会在上下文管理器对象上调用 __exit__ 方法，以此扮演 finally 子句的角色。

# contextlib 模块中的实用工具
https://docs.python.org/3/library/contextlib.html
"""

class LookingGlass():
    def __enter__(self):
        import sys
        self.original_write = sys.stdout.write
        sys.stdout.write = self.reverse_write
        return 'JABBERWOCKY'

    def reverse_write(self, text):
        self.original_write(text[::-1])

    def __exit__(self, exc_type, exc_value, trackback):
        import sys
        sys.stdout.write = self.original_write
        if exc_type is ZeroDivisionError:
            print("Please DO NOT divide by zero!")
            return True


# ----------------------------
if __name__ == "__main__":
    # 1. 把文件对象当成上下文管理器使用
    with open("15_1.py", encoding='utf_8') as fp:
        src = fp.read(60)
    
    print(len(src))
    print(fp)
    print(fp.closed, fp.encoding)
    # print(fp.read(60)) # ValueError: I/O operation on closed file.

    #  LookingGlass 上下文管理器类
    print("----------------------------")
    with LookingGlass() as what:
        print('Alice, Kitty and Snowdrop')
        print(what)

    print(what)
    print('Back to normal.')