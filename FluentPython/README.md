# 流畅 Python; Fluent Python

## Chapter01 Python 数据模型

- Python 风格 (Pythonic); import this (the Zen of Python)
- https://www.python.org/doc/humor/#the-zen-of-python
- 特殊方法(dunder method 双下方法 or magic method 魔术方法)名字以两个下划线开头，以两个下划线结尾 \_\_getitem\_\_ 
- 特殊方法的存在是为了被 Python 解释器调用的
- 特殊方法的调用通常是隐式的
- 通过内置函数 (len, str, iter 等)来使用特殊方法是最好的选择
- Python 语言参考手册中 ["Data Model"](https://docs.python.org/3/reference/datamodel.html)
- https://stackoverflow.com/users/95810/alex-martelli

## Chapter02 序列构成的数组

- 内置序列类型：容器序列 and 扁平序列; 可变序列 and 不可变序列
- 列表推导式 list comprehension(listcomps): 构建列表的方法
- 生成器表达式 generator expression(genexps): 生成各种类型的元素并用它们来填充序列
- 元组，不可变列表和没有字段的记录；元组拆包
- Python 序列(list, tuple, str) 的切片操作
- 在切片和区间操作里不包含区间范围的最后一个元素是 Python 的风格，这个习惯符合 Python、 C 和其他语言里以 0 作为起始下标的传统
- 增量赋值不是一个原子操作
- 如果一个函数或者方法对对象进行的是就地改动，那它就应该返回 None，好让调用者知道传入的参数发生了变动，而且并未产生新的对象
- 在某些情况下可以替换列表的数据类型(数组, 队列, 双端队列, ...)
- memoryview 是一个内置类，它能让用户在不复制内容的情况下操作同一个数组的不同切片
- Numpy and SicPy 数值处理和计算高效库
- 双向队列和其他形式的队列
- [Jupyter](https://jupyter.org/install)

## Chapter03 字典和集合

- dict 类型 Python 语言的基石
- 模块的命名空间、实例的属性和函数的关键字参数中都可以看到字典的身影
- 跟它有关的内置函数都在 \_\_builtins\_\_.\_\_dict\_\_ 模块中
- Python 对字典的实现做了高度优化，而散列表则是字典类型性能出众的根本原因
- 集合 set 的实现其实也依赖于散列表
- 集合的本质是许多唯一对象的聚集。因此，集合可以用于去重
- 字典推导式 dict comprehension(dictcomps)
- 集合推导式 set comprehension(setcomps)
- 散列表的工作原理: hash(); 散列值与相等性; 散列算法与散列冲突
- 散列表带来的潜在影响 (什么样的数据类型可作为键、不可预知的顺序，等等)

## Chapter04 文本和字节序列

- 人类使用文本，计算机使用字节序列
- Python 3 明确区分了人类可读的文本字符串和原始的字节序列
- Unicode 字符串、二进制序列，以及在二者之间转换时使用的编码
- 把码位转换成字节序列(字符的具体表述)的过程是编码；把字节序列转换成码位(字符的标识)的过程是解码
- Python 自带了超过 100 种编解码器（codec, encoder/decoder），用于在文本和字节之间相互转换
- 处理文本的最佳实践是 "Unicode 三明治"
- Unicode 领域就变得相当复杂:文本规范化(即为了比较而把文本转换成统一的表述)和排序
- unicode.normalize and 大小写折叠; Unicode 文本排序
- 支持字符串和字节序列的双模式API

## Chapter05 一等函数

- 把函数视作对象
- 高阶函数和函数式编程
- 匿名函数; 可调用对象
- 定位参数; 关键字参数; 获取参数信息
- 函数注解
- 支持函数式编程的 Python 包

## Chapter06 使用一等函数实现设计模式

- 虽然设计模式与语言无关，但这并不意味着每一个模式都能在每一门语言中使用
- 重构 “策略” 模式
- “命令” 模式

## Chapter07 函数装饰和闭包

- 函数装饰器用于在源码中 "标记" 函数，以某种方式增强函数的行为
- 函数装饰器; 闭包; nonlocal 保留关键字
- 闭包还是回调式异步编程和函数式编程风格
- 解释清楚函数装饰器的工作原理, 包括最简单的注册装饰器和较复杂的参数化装饰器
- functools 标准库中的装饰器: functools.wraps, functools.lru_cache, functools.singledispatch
- 标准库中的装饰器: property、 classmethod 和 staticmethod

## Chapter08 对象引用, 可变性和垃圾回收

- Python 变量; 引用式变量; 对象和对象的名称
- 解决 Python 程序中许多不易察觉的 bug 的关键
- Python 唯一支持的参数传递模式是共享传参 call by sharing
- 可变默认值导致的这个问题说明了为什么通常使用 None 作为接收可变值的参数的默认值
- Python 对象的 del 和垃圾回收？弱引用和引用计数
- https://docs.python.org/3/reference/datamodel.html

## Chapter09 符合 Python 风格的对象

- 得益于 Python 数据模型，自定义类型的行为可以像内置类型那样自然
- Python 鸭子类型(duck typing)：只需按照预定行为实现对象所需的方法即可
- 自己定义类，而且让类的行为跟真正的 Python 对象一样
- 利用 \_\_slots\_\_ 节省内存
- 如何以及何时使用 @classmethod 和 @staticmethod 装饰器
- Python 的私有属性和受保护属性的用法、约定和局限
- 格式说明符/格式规范微语言

## Chapter10 序列的修改、散列和切片

- 在 Python 中创建功能完善的序列类型无需使用继承，只需实现符合序列协议的方法
- 在面向对象编程中，协议是非正式的接口，只在文档中定义，在代码中不定义
- 把协议当作正式接口; 鸭子类型 duck typing
- 协议和鸭子类型之间的关系，以及对自定义类型的实际影响
- [向量空间模型和余弦相似性](https://en.wikipedia.org/wiki/Vector_space_model)
- zip 函数的名字取自拉链系结物 zipper fastener; zip and unpacking
- [Fold Higher-Order Function](https://en.wikipedia.org/wiki/Fold_(higher-order_function))

## Chapter11 接口: 从协议到抽象基类

-  Python 风格的角度探讨接口
- 从鸭子类型的代表特征动态协议
- 接口更明确、能验证实现是否符合规定的抽象基类 Abstract Base Class， ABC
- 序列协议是 Python 最基础的协议之一。即便对象只实现了那个协议最基本的一部分，解释器也会负责任地处理

## Chapter12 继承的优缺点

- 继承的初衷是让新手顺利使用只有专家才能设计出来的框架
- 子类化内置类型的缺点
- 多重继承和方法解析顺序 Method Resolution Order,MRO

## Chapter13 正确重载运算符

- 运算符重载的作用是让用户定义的对象使用中缀运算符
- Python 如何处理中缀运算符中不同类型的操作数
- 使用鸭子类型或显式类型检查处理不同类型的操作数
- 中缀运算符如何表明自己无法处理操作数

## Chapter14 可迭代的对象、迭代器和生成器

- 迭代是数据处理的基石,扫描内存中放不下的数据集时，要找到一种惰性获取数据项的方式，即按需一次获取一个数据项。这就是迭代器模式 Iterator pattern
- 语言内部使用 iter(...) 内置函数处理可迭代对象的方式
- 如何使用 Python 实现经典的迭代器模式; 详细说明生成器函数的工作原理
- 如何使用生成器函数或生成器表达式代替经典的迭代器; 如何使用标准库中通用的生成器函数
- 如何使用 yield from 语句合并生成器

## Chapter15 上下文管理器和 else 块

- with 语句和上下文管理器
- for、 while 和 try 语句的 else 子句

## Chapter16 协程

- 从根本上把 yield 视作控制流程的方式，这样就好理解协程
- 生成器作为协程使用时的行为和状态; 使用装饰器自动预激协程
- 调用方如何使用生成器对象的 .close() 和 .throw(...) 方法控制协程; 协程终止时如何返回值
- yield from 新句法的用途和语义

## Chapter17 使用 future 处理并发

-  future 指一种对象，表示异步执行的操作
- [future](https://pypi.org/project/futures/)
- [TQDM](https://github.com/tqdm/tqdm#usage)

## Chapter18 使用 asyncio 包处理并发

- [asyncio](https://pypi.python.org/pypi/asyncio)
- 对比一个简单的多线程程序和对应的 asyncio 版，说明多线程和异步任务之间的关系
- asyncio.Future 类与 concurrent.futures.Future 类之间的区别
- 摒弃线程或进程，如何使用异步编程管理网络应用中的高并发
- 在异步编程中，与回调相比，协程显著提升性能的方式
- 如何把阻塞的操作交给线程池处理，从而避免阻塞事件循环

## Chapter19 动态属性和特性

- Python 中，数据的属性和处理数据的方法统称属性 attribute, 其实，方法只是可调用的属性
- Python 还可以创建特性 property, 在不改变类接口的前提下,使用存取方法(即读值方法和设值方法)修改数据属性

## Chapter20 属性描述符

- 描述符是对多个属性运用相同存取逻辑的一种方式
- 描述符是实现了特定协议的类，这个协议包括 \_\_get\_\_、\_\_set\_\_ 和 \_\_delete\_\_ 方法。property 类实现了完整的描述符协议

## Chapter21 类元编程

- 类元编程是指在运行时创建或定制类的技艺
- 类装饰器能使用更简单的方式解决很多问题
- Python 是给法定成年人使用的语言