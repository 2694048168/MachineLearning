#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 6-1 重构 策略 设计模式
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23

# 使用函数实现“策略”模式
每个具体策略都是一个类，而且都只定义了一个方法，即 discount。
此外策略实例没有状态（没有实例属性）
它们看起来像是普通的函数——的确如此,
重构，把具体策略换成了简单的函数，而且去掉了抽象类

# 找出模块中的全部策略
在 Python 中，模块也是一等对象，而且标准库提供了几个处理模块的函数
Python 文档是这样说明内置函数 globals 的
globals() 返回一个字典，表示当前的全局符号表
这个符号表始终针对当前模块（对函数或方法来说，是指定义它们的模块，而不是调用它们的模块）
"""

from abc import ABC, abstractmethod
from collections import namedtuple
import inspect


Customer = namedtuple('Customer', 'name fidelity')

class LineItem():
    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price
    
    def total(self):
        return self.price * self.quantity

class Order():
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
            # ----------------------------------
            # 重构 策略 设计模式
            # discount = self.promotion(self)
            # ----------------------------------
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())

class Order_refactoring():
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            # discount = self.promotion.discount(self)
            # ----------------------------------
            # 重构 策略 设计模式
            discount = self.promotion(self)
            # ----------------------------------
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())

class Promotion(ABC): # 策略：抽象基类
    @abstractmethod
    def discount(self, order):
        """返回折扣金额（正值）"""

class FidelityPromo(Promotion): # 第一个具体策略
    """为积分为1000或以上的顾客提供5%折扣"""
    def discount(self, order):
        return order.total() * .05 if order.customer.fidelity >= 1000 else 0

class BulkItemPromo(Promotion): # 第二个具体策略
    """单个商品为20个或以上时提供10%折扣"""
    def discount(self, order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * .1
        return discount

class LargeOrderPromo(Promotion): # 第三个具体策略
    """订单中的不同商品达到10个或以上时提供7%折扣"""
    def discount(self, order):
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * .07
        return 0

# ----------------------------------
# 重构 策略 设计模式
def fidelity_promo(order):
    """为积分为1000或以上的顾客提供5%折扣"""
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

def bulk_item_promo(order):
    """单个商品为20个或以上时提供10%折扣"""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
    return discount

def large_order_promo(order):
    """订单中的不同商品达到10个或以上时提供7%折扣"""
    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * .07
    return 0
# ----------------------------------
def best_promo(order):
    """选择可用的最佳折扣"""
    return max(promo(order) for promo in promos)


# ----------------------------
if __name__ == "__main__":
    # 使用不同促销折扣的 Order 类示例
    print("-------------------------")
    joe = Customer("John Doe", 0)
    ann = Customer("Ann Smith", 1100)
    cart = [LineItem("banana", 4, .5),
            LineItem("apple", 10, 1.5),
            LineItem("watermellon", 5, 5.0)]
    print(Order(joe, cart, FidelityPromo()))

    banana_cart = [LineItem('banana', 30, .5), 
                   LineItem('apple', 10, 1.5)]
    print(Order(joe, banana_cart, BulkItemPromo()) )

    long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]
    print(Order(joe, long_order, LargeOrderPromo()))
    print(Order(joe, cart, LargeOrderPromo()))

    print("-------------------------")
    # 重构 策略 设计模式
    print(Order_refactoring(joe, cart, fidelity_promo))
    print(Order_refactoring(joe, banana_cart, bulk_item_promo))
    print(Order_refactoring(joe, long_order, large_order_promo))
    print(Order_refactoring(joe, cart, large_order_promo))

    print("-------------------------")
    # 找出模块中的全部策略 globals
    promos = [globals()[name] for name in globals() if name.endswith('_promo') and name != 'best_promo']

    # 内省单独的 promotions 模块，构建 promos 列表
    promos = [func for name, func in inspect.getmembers(inspect.promotions, inspect.isfunction)]