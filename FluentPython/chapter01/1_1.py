#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 1-1 一摞有序的纸牌
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-14
"""

import collections
import random

# namedtuple 用于构建只有少数属性但是没有方法的对象
Card = collections.namedtuple("Card", ["rank", "suit"])


# class FrenchDeck(object):
class FrenchDeck():
    """虽然 FrenchDeck 隐式地继承了 object 类，但功能却不是继承而来的。通过数据模型和一些合成来实现这些功能。
    通过实现 __len__ 和 __getitem__ 这两个特殊方法， FrenchDeck 就跟一个 Python 自有的序列数据类型一样，
    可以体现出 Python 的核心语言特性（例如迭代和切片）。
    同时这个类还可以用于标准库中诸如 random.choice、 reversed 和 sorted 这些函数。
    另外，对合成的运用使得 __len__ 和 __getitem__ 的具体实现可以代理给 self._cards 这个 Python 列表（即 list 对象）。

    如何洗牌, 按照目前的设计， FrenchDeck 是不能洗牌的，
    因为这摞牌是不可变的 (immutable) : 卡牌和它们的位置都是固定的,
    除非我们破坏这个类的封装性，直接对 _cards 进行操作。
    
    其实只需要一行代码来实现 __setitem__ 方法，洗牌功能就不是问题了。

    Returns:
        _type_: _description_
    """
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = "spades diamonds clubs hearts".split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]


# 按照常规，用点数来判定扑克牌的大小， 2 最小、 A 最大；
# 同时还要加上对花色的判定，黑桃最大、红桃次之、方块再次、梅花最小。
# 下面就是按照这个规则来给扑克牌排序的函数，梅花 2 的大小是 0，黑桃 A 是 51
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value  * len(suit_values) + suit_values[card.suit]


# ----------------------------
if __name__ == "__main__":
    beer_card = Card("7", "diamonds")
    print(beer_card)

    # 标准 Python 集合类型一致，用 len() function 查看有多少张牌
    # 这是由 __len__ 方法实现的
    deck = FrenchDeck()
    print(f"the total number of Deck is {len(deck)}")

    # 简单容易的可以从中抽取特定的一张纸牌
    # 这是由 __gititem__ method 实现的
    print("the first item of Deck is : {}".format(deck[0]))
    print("the last item of Deck is : {}".format(deck[-1]))

    # 随机抽取一张纸牌，使用 Python 内置函数
    # Python 已经内置了从一个序列中随机选出一个元素的函数 
    # random.choice，直接把它用在这一摞纸牌实例上就好
    random_item_1 = random.choice(deck)
    random_item_2 = random.choice(deck)
    print("An random element from Deck is : {}".format(random_item_1))
    print("An random element from Deck is : {}".format(random_item_2))

    # 因为 __getitem__ 方法把 [] 操作交给了 self._cards 列表，
    # 所以 deck 类自动支持切片 slicing 操作。
    # 查看一叠纸牌最上面的 3 张
    print(f"the top three decks of Deck is :\n{deck[:3]}")
    # 只看牌面是 A 的切片操作
    print(f"the A deck of Deck is :\n{deck[12::13]}")

    # 仅仅实现了 __getitem__ 方法，这一摞牌就变成可迭代的了
    for card_iter in deck:
        print(card_iter)
    # 反向迭代
    for card_iter in reversed(deck):
        print(card_iter)

    # 迭代通常是隐式的，譬如说一个集合类型没有实现 __contains__ 方法，
    # 那么 in 运算符就会按顺序做一次迭代搜索。
    # 于是， in 运算符可以用在 FrenchDeck 类上，因为它是可迭代的
    print(Card("Q", "hearts") in deck)
    print(Card("7", "bearts") in deck)

    # 排序 默认是升序 (从小到大)
    for card in sorted(deck, key=spades_high, reverse=False):
        print(card)
    # 设置为降序 (从大到小)
    for card in sorted(deck, key=spades_high, reverse=True):
        print(card)