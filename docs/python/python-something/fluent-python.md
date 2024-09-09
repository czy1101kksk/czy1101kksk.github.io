# 🛣 《流畅的Python》
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>


### ```__getitem__``` 与 ```__len__```
---

!!! info "魔法方法"
    <font size = 3.5>
    魔法方法(magic method)是Python中具有双下划线开头和结尾的特殊方法，因此被称为“双下划线方法”。
    </font>

- ```__getitem__```方法：允许我们使用方括号[]表示法来定义访问自定义类的元素的方法，类似于我们在列表、字典所做的操作:

    ```element = MyInstance[index]```

- ```__len__```方法：用于返回对象的长度，当我们使用内置的len()函数对一个对象进行长度判断时，实际上是调用了该对象的```__len__()```方法

<B>使用上述方法实现一组纸牌类（Card）</B>

```python
import collections 

Card = collections.namedtuple('Card', ['rank', 'suit'])
# 创建具有命名字段的 tuple 子类的 factory 函数 (具名元组)

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]
    
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck() #实例化
```

```__getitem__``` 与 ```__len__```实现了```FrenchDeck```类的迭代、切片、len()等操作。实现 ```__len__``` 和 ```__getitem__``` 两个特殊方法后，FrenchDeck 的行为就像标准的 Python 序列一样

```python
>>> deck[:3]
>>> [Card(rank='2', suit='spades'), Card(rank='3', suit='spades'),
Card(rank='4', suit='spades')]

>>> random.choice(deck)
>>> Card(rank='K', suit='spades')

>>> len(deck)
>>> 52

for card in deck:
    print(card)
>>>Card(rank='2', suit='spades')
Card(rank='3', suit='spades')
Card(rank='4', suit='spades')
...

Card('7', 'beasts') in deck
>>>False
```


### ```__repr__```、```__abs__```、```__add__``` 和 ```__mul__```

实现一个二维向量类，即：
```python
>>> v1 = Vector(2, 4)
>>> v2 = Vector(2, 1)
>>> v1 + v2
Vector(4, 5)
```
并且考虑向量的加法、乘法、标量积、绝对值特性

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'Vector({self.x!r}, {self.y!r})'
        # 使用 !r 以标准的表示形式显示属性

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
        
```

特殊方法 ```__repr__ ```供内置函数repr调用，获取对象的<B>字符串表示形式</B>。如未定义 ```__repr__``` 方法，```Vector``` 实例在 Python 控制台中显示为 ```<Vector object at 0x10e100070>``` 形式。

### ```__bool__```

默认情况下，用户定义类的实例都是真值，除非实现了 ```__bool__``` 或 ```__len__``` 方法。简单来说，```bool(x)``` 调用 ```x.__bool__()```，以后者返回的结果为准。如果没有实现 ```__bool__``` 方法，则 Python 尝试调用 ```x.__len__()```；如果该方法返回零值，则 ```bool``` 函数返回 ```False```，否则返回 ```True```。

```python
# Vector.__bool__方法

def __bool__(self):
    return bool(self. or self.y)
```

### 序列类型

- 容器序列：存放所包含对象的引用，对象可以是任何类型；

- 扁平序列：在自己的内存空间中存储所含内容的值