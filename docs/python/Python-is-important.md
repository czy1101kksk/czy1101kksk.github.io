# 🔗Python学习笔记
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
!!! info "想说的话"
    <font size = 4><B>人生苦短,我学Pyhton.</B></font>
    
    <font size = 3>

    python着实是我学习实践上的重点,我的培养方案中Python程序设计是必修的一门课,本页结合课内要求和**[CS50p](https://cs50.harvard.edu/python/)**做一个比较完善的个人笔记,用于补充python基础零散知识+应付课程考试
    
    官方文档:[https://docs.python.org/3/library/functions.html](https://docs.python.org/3/library/functions.html)
    </font>

### 一些零碎知识点
---

#### 语句格式
```python
a=1;b=2;c=3         # 用分号在一行中输入多个语句

sum = a + b \       # 使用'\'来续行
    + c
```

#### 科学计数法e
对于很大的浮点数,可用e代替10,其中e后面的数字必须为整数
```python
>>>1.32e-1
0.132
>>>1.32e0
1.32
```

#### 复数 real + imag * j
>   ```python
    >>>type(3+4J)
    complex               # 复数的数据类型
    >>>a = complex(3,4)
    >>>a.real, a.imag     # 复数的实部、虚部为浮点数(float)
    (3.0,4.0)
    ```

#### 进制转换
---

>- 0b,0B 二进制
 - 0x,0X 十六进制
 - 0o,0O 八进制

### Ascii码-Unicode编码-UTF-8编码
---
pta和课本上有相关题目,因此做简单的概念区分

- <B>"Ascii码"</B>:用二进制编码来表示非数值的文字与符号(数字化),如今普遍采用的就是ASCII(美国信息交换标准代码),有7位二进制与八位二进制两种版本(国际通用7位,有$2^7 = 128$个元素),与 Unicode 编码和 UTF-8 编码兼容
> 字符 A:65,a:97,Z:90,z:122

- <B>"Unicode编码"</B>:包含了世界上几乎所有的字符，包括各种语言的文字、符号、表情符号等。它是目前使用最广泛的字符编码方案，可以满足不同语言和文化之间的交流需求。Python3中的字符串是Unicode字符串而不是字节数组.

- <B>"UTF-8"</B>:是 Unicode 编码的一种实现方式,使用1到4个字节来表示不同的字符,特点是对不同范围的字符使用不同长度的编码.对于0x00-0x7f之间的字符,UTF8与ascii编码完全一致.

### print(*objects, sep=' ', end='\n', file=None, flush=False):
---
!!! info ""
    <font size = 3>print函数自带sep=' ',输出多个变量时有空格隔开</font>

### 字符串int
---

>Python3d的字符串缺省编码为Unicode编码

!!! advice "特性"
    - <font size = 4>可拼接</font>:
        ```python
        >>>"abc" + '123'
        'abc123'
        >>>"abc" + 123
        TypeError: can only concatenate str (not "int") to str
        ```
    - <font size = 4>可复制</font>:
        ```python
        >>>'abc' * 5
        'abcabcabcabcabc'
        >>>abc' * 0
        ''     # 空字符
        ```

#### 字符串转换方法
---
<font size = 4>

>方法将字符串转换输出,但没有改变原变量

|方法|功能|
|-|-|
|x.capitalize()|#返回首字母大写其他字母都小写的字符串|
|x.casefold()|#返回所有字母都小写的字符串|
|x.title()|#返回每个单词首字母大写其他字母小写的字符串|
|x.upper() / x.lower()|#将所有字母都转换成大/小写|
|x.swapcase()|#反转字符串大小写|
|x.strip()|#只去除字符串两端的空格|
|x.center(width, [fillchar])|字符串居中对齐的方法,在字符串两侧填充指定的字符|

```python
>>>name = 'zju hello'
>>>name.capitalize()          # 转换为首字母大写的字符串
'Zju hello'             
>>>name.title()               # 每个单词首字母大写,仅识别空格隔开的单词
'Zju Hello'
>>>name.upper()
'ZJU HELLO'
>>>block = '  twist zz '
>>>block.strip()
'twist zz'                     # 仅去除字符串两端空格
>>>sentence = "Python is awesome"
>>>sentence.center(24, '*')
'***Python is awesome****'
```

</font>

### 浮点数相关
```python 
>>>divmod(9.0,2.5)      # 求(商,余数)
(3.0, 1.5)              # 范围tuple类型
>>>round(4.55,1)              # 四舍五入 round( number[, ndigits=None] ) ndigits控制保留到小数点后几位
4 # 而不是3.6, Python 3.x采用“round half to even”,将.5的值舍入到最近的偶数结果，而不是向上舍入  
>>>abs(-44)
44
```

### 布尔值相关




