# 🛣[Deep Learning]Stanford CS224n:Natural Language Processing  
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

!!! info "想说的话🎇"
    <font size = 3.5>
    初次接触深度学习是在大一寒假准备打数模比赛的时候,断断续续也就学了一小段时间,先尝试了李沐的[Dive into deep learing](https://zh-v2.d2l.ai)后面在确定本科科研方向后决定看cs224n和Hugging face的[NLPcourse](https://huggingface.co/learn/nlp-course/chapter1/1)掌握一些基础的内容,因此专门写下来用作复习(大佬轻喷😥)
    
    🔝课程网站：http://web.stanford.edu/class/cs224n/index.html
    
    👀一些资源: 
    https://www.bilibili.com/video/BV1jt421L7ui/(B站双语精翻)
    https://www.showmeai.tech/tutorials/36?articleId=231(课件翻译+知识梳理)

    </font>

![](img/cs224n.png)

## 🥗Lecture 1 
---
<font size = 4>

!!! note "Representing words as discrete symbols - ont-hot vectors"
    <font size = 4>
    The tradional NLP use <B>one-hot vectors</B> to represent words as discrete symbol,

    ```python
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder()

    X = [['Male', 1], ['Female', 3], ['Female', 2]]

    encoder.fit(X)
    encoded_data = encoder.transform(X)

    dense_array = encoded_data.toarray()

    print(dense_array)

    #: [[0. 1. 1. 0. 0.]
    #   [1. 0. 0. 0. 1.]
    #   [1. 0. 0. 1. 0.]]
    ``` 
    </font>

- All the one-hot vectors are orthogonal, and there is no natural notion of similarity for them ,making the dimension of the one-hot vectors too large.



</font>