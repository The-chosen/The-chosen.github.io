---
layout: post
title: "5. Convolutional neural networks (CNNs)"
subtitle: 'Convolutional neural networks (CNNs)'
author: "Y. Yang"
header-style: text
tags:
  - Stanford-CS231N
---


# 1. Fully connected layer(dense layer)
all the neurons are connected 

# 2. Convolution layer
就是过滤器在前一层滑动，每一个位置就对应一个值（点乘）。过滤器的深度就等于前一层的深度。

output of convolution layer 叫做 activation map。注意：一个filter产生的输出就叫做一个activation map

一开始的卷积层学一些lower feature，靠后的卷积层学一些higher feature

过滤之后的长度计算公式：
((N - F) / stride + 1)

此处有几个超参数：
1. stride(1 2)
2. size of filter(3 5 7 ...)
3. number of filter(power of 2)
4. amount of padding(sizeOfFilter // 2)


# 3. Pooling layer
makes the representation smaller and more manageable.

Pooling Operates over each activation map independently.

Pooling主要两种
- max
- avg

Pooling层的参数主要就是
1. size of filter 
2. stride 

在pooling层，size of filter = stride