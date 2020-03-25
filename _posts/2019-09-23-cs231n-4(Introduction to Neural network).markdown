---
layout: post
title: "4. Introduction to Neural network"
subtitle: 'Introduction to Neural network'
author: "Y. Yang"
header-style: text
tags:
  - Stanford-CS231N
---


# 1. Computational graphs
Suppose we have f(x,y,z) = (x+y)z

Then graph can be represented this way:

```
X         
  \
   (+)--> q ---(*)--> f
  /           /
Y            /
            /
           /
Z---------/
```

```
q = (x+y)              # dq/dx = 1 , dq/dy = 1
f = qz                 # df/dq = z , df/dz = q
```

```
df/dq = z
df/dz = q
df/dx = df/dq * dq/dx = z * 1 = z       # Chain rule
df/dy = df/dq * dq/dy = z * 1 = z       # Chain rule
```
在进行反向传播的 时候使用计算图会很方便

越是使用简单的操作作为节点，节点就会越多，计算图就会越大

