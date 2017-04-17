---
layout: single
title: "How to derive gradients in backprop without knowing matrix calculus (Part 1)"
date: 2017-04-17
categories: Deep-Learning
author_profile: false
comment: true
---

I believe lots of people had a hard time deriving the gradients in back propagation when they are learning about neural networks (or deep learning). You understand every example in the lecture, but when it comes to the homework you will find everything vectorized, making your skills on scalar calculus nearly useless. Perhaps matrix differentiation is the biggests obstacle for a newcomer to deep learning. 

When I was learning CS231n I also struggled on deriving vectorized gradients for quite a long time. Although Karpathy has offered a tutorial on matrix calculus [here](http://cs231n.stanford.edu/vecDerivs.pdf), it did not make things easier because understanding the tutorial itself is also challenging. However, after struggling for weeks, I come to realize that deriving the gradients in backprop is not hard at all, as long as you are aware of the following two rules.

### Rule one: Use dimension analysis.

The first rule in deriving gradients in a neural network is: do not compute matrix-matrix gradients directly unless you are very confident on your matrix calculus skills. By using dimension analysis, one can work out every matrix-matrix gradients indirectly with scalar calculus in neural networks. I regard dimension analysis as the most useful tool here. Dimension analysis can save you from all the troublesome problems like analyzing the gradients element by element, wondering whether to sum or not, arranging matrix multiplication order, considering when to transpose a matrix and so on. It is also mentioned a bit in the course notes in CS231n.

So what is dimension analysis? Let's take an example. Suppose the forward pass is $$score=XW+b$$, where the shape of `X` is NxD, `W` is DxC and `b` is 1xC, so the shape of `score` is NxC. Now the gradient of `Loss` (marked as `L` below) to `score` is given to us by the previous layer. Let's derive the gradients of `Loss` to `W` and `b`.

We know that $$\frac{dL}{dscore}$$ is a matrix of NxC, because `Loss` is a scalar, so every single element's change in `score` will cause `Loss` to change. So we have 

$$\frac{dL}{dW}=\frac{dL}{dscore}\frac{dscore}{dW}$$

OK, here comes the problem. We need $$\frac{dscore}{dW}$$, but both `score` and `W` are matrices. How to compute that gradient? I believe many of those who gave up machine learning is because they found they cannot even work out such a "simple" gradient and quickly lost confidence.

Actually here is where dimension analysis shows its power. We do not calculate $$\frac{dscore}{dW}$$ directly, instead we work it out with the help of the other two gradients. First, we consider the shape of it. We know $$\frac{dL}{dW}$$ is of DxC because it should be of the same size with W, and $$\frac{dL}{dscore}$$ is of NxC, so we soon found $$\frac{dscore}{dW}$$ should be of DxN, because (DxN) x (NxC) => (DxC). BTW, you can notice that the multiplication order is not right. It should be this:

$$\frac{dL}{dW}=\frac{dscore}{dW}\frac{dL}{dscore}$$

So we know $$\frac{dscore}{dW}$$ is of DxN, it is almost done. Since $$score = XW + b$$，if they were scalars, the gradient of `score` to `W` would be `X` itself. X is of NxD, we want DxN, just transpose it: 

$$\frac{dL}{dW}=\frac{dL}{dscore}X^T$$

That's all.

Look, we don't use traditional detailed element-wise analysis on things like what $$\frac{dscore_{11}}{dW_{11}}$$ is like. Instead, we leverage our knowledge on pure scalar calculus and worked out the desired gradient by using $$\frac{dL}{dW}$$ and $$\frac{dL}{dscore}$$. This is the most efficient way to figure out gradients in neural networks.

Why it always works? The key point is that `Loss` is always a scalar. The size of the gradient of a scalar to a matrix is always the same with the matrix itself. Therefore, you can always work out the matrix-matrix gradient by using two scalar-matrix gradients of which you know the size. First figure out the size of the unknown matrix-matrix gradient, then guess the value of it with scalar calculus, finally find a proper way to make it fit the desired shape and you are done.

What about $$\frac{dL}{db}$$? We know $$\frac{dL}{dscore}$$ is of NxC, $$\frac{dL}{db}$$ is 1xC, and $$\frac{dscore}{db}$$ looks like 1, so you may have realized that $$\frac{dscore}{db}$$ is all 1 by 1xN, because (1xN) x (NxC) => (1xC). This is equivalent to summing up $$dscore$$ vertically, reducing it from NxC to 1xC. 

It will be kind of hard to come up with the sum operation if you analyze it element-wisely. The sum is introduced by "broadcasting" in Python. `XW` is a matrix of NxC, but `b` is only 1xC, so they cannot be added together in theory. But what we want to do here is to add `b` to every row in `XW`. So we actually duplicated `b` for N times, forcing it to become an NxC matrix, and then added it to `XW`. Therefore, when computing the gradients, you have to sum up the gradient for every row since `b` is actually involved into the calculation for N times. Remember the gradient rule: if a variable is involved into multiple calculations, then you should sum up its gradient in each calculation. One illustration could make this more clear.

![broadcasting](/assets/broadcasting.png)

(Picture from: [https://www.zhihu.com/question/47024992/answer/103962301])

In short, never try to compute matrix-matrix gradients directly in neural networks. Make good use of dimension analysis and it can save you tons of trouble.