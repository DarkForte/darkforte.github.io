---
layout: single
title: "How to derive gradients in backprop without knowing matrix calculus (Part 2)"
date: 2017-04-17
categories: Deep-Learning
author_profile: false
comments: true
---

In the [previous post](/deep-learning/how-to-derive-gradients-in-backprop1/), we showed how powerful and effective dimension analysis is. However, with it alone you will still step in trouble when taking derivatives in back propagation. The second rule is also quite important to know.

### Rule two: Stick to the chain rule.

I used to think that the chain rule is only for beginners. For example, if $$h=e^{wx+b}$$, I can figure out $$\frac{dh}{dw}=e^{wx+b}{x}$$  in ten seconds. What is the point of applying chain rule on such a simple compound function?

Unfortunately, things are not so easy in neural networks. When it comes to matrix calculus, such clever derivations can soon push you into a dead end. Still taking the example above, if all variables are matrices, we have $$H=e^{XW+b}$$. If we want to know $$\frac{dL}{dW}$$, the following equation can be quickly derived:

$$\frac{dL}{dW}=\frac{dL}{dH}\frac{dH}{dW}$$

The derivative of `L` to `H` is given by the previous layer and we want to solve $$\frac{dH}{dW}$$ here. If you have mastered dimension analysis, perhaps you will consider it to be a `DxN` matrix. After that, you will be lost and don't know how to proceed from here. It is even hard to give an appropriate draft of the derivative that "seems" to be correct. The reason is that $$\frac{dH}{dW}$$ is not a common 2-D matrix. Instead, it is a 4-D tensor that is far out of control for a beginner.

This is a common trap for a beginner where they try to figure out the final gradient in one single step, not using any intermediate variables. You will be very likely to encounter high dimensional tensors if you do so because the result of a matrix-matrix gradient is supposed to be a 4-D tensor from the very beginning. However, you will find everything in your reach if you frankly added an intermediate variable $$S=XW+b$$. Matrix-matrix gradients are 2-D matrices if a basic arithmetic operation is performed only. If all the intermediate variables are matrices, we will be able to use dimension analysis.

Let's try again. We have 

$$\frac{dL}{dS}=\frac{dL}{dH}\frac{dH}{dS}$$

where `dS` is of `NxC` and `dH` is of `NxC` too. Taking $$H=e^S$$ into account, the most possible shape of $$\frac{dH}{dS}$$ would also be `NxC`, which is $$e^S$$ itself. In this way we know this is actually an element-wise multiplication. 

Now we have 

$$\frac{dL}{dS}=e^S\frac{dL}{dH}$$

, the only thing to do is to figure out 

$$\frac{dL}{dW}=\frac{dL}{dS}\frac{dS}{dW}$$.

This is identical to the example in the previous post. It is easy to get $$\frac{dS}{dW}=X^T$$, done.

With all the derivatives known, let's look back to 

$$\frac{dL}{dW}=\frac{dL}{dH}\frac{dH}{dW}$$

and see where went wrong if you mistakenly consider $$\frac{dH}{dW}$$ to be a `DxN` matrix. Look at the following equation

$$\frac{dH}{dW}=\frac{dH}{dS}\frac{dS}{dW}$$,

we already know that $$\frac{dH}{dS}=e^S$$, $$\frac{dS}{dW}=X^T$$, these two matrices have shapes `NxC` and `DxN` respectively. You will never be able to get a `DxN` matrix no matter how you arrange them. The root cause is that the derivative of `H` to `W` is not a matrix, but a tensor. If we stick to the chain rule, we are able to avoid such tensors and can happily compute each gradient with dimension analysis and scalar calculus.

### An example: Softmax

We have learned the two rules in computing gradients in neural networks. Let's try a concrete example now: the softmax layer.

Softmax layer is usually the output layer of a network. Its forward pass is:

$$Loss=\frac{1}{n}\Sigma_{N}{-ln(P_i)}$$

$$P=\frac{e^{score_{y_i}}}{\Sigma_y{e^{score_y}}}$$

$$score=XW+b$$

Suppose `X` is a `NxD` matrix, and there are `C` classes, then `W` should be `DxC`, and `b` should be `1xC`. `Pi` refers to the i-th data's predicted probability to be its correct class. Please refer to the course notes in CS231n to get more detailed information on softmax. Our goal here is to compute the derivatives of `Loss` in terms of `W`, `X` and `b` respectively. To make reading simpler, all the derivatives $$dx$$ in the following part refers to $$\frac{d(Loss)}{dx}$$. 

First, we replace `P` in `Loss`:

$$Loss=\frac{1}{n}\Sigma_{N}{(-score_{y_i}+ln\Sigma_ye^{score_y})}$$

Don't go too fast, let's consider the first term and second term one by one. We define $$rowsum=\Sigma_ye^{score_y}$$ as an intermediate variable. `rowsum` is the exponential sum of every row in `score`, which is a `Nx1` matrix, so we have

$$Loss=\frac{1}{n}\Sigma_{N}{(-score_{y_i}+ln(rowsum_i))}$$

Consider $${d(score)}$$ first. It is of the same size `NxC` as `score`. Ignoring the 1/N term in the front, `d(score)` is actually all `zero`s, with only `one`s in every row at the position of their correct classes. It is better explained in Python code:

```python
dscore = np.zeros_like(score)
dscore[range(N),y] -= 1
```

Then let's compute $$d(rowsum)$$, it is...$$\frac{1}{rowsum}$$, so simple.

Now it's time for $$rowsum=\Sigma_ye^{score_y}$$. Do not directly compute $$\frac{d(rowsum)}{d(score)}$$ because they are both matrices and the derivative is hard to compute. Rather, since `score` is involved we can compute $$\frac{d(Loss)}{d(score)}$$. You may still remember that we have computed a `d(score)` just now, and we compute another here, which means `score` is involved into two parts of calculations in `Loss`, and it is indeed the fact (the first time as $$score_{y_i}$$, and the second time in `rowsum`). We can add them together according to the rule of taking derivatives.

When you think in this way, it's not hard to compute `d(scores)` anymore:

$$\frac{d(Loss)}{d(score)}=\frac{d(Loss)}{d(rowsum)}\frac{d(rowsum)}{d(score)}$$

The left part is a `NxC` matrix, the right part has a known matrix of `Nx1`, so the remaining part can be either `1xC` or `NxC`. This requires a bit of deeper consideration. It is not hard to reach the conclusion that it should be of `NxC` because every element in `score` affects only one element in `rowsum`, so there is no reason to sum it up. A reasonable `NxC` matrix is $$e^{score}$$ itself. At the end, we got:

```python
dscore += drowsum.dot(np.exp(score))
dscore /= N # there is a 1/N in the front
```

We are almost done here. All the left is computing the derivatives of `score` to `W`, `X` and `b`, I believe you can do it yourself since this example has appeared many times in this tutorial. :)

You may notice that the result in the second part is actually `P` itself. It can make the computation faster but it doesn't matter if you didn't notice it.

If you have mastered the two rules, you may find that even the gradients for Batch Normalization won't block you off anymore. Have a try at it.

Hope this can help the beginners who are struggling with gradient computing out a little bit.

Thank you for reading!