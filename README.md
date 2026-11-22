# conv2d-backprop

# Conv2D Backpropagation

![Python](https://shields.io)
![NumPy](https://shields.io)
![License: MIT](https://shields.io)

A minimal, readable implementation of a **2D Convolutional Layer (`Conv2D`)** built from scratch in pure NumPy. This repository focuses on making the forward pass and backpropagation gradient math transparent, verified against numerical finite-difference gradient checking.

---

## Backpropagation Math

Given an input tensor $X$ of shape $(N, C_{\text{in}}, H, W)$, filter weights $W$ of shape $(C_{\text{out}}, C_{\text{in}}, K, K)$, bias $b$ of shape $(C_{\text{out}},)$, and the upstream loss gradient $\frac{\partial L}{\partial Y}$ of shape $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$:

1. **Bias Gradient ($\frac{\partial L}{\partial b}$)**  
   Sum the upstream gradients across the batch, height, and width dimensions for each output filter $f$:
   $$\frac{\partial L}{\partial b_f} = \sum_{n=0}^{N-1} \sum_{i=0}^{H_{\text{out}}-1} \sum_{j=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial Y_{n, f, i, j}}$$

2. **Weight Gradient ($\frac{\partial L}{\partial W}$)**  
   Accumulate the product of each input receptive field patch and its corresponding upstream gradient scalar:
   $$\frac{\partial L}{\partial W_f} = \sum_{n=0}^{N-1} \sum_{i=0}^{H_{\text{out}}-1} \sum_{j=0}^{W_{\text{out}}-1} X_{\text{padded}}[n, :, iS:iS+K, jS:jS+K] \cdot \frac{\partial L}{\partial Y_{n, f, i, j}}$$

3. **Input Gradient ($\frac{\partial L}{\partial X}$)**  
   Distribute the upstream gradient back into the padded input regions weighted by filter $W_f$, then slice off the padding:
   $$\frac{\partial L}{\partial X_{\text{padded}}}[n, :, iS:iS+K, jS:jS+K] \mathrel{+}= W_f \cdot \frac{\partial L}{\partial Y_{n, f, i, j}}$$

---

## License

This project is licensed under the [MIT License](LICENSE).
