import numpy as np
from conv2d import Conv2D

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + eps
        pos = f()
        x[idx] = old_val - eps
        neg = f()
        x[idx] = old_val
        
        grad[idx] = (pos - neg) / (2 * eps)
        it.iternext()
    return grad

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(2, 3, 5, 5)
    conv = Conv2D(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    
    out = conv.forward(X)
    dY = np.random.randn(*out.shape)
    dX, dW, db = conv.backward(dY)

    loss_fn = lambda: np.sum(conv.forward(X) * dY)

    num_dX = numerical_gradient(loss_fn, X)
    num_dW = numerical_gradient(loss_fn, conv.W)
    num_db = numerical_gradient(loss_fn, conv.b)

    print("dX max error:", np.max(np.abs(dX - num_dX)))
    print("dW max error:", np.max(np.abs(dW - num_dW)))
    print("db max error:", np.max(np.abs(db - num_db)))
      
