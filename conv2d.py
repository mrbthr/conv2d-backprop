import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.cache = None

    def forward(self, X):
        """
        X shape: (batch_size, in_channels, H, W)
        """
        N, C, H, W = X.shape
        K = self.kernel_size
        S, P = self.stride, self.padding

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        X_padded = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        out = np.zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):
            for f in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * S, j * S
                        patch = X_padded[n, :, h_start:h_start+K, w_start:w_start+K]
                        out[n, f, i, j] = np.sum(patch * self.W[f]) + self.b[f]

        self.cache = (X, X_padded)
        return out

    def backward(self, dY):
        """
        dY shape: (batch_size, out_channels, H_out, W_out)
        Returns dX, dW, db
        """
        X, X_padded = self.cache
        N, C, H, W = X.shape
        _, F, H_out, W_out = dY.shape
        K = self.kernel_size
        S, P = self.stride, self.padding

        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(self.W)
        db = np.sum(dY, axis=(0, 2, 3))

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * S, j * S
                        patch = X_padded[n, :, h_start:h_start+K, w_start:w_start+K]
                        
                        # Accumulate weight and input gradients
                        dW[f] += patch * dY[n, f, i, j]
                        dX_padded[n, :, h_start:h_start+K, w_start:w_start+K] += self.W[f] * dY[n, f, i, j]

        
        dX = dX_padded[:, :, P:P+H, P:P+W] if P > 0 else dX_padded
        return dX, dW, db
