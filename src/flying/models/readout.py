"""Only this affine softmax classifier is trained (full-batch Adam)."""
import numpy as np
from scipy.special import softmax

class Readout:
    def fit(self, x, y, epochs=400, learning_rate=0.03, l2=1e-5, feature_stats=None):
        if len(x) != len(y) or len(x) == 0 or epochs < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("Invalid training configuration")
        if feature_stats is None:
            self.mean = x.mean(axis=0)
            self.scale = np.maximum(x.std(axis=0), 1e-5)
        else:
            self.mean, self.scale = (np.array(a, dtype=float, copy=True) for a in feature_stats)
            if (self.mean.shape != (x.shape[1],) or self.scale.shape != self.mean.shape
                    or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                    or np.any(self.scale <= 0)):
                raise ValueError("Invalid fixed feature statistics")
        z = self.features(x)
        self.weights = np.zeros((z.shape[1], 10))
        m = np.zeros_like(self.weights); v = m.copy(); history = []
        target = np.eye(10)[y]
        for epoch in range(1, epochs + 1):
            prob = softmax(z @ self.weights, axis=1)
            reg = self.weights.copy(); reg[-1] = 0
            grad = z.T @ (prob - target) / len(y) + l2 * reg
            m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad**2
            self.weights -= learning_rate * (m / (1 - 0.9**epoch)) / (np.sqrt(v / (1 - 0.999**epoch)) + 1e-8)
            p = softmax(z @ self.weights, axis=1)
            history.append(dict(epoch=epoch, accuracy=float(np.mean(p.argmax(axis=1) == y)),
                                cross_entropy=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean())))
        return history

    def features(self, x):
        x = np.atleast_2d(x)
        return np.column_stack(((x - self.mean) / self.scale, np.ones(len(x))))

    def predict(self, x):
        return np.argmax(self.features(x) @ self.weights, axis=1)
