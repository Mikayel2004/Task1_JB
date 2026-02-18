import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, cfg):
        self.vocab_size = vocab_size
        self.dim = cfg.EMBED_DIM
        self.lr = cfg.LEARNING_RATE
        self.max_exp = cfg.MAX_EXP
        
        # Xavier-like init
        scale = 1.0 / self.dim
        self.w_in = np.random.uniform(-scale, scale, (vocab_size, self.dim))
        self.w_out = np.random.uniform(-scale, scale, (vocab_size, self.dim))

    def _sigmoid(self, x):
        if x > self.max_exp: return 1.0
        if x < -self.max_exp: return 0.0
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, center_id, context_id, label):
        # Returns gradient for w_in to be accumulated.
        v_in = self.w_in[center_id]
        v_out = self.w_out[context_id]

        # Forward
        score = np.dot(v_in, v_out)
        pred = self._sigmoid(score)

        # Backward (BCELoss grad)
        # grad = pred - 1 (for positive) or pred - 0 (for negative)
        grad_err = (pred - label) * self.lr

        # Update context vector (w_out) immediately
        self.w_out[context_id] -= grad_err * v_in

        # Return partial grad for center vector (w_in)
        return grad_err * v_out
