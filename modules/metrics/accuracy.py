from sklearn.metrics import accuracy_score
import numpy as np


class Accuracy:
    def __init__(self):
        return

    def __call__(self, logits: np.ndarray, y: np.ndarray):
        """logits:(N,10),y:(N)"""
        logits = logits.argmax(axis=1)
        assert logits.shape == y.shape, "shapes don't match"
        return accuracy_score(y_pred=logits, y_true=y)
