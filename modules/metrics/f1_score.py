from sklearn.metrics import f1_score
import numpy as np


class F1_score:
    def __init__(self):
        pass

    def __call__(self, logits: np.ndarray, y: np.ndarray):
        """logits:(N,10),y:(N)"""
        logits = logits.argmax(axis=1)
        assert logits.shape == y.shape, "shapes don't match"
        return f1_score(y_pred=logits, y_true=y, average="weighted")
