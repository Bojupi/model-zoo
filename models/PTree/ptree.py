import numpy as np
import pandas as pd

from _ptree_cpp import PTree as _PTree

class PTree(_PTree):

    def __init__(self, data: pd.DataFrame, max_depth: int, q: int, min_sample_leaf: int, factor: None|np.ndarray = None, gamma=1e-4):
        value = data.values.astype(np.float64)
        offset = np.concatenate([[0],  np.cumsum(data.groupby("date").size().to_numpy())], dtype=np.int32)
        if factor is not None:
            super().__init__(value, offset, max_depth, q, min_sample_leaf, factor, gamma)
        else:
            super().__init__(value, offset, max_depth, q, min_sample_leaf, gamma)
        
    def predict(self, data: pd.DataFrame):
        value = data.values.astype(np.float64)
        res = super().predict(value)
        out = pd.DataFrame(res, index=data.index, columns=["value", "id"])
        out["weight"] = data["weight"]
        return (out.groupby(["date", "id"], group_keys=False)["weight"].apply(lambda x: x / x.sum()) * out["value"]).values