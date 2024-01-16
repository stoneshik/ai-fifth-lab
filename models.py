import math
import pandas as pd
import numpy as np


class KNeighboursClassificator:

    @classmethod
    def __distance(cls, l1, l2) -> float:
        assert len(l1) == len(l2)
        return sum([(l1_i - l2_i) ** 2 for l1_i, l2_i in zip(l1, l2)]) ** 0.5

    # Search function for the most frequent sample value
    @classmethod
    def __most_frequent(cls, l) -> str:
        count = {}
        for l_i in l:
            if l_i in count.keys():
                count[l_i] += 1
            else:
                count[l_i] = 1
        count = sorted(count.items(), key=lambda item: item[1], reverse=True)
        return count[0][0]

    # Classification function
    def classification(self, data, df, k: int) -> str:
        dist = []
        # Calculation of distances to each point of the training sample
        for i in range(df.shape[0]):
            dist.append((i, self.__distance(data, df.iloc[i, :-1])))
        # Search for values of the target variable
        dist.sort(key=lambda item: item[1])
        values = [df.iloc[d[0], -1] for d in dist[:k]]
        return self.__most_frequent(values)


