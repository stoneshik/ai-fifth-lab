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
        # Расчет расстояний до каждой точки обучающей выборки
        for i in range(df.shape[0]):
            dist.append((i, self.__distance(data, df.iloc[i, :-1])))
        # Поиск значений целевой переменной
        dist.sort(key=lambda item: item[1])
        values = [df.iloc[d[0], -1] for d in dist[:k]]
        return self.__most_frequent(values)


class ConfusionMatrix:
    def __init__(self, test_data, y, y_result):
        self.test_data = test_data
        self.y = y
        self.y_result = y_result
        self.tp = []
        self.fp = []
        self.tn = []
        self.fn = []

    def create_confusion_matrix(self):
        for test_data_i, y_value, y_result_value in zip(range(self.test_data.shape[0]), self.y, self.y_result):
            if y_result_value == y_value == 100.0:
                self.tp.append(self.test_data[test_data_i, 0:-2])


class RandomModel:
    def __init__(self, normalized_train_data, normalized_test_data):
        self.normalized_train_data = normalized_train_data
        self.normalized_test_data = normalized_test_data
        self.classifier = KNeighboursClassificator()

    def result(self, k):
        my_pred = [
            self.classifier.classification(self.normalized_test_data.iloc[i, :-1], self.normalized_train_data, k)
            for i in range(self.normalized_test_data.shape[0])
        ]
        y_result = [(self.normalized_test_data.iloc[i, -1], my_pred[i]) for i in range(self.normalized_test_data.shape[0])]

        print('My algorithm\'s accuracy:', sum([test == pred for test, pred in l]) / len(l))
