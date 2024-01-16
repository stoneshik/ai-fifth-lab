import random


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
        self.__create_confusion_matrix()

    def __create_confusion_matrix(self):
        for test_data_i, y_value, y_result_value in zip(range(len(self.y)), self.y, self.y_result):
            if y_result_value == y_value:
                if y_result_value == 100.0:
                    self.tp.append((self.test_data.iloc[test_data_i, 0:-1]))
                else:
                    self.tn.append((self.test_data.iloc[test_data_i, 0:-1]))
            else:
                if y_result_value == 100.0:
                    self.fp.append((self.test_data.iloc[test_data_i, 0:-1]))
                else:
                    self.fn.append((self.test_data.iloc[test_data_i, 0:-1]))

    def print_count(self):
        print(f"TP: {len(self.tp)}")
        print(f"FP: {len(self.fp)}")
        print(f"TN: {len(self.tn)}")
        print(f"FN: {len(self.fn)}")
        print(f"Всего элементов: {len(self.y)}")

    def print_accuracy(self):
        print(f"Точность: {(len(self.tp) + len(self.tn)) / (len(self.y))}")


class RandomModel:
    def __init__(self, normalized_train_data, normalized_test_data):
        self.features = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age')
        self.selected_features = self.__select_features()
        self.train_data = self.__set_train_data(normalized_train_data)
        self.test_data = self.__set_test_data(normalized_test_data)
        self.classifier = KNeighboursClassificator()

    def __select_features(self):
        k_value = random.randint(1, len(self.features))
        return random.sample(self.features, k=k_value)

    def __set_train_data(self, normalized_train_data):
        return normalized_train_data.loc[:, self.selected_features]

    def __set_test_data(self, normalized_test_data):
        return normalized_test_data.loc[:, self.selected_features]

    def result(self, k):
        y_result = [
            self.classifier.classification(self.test_data.iloc[i, :-1], self.train_data, k)
            for i in range(self.test_data.shape[0])
        ]
        confusion_matrix = ConfusionMatrix(self.test_data, self.test_data.iloc[:, -1], y_result)
        confusion_matrix.print_count()
        confusion_matrix.print_accuracy()
        print()
