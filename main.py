import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NormalizeData:
    @classmethod
    def __normalize_column(cls, column):
        """
        Используется мин макс нормализация результат принимает значение от 0 до 100
        """
        column_min = column.min()
        column_max = column.max()
        normalize_func = np.vectorize(lambda x: (x - column_min) / (column_max - column_min) * 100)
        return normalize_func(column)

    @classmethod
    def normalize_data(cls, data):
        num_rows = len(data)
        num_columns = len(data.columns)
        matrix = np.zeros((num_rows, num_columns))
        for num_column, column in enumerate(data.columns):
            matrix[:, num_column] = cls.__normalize_column(data[column].to_numpy())
        return matrix


class VisualizeData:
    @classmethod
    def __visualize_info_column(cls, column, num_column, name_column):
        print(f"Информация о столбце №{num_column}: {name_column}")
        print(f"Среднее значение: {column.mean()}")
        print(f"Стандартное отклонение: {column.std()}")
        print(f"Минимальное значение: {column.min()}")
        print(f"Максимальное значение: {column.max()}")
        print(f"Первый квантиль: {np.quantile(column, 0)}")
        print(f"Второй квантиль: {np.quantile(column, 0.25)}")
        print(f"Третий квантиль: {np.quantile(column, 0.50)}")
        print(f"Четвертый квантиль: {np.quantile(column, 0.75)}")
        print(f"Пятый квантиль: {np.quantile(column, 1)}")
        print()
        plt.hist(column, bins=15, edgecolor='black')
        plt.title(name_column)
        plt.show()

    @classmethod
    def __visualize_info_column_result(cls, column, num_column, name_column):
        print(f"Информация о столбце №{num_column}: {name_column}")
        print("Значения принимают либо 0.0 либо 1.0\n")
        bool_count_zero = (column == 0.0)
        bool_count_one = (column == 1.0)
        count_zero = np.count_nonzero(bool_count_zero)
        count_one = np.count_nonzero(bool_count_one)
        fig, ax = plt.subplots()
        ax.pie([count_zero, count_one], labels=[f'0.0 ({count_zero})', f'1.0 ({count_one})'], autopct='%1.1f%%')
        plt.title(name_column)
        plt.show()

    @classmethod
    def visualize_info_dataset(cls, data):
        num_rows = len(data)
        num_columns = len(data.columns)
        print("Информация о датасете")
        print("Содержит столбцы: Беременность; Глюкоза; Артериальное давление; Толщина кожи; Инсулин; ИМТ; Родословная; Возраст; Результат")
        print(f"Количество строк: {num_rows}")
        print(f"Количество столбцов: {num_columns}\n")
        cls.__visualize_info_column(data.iloc[:, 0], 1, "Беременность")
        cls.__visualize_info_column(data.iloc[:, 1], 2, "Глюкоза")
        cls.__visualize_info_column(data.iloc[:, 2], 3, "Артериальное давление")
        cls.__visualize_info_column(data.iloc[:, 3], 4, "Толщина кожи")
        cls.__visualize_info_column(data.iloc[:, 4], 5, "Инсулин")
        cls.__visualize_info_column(data.iloc[:, 5], 6, "ИМТ")
        cls.__visualize_info_column(data.iloc[:, 6], 7, "Родословная")
        cls.__visualize_info_column(data.iloc[:, 7], 8, "Возраст")
        cls.__visualize_info_column_result(data.iloc[:, 8], 9, "Результат")
        print()


def main():
    data = pd.read_csv('diabetes.csv')
    VisualizeData.visualize_info_dataset(data)
    normalized_data = NormalizeData.normalize_data(data)


if __name__ == '__main__':
    main()