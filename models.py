import math
import pandas as pd
import numpy as np


class KNeighboursClassificator:
    def __init__(self, x_matrix, y_matrix, param_names):
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.param_names = param_names
        self.b_vector = self.__calc_b_vector()  # вектор с найденными коэффициентами
        self.y_fit = self.__fit()
        self.epsilon = self.__calc_epsilon()  # вектор остатков
        self.r_square = self.__calc_r_square()  # коэффициент детерминации

    def __dist(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def __calc_r_square(self, trainData, testData, k, numberOfClasses):
        testLabels = []
        for testPoint in testData:
            # Claculate distances between test point and all of the train points
            testDist = [
                [math.dist(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))
            ]
            # How many points of each class among nearest K
            stat = [0 for i in range(numberOfClasses)]
            for d in sorted(testDist)[0:k]:
                stat[d[1]] += 1
            # Assign a class with the most number of occurences among K nearest neighbours
            testLabels.append(sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1])
        return testLabels


