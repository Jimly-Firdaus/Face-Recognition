import numpy as np

def mean(matrix):
    mean_matrix = [0 for i in range(65536)]
    for i in range(len(matrix)):
        temp = 1/len(matrix) * matrix[i]
        mean_matrix = np.add(temp, mean)
    return mean_matrix

def selisih(mean, matrix):
    matrix_selisih = [[0 for i in range(65536)] for i in range(len(matrix))]
    for i in range(len(matrix)):
        matrix_selisih[i] = np.subtract(matrix[i], mean)
    return matrix_selisih

def covariant():
    pass

def eigenVal():
    pass

def eigenVector():
    pass

def eigenFace():
    pass

def euclideanDistance():
    pass