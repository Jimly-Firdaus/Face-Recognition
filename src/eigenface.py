import numpy as np

def mean(matrix):
    mean_matrix = [0 for i in range(matrix.shape[1])]
    for i in range(len(matrix)):
        temp = 1/len(matrix) * matrix[i]
        mean_matrix = np.add(temp, mean_matrix)
    return mean_matrix

def selisih(mean, matrix):
    matrix_selisih = [[0 for i in range(matrix.shape[1])] for j in range(len(matrix))]
    for i in range(len(matrix)):
        matrix_selisih[i] = np.subtract(matrix[i], mean)
    return matrix_selisih

def covariant(matrixSelisih):
    # Matrix = ATranspose * A
    transposeMatSelisih = np.transpose(matrixSelisih)
    matrixCovariant = np.matmul(transposeMatSelisih, matrixSelisih)

    return matrixCovariant

def eigenVal():
    pass

def eigenVector():
    pass

def eigenFace(matrixEigenVector, matrixSelisih):
    matrixEigenFace = [[0 for i in range(matrixSelisih.shape[1])] for j in range(len(matrixSelisih))]
    for i in range():
        matrixEigenFace[i] = np.matmul(matrixEigenVector, matrixSelisih[i])

    return matrixEigenFace

def euclideanDistance():
    pass