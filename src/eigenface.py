import numpy as np
import sympy as sy

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
    matrix_selisih = np.array(matrix_selisih)
    return matrix_selisih

def covariant(matrixSelisih):
    # Matrix = ATranspose * A (Dimension : N^2 * N^2)
    transposeMatSelisih = np.transpose(matrixSelisih)
    # matrixCovariant = np.matmul(transposeMatSelisih, matrixSelisih)
    # Matrix = ATranspose * A (Dimension : M * M)
    matrixCovariant = np.matmul(matrixSelisih, transposeMatSelisih)
    return matrixCovariant

def eigenVal(matrixCovariant):
    lamda = sy.symbols('lamda')
    p = matrixCovariant.charpoly(lamda)
    eigenVals = sy.solve(p, lamda)
    return eigenVals

def eigenVector(matrixCovariant):
    w, matrixEigenVector = np.linalg.eig(matrixCovariant)
    return matrixEigenVector

def eigenFace(matrixEigenVector, matrixSelisih):
    matrixEigenFace = [[0 for i in range(matrixSelisih.shape[1])] for j in range(len(matrixSelisih))]
    for i in range():
        matrixEigenFace[i] = np.matmul(matrixEigenVector, matrixSelisih[i])

    return matrixEigenFace

def eigenFaceBaru(matrixEigenVector, matrixMean, vectorGambar): 
    matrixpengali = vectorGambar - matrixMean
    vectorEigenFace = np.matmul(matrixEigenVector, matrixpengali)

    return vectorEigenFace

def panjangvector(arrayVector):
    total = 0
    for i in range(len(arrayVector)):
        total += (arrayVector[i]**2)
    return total**(1/2)

def euclideanDistance(vectorEigenFace, matrixEigenFace):
    matrixEuclidean = [0 for i in range(len(matrixEigenFace))]
    for i in range(len(matrixEigenFace)):
        matrixhasil = np.subtract(matrixEigenFace[i], vectorEigenFace)
        euclidean = panjangvector(matrixhasil)
        matrixEuclidean[i] = euclidean
    return matrixEuclidean