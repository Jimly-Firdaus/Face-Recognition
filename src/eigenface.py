import numpy as np
import sympy as sy


def mean(matrix):
    mean_matrix = [[0 for i in range(matrix.shape[1])] for j in range(1)]
    for i in range(len(matrix)):
        temp = 1/len(matrix) * matrix[i]
        mean_matrix = np.add(temp, mean_matrix)
    return mean_matrix

def selisih(mean, matrix):
    matrix_selisih = [[0 for i in range(matrix.shape[1])] for j in range(len(matrix))]
    for i in range(len(matrix)):
        matrix_selisih[i] = np.subtract(matrix[i], mean[0])

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

# def multiplyMat(matrix1, matrix2):
#     result = [[0 for i in range(1)] for j in range(len(matrix1))]
#     result = np.array(result)
#     temp = 0
#     for i in range(len(matrix1)):
#         for j in range(len(matrix2)):
#             temp += matrix1[i][j]*matrix2[j]
#         result[i] = temp
#         temp = 0
#     return result


def trueEigenVector(matrixSelisih ,matrixEigenVector):
    newMatSelisih = np.linalg.pinv(matrixSelisih)
    matrixEigenBaru = [[0 for i in range(matrixSelisih.shape[1]) for j in range(len(matrixEigenVector))]]
    matrixEigenBaru = np.array(matrixEigenBaru)
    # transpose = np.transpose(matrixEigenVector)
    for i in range(len(matrixEigenVector)):
        matrixEigenBaru = np.matmul(newMatSelisih, matrixEigenVector)

    return matrixEigenBaru

def normEigenVector(matrixEigenVector):
    arrayNormVal = [0 for i in range(len(matrixEigenVector))]
    matrixVectorBaru = [[0 for i in range(matrixEigenVector.shape[1])] for j in range(len(matrixEigenVector))]
    matrixVectorBaru = np.array(matrixVectorBaru)
    matrixVectorBaruFloat = matrixVectorBaru.astype(np.float16)
    tempSum = 0
    for i in range(matrixEigenVector.shape[1]):
        for j in range(len(matrixEigenVector)):            
            tempSum += matrixEigenVector[j][i]**2
        tempSum = tempSum**(1/2)
        arrayNormVal[i] = tempSum
    # arrayNorm full
    for i in range(matrixEigenVector.shape[1]):
        for j in range(len(matrixEigenVector)):
            matrixVectorBaruFloat[j][i] = matrixEigenVector[j][i] * (1/arrayNormVal[i])
    return matrixVectorBaruFloat

def eigenFace(matrixVectorBaru, matrixSelisih):
    matrixEigenFace = [[0 for i in range(matrixVectorBaru.shape[1])] for j in range(len(matrixSelisih))]
    matrixEigenFace = np.array(matrixEigenFace)
    matrixEigenFaceFloat = matrixEigenFace.astype(np.float32)
    for i in range(len(matrixSelisih)):
        for j in range(matrixVectorBaru.shape[1]):
            temp = 0
            for k in range(matrixVectorBaru.shape[0]):
                temp += matrixVectorBaru[k][j] * matrixSelisih[i][k]
            matrixEigenFaceFloat[i][j] = temp
    return matrixEigenFaceFloat

def selisihEigenBaru(vectorGambarBaru, matrixMean):
    vectorSelisih = np.subtract(vectorGambarBaru, matrixMean)
    return vectorSelisih


def eigenFaceBaru(vectorSelisih, matrixVectorBaru):
    matrixEigenBaru = [0 for i in range(matrixVectorBaru.shape[1])]
    matrixEigenBaru = np.array(matrixEigenBaru)
    matrixEigenBaruFloat = matrixEigenBaru.astype(np.float16)
    for j in range(matrixVectorBaru.shape[1]):
        temp = 0
        for k in range(matrixVectorBaru.shape[0]):
            temp += matrixVectorBaru[k][j] * vectorSelisih[0][k]
        matrixEigenBaruFloat[j] = temp
    return matrixEigenBaruFloat


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

def getMinIndex(matrixEuclidean):
    minVal = matrixEuclidean[0]
    index = 0;
    for i in range(len(matrixEuclidean)):
        if minVal > matrixEuclidean[i]:
            minVal = matrixEuclidean[i]
            index = i
    return (index + 1, minVal)

def getMinIndex1(matrixEuclidean, euclideanCap):
    minVal = matrixEuclidean[0]
    index = 0;
    for i in range(len(matrixEuclidean)):
        if (minVal > matrixEuclidean[i] and matrixEuclidean[i] < euclideanCap):
            minVal = matrixEuclidean[i]
            index = i
    return index

def getPersentase(matrixEuclidean, euclideanIndex, euclideanCap):
    percentage = (euclideanCap - matrixEuclidean[euclideanIndex]) / euclideanCap
    return percentage*100