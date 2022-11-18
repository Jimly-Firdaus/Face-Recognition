import numpy as np
import sympy as sy
import os
import cv2

def imagetoVector(img):
    image = cv2.imread(r"" + img)
    image = cv2.resize(image, (256,256), interpolation= cv2.INTER_AREA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = gray_image.flatten()
    return result

def vectortoMatrix(path):
    dir = os.listdir(r"" + path)
    matrixImage = [[0 for i in range(65536)] for i in range(len(dir))]
    matrixImage = np.array(matrixImage)
    i = 0
    for image in dir:
        matrixImage[i] = imagetoVector(path+"/"+image)
        i += 1
    return matrixImage

def mean(matImgVec):
    mean_matrix = [[0 for i in range(matImgVec.shape[1])] for j in range(1)]
    for i in range(len(matImgVec)):
        temp = 1/len(matImgVec) * matImgVec[i]
        mean_matrix = np.add(temp, mean_matrix)
    return mean_matrix

def selisih(mean, matImgVec):
    matrix_selisih = [[0 for i in range(matImgVec.shape[1])] for j in range(len(matImgVec))]
    for i in range(len(matImgVec)):
        matrix_selisih[i] = np.subtract(matImgVec[i], mean[0])

    matrix_selisih = np.array(matrix_selisih)
    return matrix_selisih

def covariance(matSelisih):
    matCov = matSelisih @ np.transpose(matSelisih)
    matCov = np.divide(matCov, len(matCov))
    return matCov

def eig(matCov):
    eigVal , eigVec = np.linalg.eig(matCov)
    # Grouping eigen pairs
    reducedEigVec = np.array(eigVec).transpose()
    # Forming eigenspace
    return reducedEigVec

def projection(matImgVec, reducedEigVec):
    # Calc Eig Faces
    projection = np.dot(matImgVec.transpose(), reducedEigVec)
    projection = projection.transpose()
    return projection

def weightDataset(datasetProjection, matSelisih):
    weightDataset = np.array([np.dot(datasetProjection, i) for i in matSelisih])
    return weightDataset

def recogniseUnknownFace(pathDataset, pathTestFace, meanDataset, projectionVec, weightDataset):
    # Get test face vector
    testFace = cv2.imread(r"" + pathTestFace)
    testFace = cv2.resize(testFace, (256,256), interpolation= cv2.INTER_AREA)
    gray_image = cv2.cvtColor(testFace, cv2.COLOR_BGR2GRAY)
    testFace = gray_image.flatten()

    # Get test face normalised vector face
    vecSelisihTestFace = np.subtract(testFace, meanDataset)

    # Calc test face weight
    weightTestFace = np.dot(projectionVec, vecSelisihTestFace)
    
    # Calculate euclidean distance (in matrix form)
    euclidMat = np.absolute(weightDataset - weightTestFace)
    euclidDistance = np.linalg.norm(euclidMat, axis=1)
    print(euclidDistance)
    minimumImagesIndex = int(np.where(euclidDistance == euclidDistance.min()))[0]
    print(euclidDistance[minimumImagesIndex])
    print(min(euclidDistance))
    # minimumImagesIndex = 1
    print("Matched with image - " + str(minimumImagesIndex))

    imageFiles = [os.path.join(pathDataset, p) for p in os.listdir(pathDataset)]
    return imageFiles[minimumImagesIndex]