import eigenface
import extraction
import numpy as np
import time
import cv2
import os
import matplotlib as plt

def show_img(path):
    img = cv2.imread(path)
    imgs = cv2.resize(img, (256,256))
    cv2.imshow("Foto termirip", imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(imgs.shape)

def run():
    images_path = 'dataset/'
    sample_path = 'sample/'
    # files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    # sample = random.sample(files, 1)
    
    extraction.batch_extractor(images_path)
    
    ma = extraction.Matcher('features.pck') # return matrix np.ndarray dari vector dataset
    # # Dimension : N^2 x M
    # rawMatrix = np.transpose(ma.matrix)
    # print(rawMatrix.shape)
    
    mean = eigenface.mean(ma.matrix)
    # print(mean.shape)
    selisih = eigenface.selisih(mean, ma.matrix)
    covariance = eigenface.covariant(selisih)
    eigenVector = eigenface.eigenVector(covariance)
    # print(eigenVector)
    # print(eigenVector.shape)
    # print(selisih.shape)
    # print(selisih)
    trueEigenVector = eigenface.trueEigenVector(selisih, eigenVector)
    # print(trueEigenVector.shape)
    # print(trueEigenVector)

    # print("Matrix Normalised eigen vector: ")
    normEigenVector = eigenface.normEigenVector(trueEigenVector)
    # print(normEigenVector.shape)
    # print(normEigenVector)

    # print("Matrix Eigen Face: ")
    MatrixeigenFace = eigenface.eigenFace(normEigenVector, selisih)

    # print("Shape: ")
    # print(MatrixeigenFace.shape)
    # print(MatrixeigenFace)
    print("File test face: ")
    extraction.batch_extractor(sample_path, "sample.pck")
    sample = extraction.Input("sample.pck")
    # print(sample.matrix.shape)
    # print(np.shape(mean))
    # print("Matrix selisih baru: ")
    selisihbaru = eigenface.selisihEigenBaru(sample.matrix, mean)

    # print(np.shape(selisihbaru))
    # print(selisihbaru)

    # print("Vector eigen face yang di tes: ")
    eigenFacetes = eigenface.eigenFaceBaru(selisihbaru, normEigenVector)
    # print(eigenFacetes.shape)
    # print(eigenFacetes) 

    # print("Matrix euclidean distance: ")
    euclidean = eigenface.euclideanDistance(eigenFacetes, MatrixeigenFace)
    # print(np.shape(euclidean))
    # print(euclidean)
    resultIndex = eigenface.getMinIndex(euclidean)
    print("Result is: ")
    print(resultIndex)
    # sample
    # sampleEigenFace = eigenface.eigenFaceBaru(vectorEigen, mean, sample.matrix)

    # # Coba euclidean distance tapi pakai matrix selisih sama matrix biasa dlu karena blm ada matrix vector
    # # tes = np.subtract(ma.matrix[0], selisih[0])
    # # print(tes)
    # # print(eigenface.panjangvector(tes))
    
    # matrixakhir = eigenface.euclideanDistance(sampleEigenFace, vectorEigen)
    # print("Hasil euclidean: ")
    # print(euclidean)
    show_img(os.path.join(images_path, ma.names[resultIndex]))


    

startTime = time.time()
run()
print("--%s seconds--" %(time.time() - startTime))


