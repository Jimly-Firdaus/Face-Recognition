import eigenface2
import numpy as np
import time
import cv2
import os
import matplotlib as plt

def show_img(path):
    img = cv2.imread(path)
    imgs = cv2.resize(img, (256,256))
    cv2.imshow("Foto termirip" + path, imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(imgs.shape)

def run():
    images_path = 'dataset2/'
    sample_path = 'sample\IMG-20221117-WA0009.jpg'
    datasetMat = eigenface2.vectortoMatrix(images_path)
    
    datasetMean = eigenface2.mean(datasetMat)

    normalisedDataset = eigenface2.selisih(datasetMean, datasetMat)

    covDataset = eigenface2.covariance(normalisedDataset)

    matEigVec = eigenface2.eig(covDataset)

    datasetProjectionMat = eigenface2.projection(datasetMat, matEigVec)

    weightDataset = eigenface2.weightDataset(datasetProjectionMat, normalisedDataset)

    resultPath = eigenface2.recogniseUnknownFace(images_path, sample_path, datasetMean, datasetProjectionMat, weightDataset)
    
    show_img(resultPath)


startTime = time.time()
run()
print("--%s seconds--" %(time.time() - startTime))


