import eigenface2

def run(imagePath, samplePath):
    images_path = imagePath
    sample_path = samplePath
    
    datasetMat = eigenface2.vectortoMatrix(images_path)
    
    datasetMean = eigenface2.mean(datasetMat)

    normalisedDataset = eigenface2.selisih(datasetMean, datasetMat)

    covDataset = eigenface2.covariance(normalisedDataset)

    matEigVec = eigenface2.eig(covDataset)

    datasetProjectionMat = eigenface2.projection(datasetMat, matEigVec)

    weightDataset = eigenface2.weightDataset(datasetProjectionMat, normalisedDataset)

    resultPath, matchPercentage = eigenface2.recogniseUnknownFace(images_path, sample_path, datasetMean, datasetProjectionMat, weightDataset)

    return (resultPath, matchPercentage)
