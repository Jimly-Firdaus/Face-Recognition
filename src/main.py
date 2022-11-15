import extraction
import eigenface

def run(imagePath, samplePath):
    images_path = imagePath
    sample_path = samplePath
    
    # Extract Features on dataset
    extraction.batch_extractor(images_path)
    
    # Init Datasets
    datasetsMtrx = extraction.Matcher('features.pck')
    mean = eigenface.mean(datasetsMtrx.matrix)
    selisihMtrx = eigenface.selisih(mean, datasetsMtrx.matrix)
    covarianceMtrx = eigenface.covariant(selisihMtrx)
    eigenVectorMtrx = eigenface.eigenVector(covarianceMtrx)
    trueEigenVectorMtrx = eigenface.trueEigenVector(selisihMtrx, eigenVectorMtrx)
    normEigenVector = eigenface.normEigenVector(trueEigenVectorMtrx)
    MatrixeigenFace = eigenface.eigenFace(normEigenVector, selisihMtrx)

    extraction.batch_extractor2(sample_path)
    testFace = extraction.Input("sample.pck")
    selisihBaruMtrx = eigenface.selisihEigenBaru(testFace.matrix, mean)
    
    testFaceEigenFace = eigenface.eigenFaceBaru(selisihBaruMtrx, normEigenVector)
    euclideanDist = eigenface.euclideanDistance(testFaceEigenFace, MatrixeigenFace)

    resultIndex = eigenface.getMinIndex(euclideanDist)
    return resultIndex
