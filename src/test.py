import eigenface
import extraction
import numpy as np





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
    print(eigenVector)
    print(eigenVector.shape)
    print(selisih.shape)
    # print(np.shape(selisih))
    # print(np.shape(selisih))
    trueEigenVector = eigenface.trueEigenVector(selisih, eigenVector)
    print(trueEigenVector.shape)
    # matrix = [[1,1,1], [2,2,2], [3,3,3]]
    # result = eigenface.normEigenVector(matrix)
    # print(result)


    # # sample
    # extraction.batch_extractor(sample_path, "sample.pck")
    # sample = extraction.Input("sample.pck")
    # sampleEigenFace = eigenface.eigenFaceBaru(vectorEigen, mean, sample.matrix)

    # # Coba euclidean distance tapi pakai matrix selisih sama matrix biasa dlu karena blm ada matrix vector
    # # tes = np.subtract(ma.matrix[0], selisih[0])
    # # print(tes)
    # # print(eigenface.panjangvector(tes))
    
    # matrixakhir = eigenface.euclideanDistance(sampleEigenFace, vectorEigen)
    # print("Hasil euclidean: ")
    # print(matrixakhir)

run()



