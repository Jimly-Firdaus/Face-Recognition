import eigenface
import extraction
import numpy as np



def run():
    images_path = 'dataset/'
    # files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    # sample = random.sample(files, 1)
    
    extraction.batch_extractor(images_path)

    ma = extraction.Matcher('features.pck') # return matrix np.ndarray dari vector dataset
    # print(ma.matrix)
    print(ma.matrix.shape)
    print(type(ma.matrix))
    print(len(ma.matrix))
    print(ma.matrix)

    mean = eigenface.mean(ma.matrix)
    print("mean : ")
    print(len(mean))
    print(mean.shape[0])
    print(mean)

    selisih = eigenface.selisih(mean, ma.matrix)
    transposeSelisih = np.transpose(selisih)
    result = np.matmul(selisih, transposeSelisih)
    result *= 1/len(result)
    print(result)
    print("selisih : ")
    print(selisih)
    print(selisih.shape)
    
    covariant = eigenface.covariant(selisih)
    print("covariant : ")
    print(covariant)
    print(np.shape(covariant))

    vectoreigen = ma.matrix[0]
    panjangvectoreigen = eigenface.panjangvector(vectoreigen)
    print(panjangvectoreigen)

    # Coba euclidean distance tapi pakai matrix selisih sama matrix biasa dlu karena blm ada matrix vector
    tes = np.subtract(ma.matrix[0], selisih[0])
    print(tes)
    print(eigenface.panjangvector(tes))
    matrixakhir = eigenface.euclideanDistance(vectoreigen, selisih)
    print("Hasil euclidean: ")
    print(matrixakhir)

run()



