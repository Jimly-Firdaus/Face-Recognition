import os
import random
import extraction
import numpy as np
import pickle




def run():
    images_path = 'dataset/'
    # files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    # sample = random.sample(files, 1)
    
    extraction.batch_extractor(images_path)

    ma = extraction.Matcher('features.pck') # return matrix dari vector dataset
    # print(ma.matrix)
    

run()



