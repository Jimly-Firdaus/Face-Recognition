import os
import random
import extraction

def run():
    images_path = 'dataset/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    sample = random.sample(files, 3)
    
    extraction.batch_extractor(images_path)

    # ma = Matcher('features.pck')
    
run()