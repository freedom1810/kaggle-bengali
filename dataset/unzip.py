import pandas as pd
import numpy as np
import gc
import time
import cv2

def load_data():
    indices=[0,1,2,3]
    images = []
    count_image = 0

    for i in indices:
        df =  pd.read_parquet('data/train_image_data_{}.parquet'.format(i)) 
        HEIGHT = 137
        WIDTH = 236
        images = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
        del df
        gc.collect()

        
        image_name = pd.read_csv('data/train.csv')['image_id'].values
        for img in images:
            cv2.imwrite('data/images/' + image_name[count_image] + '.png', 255 - img)

            count_image += 1

        
        del images
        gc.collect()

# !mkdir images
tic = time.time()
load_data()
print(time.time() - tic)