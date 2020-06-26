from tha.extract.histogram import histogram_RGB, historgram_RGB_seperate
from tha.extract.texture import glmc,fast_glmc
import pandas as pd
import numpy as np
import cv2 
import os


def rename_file_dataset():
    path ='dataset'
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        count = 0
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            filename_new = str(folder) + '_' + str(count) + '.jpg'
            dst =  os.path.join(folder_path, filename_new)
            os.rename(file_path, dst) 
            count += 1

if __name__ == "__main__":

    # rename_file_dataset()

    path = 'dataset'
    for folder in os.listdir(path):

        folder_path = os.path.join(path, folder)
        features = []
        filenames = []
        labels = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is None:
                print(folder, " ",filename)
            # img = cv2.resize(img, (480,640))
            # features.append(histogram_RGB(img, 8, 'global', 4))
            # features.append(fast_glmc(img, distances =[30], levels=16))

            features.append(historgram_RGB_seperate(img, 8))
            filenames.append(filename)
            labels.append(folder)
        
        # save to file
        features = np.array(features)
        filenames = np.array(filenames)
        labels = np.array(labels)

        df=pd.DataFrame(data=features[0:,0:], index=[i for i in range( features.shape[0] )], columns=[str(i) for i in range( features.shape[1] )])
        
        df['file'] = filenames
        df['label'] = labels
        df.to_csv(str(folder) + '_histogram_rr' + '.csv')
        # df.to_csv(str(folder)  + '.csv')

