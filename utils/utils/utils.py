from time import gmtime, strftime
import numpy as np
import pandas as pd
import base64
import os

def random_name():
    time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    time = str(time) + '.jpg'
    return time

def is_before(vector, list_vectors):

    distance = [np.sum(vector-vector_i) for vector_i in list_vectors]
    # print(distance)
    if 0 in distance:
        return True #
    return False
    

def covert_img2base64(img_path):

    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return 'data:image/jpg;base64,' + str(encoded_string).split("'")[1]

def get_list_img_str_uploaded():

    path = 'F:/Current Project/Landscape Search/upload'
    img_str = []
    list_file = os.listdir(path)
    for file in list_file:
        path_file = os.path.join(path, file)
        img_str.append(covert_img2base64(path_file))

  
    return img_str, list_file

def find_have_in_database():
    
    list_imgstrs, list_file = get_list_img_str_uploaded()
    imgstr = covert_img2base64('temp.jpg')
    
    for i in range(len(list_imgstrs)):
       
        if imgstr == list_imgstrs[i]:
            
            return list_file[i]
       
    return None


def update_scores(source, target):

    FILE_RESULT = 'utils/result/result.csv'

    FILE_WEIGHT = 'utils/result/weight.csv'

    df_result = pd.read_csv(FILE_RESULT)
    
    df_result.to_csv(FILE_RESULT, index=False)
    list_img =  df_result.values[:,1].tolist()
   
    idx = list_img.index(source) 

    # print(idx)
    
    df = pd.read_csv(FILE_WEIGHT)

    max = np.max(df.iloc[idx,1:].values) + 1
    
    # print(max)
    
    print(df.iloc[idx, int(target) + 1 ])
    df.iloc[idx, int(target) + 1] = df.iloc[idx, int(target) + 1] +  0.15 * (max - df.iloc[idx, int(target) + 1])

    print(df.iloc[idx, int(target) + 1])
    
    df.to_csv(FILE_WEIGHT, index=False)
    
if __name__ == '__main__':

    path = 'F:/Current Project/Landscape Search/upload/2020-06-27-18-09-21.jpg'

    path2 = 'F:/Current Project/Landscape Search/upload/2020-06-27-18-09-21.jpg'

    # x = np.array([1,2])

    # y = np.array([[1,2], [3,4]])
    # print(is_before(x, y))

    str1 = covert_img2base64(path)
    print(str1)
    str2 = covert_img2base64(path2)

    print(str1 == str2)