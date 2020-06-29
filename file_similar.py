from utils.utils.clustering import KdTreeDecisionTree
from tha.compare.similary import Euclidean,Manhattan 
from tha.extract.histogram import histogram_RGB 
from tha.extract.texture import fast_glmc
from tha.compare.similary import Euclidean
from imutils import build_montages
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np 
import os
import cv2


def get_top_file(ft, list_ft, list_file, top = 30):

    distance =  [Euclidean(ft, ft_i) for ft_i in list_ft]

    index_sort = np.argsort(distance)
    # print(index_sort)
    index_return = index_sort[:top]


    return [list_file[i] for i in index_return]


def normalize_ft_texture(list_ft):
    # chuẩn hóa miền giá trị của từng thuộc tính về trong khoảng (0,1)
    list_ft = np.array(list_ft)
    ft_nomalize =None
    for i in range(list_ft.shape[1] ):
        scaler = MinMaxScaler()
        vector = np.expand_dims(list_ft[:, i], axis = -1)
        ft = scaler.fit_transform(vector)
        if ft_nomalize is None:
            ft_nomalize = ft
        else:
            ft_nomalize = np.concatenate((ft_nomalize,ft), axis = 1)

    return ft_nomalize[-1, :],ft_nomalize[:-1,:]

def get_list_file_similarity(img_path=None, img=None, clf = None):

    if img_path is None:

        img = cv2.imread(img_path)

    if clf is None:
        clf = KdTreeDecisionTree("utils/model/model.pkl")

    ft_hist = clf.get_ft_RGB(img_path)

    ft_texture = clf.get_ft_texture(img_path)
    
    cluster = clf.predict(ft_hist, ft_texture)

    print("[INFOR]: Cluster ", cluster)

    path_ft = "utils/extracted"

    file = [str(cluster)+".csv", str(cluster)+"_histogram.csv"]

    file = [os.path.join(path_ft, file_tmp) for file_tmp in file]

    # get file using RGB
    df = pd.read_csv(file[1])
    list_ft_1 = df.iloc[:, 1:-2].values
    list_file = df.iloc[:, -2].values
    colors = get_top_file(ft_hist, list_ft_1, list_file, top=20)
    # print(colors)

    # get file using texture
    df = pd.read_csv(file[0])
    list_ft_2 = df.iloc[:, 1:-2].values
    list_file = df.iloc[:, -2].values

    list_ft_2 = list_ft_2.tolist()
    list_ft_2.append(ft_texture)

    ft, list_ft_2 = normalize_ft_texture(list_ft_2)

    list_ft_all = np.concatenate((list_ft_2, list_ft_1), axis = 1)
    ft_all = np.concatenate((ft, ft_hist), axis = 0)

    # textures = get_top_file(ft, list_ft, list_file, top=25)
    # print(textures)

    result = get_top_file(ft_all, list_ft_all, list_file, top=10)
    # print(result)

    tmp = [file for file in result if file not in colors]
    
    # print(colors)
    # print(tmp)

    return cluster, colors, tmp

def get_result(path):

    labels, result, tmp = get_list_file_similarity(path)

    # list_img = [path.split('/')[-1]]
    list_img = result
    list_img += tmp

    return labels, list_img[:18]

def get_result_as_row(file_name):

    file_name = 'upload/'+ file_name
    FILE_RESULT = 'utils/result/result.csv'

    FILE_WEIGHT = 'utils/result/weight.csv'

    # df =  pd.read_csv(FILE_RESULT)
    # df.to_csv(FILE_RESULT, index=False)

    # df =  pd.read_csv(FILE_WEIGHT)
    # df.to_csv(FILE_WEIGHT, index=False)


    df_result = pd.read_csv(FILE_RESULT).values
    df_weight = pd.read_csv(FILE_WEIGHT).values

    
    list_img =  df_result[:,1].tolist()
    
    # print(list_img)
    idx = list_img.index(file_name) 
    
    list_img = df_result[idx,2:]
    
    list_weight = df_weight[idx, 1:]
   
    index_sort = np.argsort(list_weight)
    return [list_img[i] for i in np.flip(index_sort)], np.flip(index_sort)

if __name__ == '__main__':

    # path = 'F:/Current Project/Landscape Search/dataset/temple/temple_97.jpg'
    # labels, result, tmp = get_list_file_similarity(path)

    # # list_img = [path.split('/')[-1]]
    # list_img = result
    # list_img += tmp
    # imgs = []
    # img = cv2.imread(path)
    # imgs.append(img)
    # for file in list_img:
    #     path = os.path.join('dataset', labels, file)
    #     img = cv2.imread(path)
    #     img = cv2.resize(img,(128,128))
    #     imgs.append(img)

    # frame = build_montages(imgs, (96, 96), (6, 6))[0]

    # cv2.imshow("result", frame)
    # cv2.waitKey(0)

    # FILE_RESULT = 'utils/result/result.csv'

    # FILE_WEIGHT = 'utils/result/weight.csv'

    # # make dataset initial

    # labels, result, tmp = get_list_file_similarity(path)

    # list_img = [path.split('/')[-1]]
    # list_img += result
    # list_img += tmp

    # list_img = list_img[:19]

    # list_weight = [(len(list_img) - i - 1) for i in range(len(list_img)-1)]

    # data = {str(key): list_img[key] for key in range(len(list_img))}

    # df = pd.DataFrame(data, columns = [str(key) for key in range(len(list_img))] , index = [0])

    # data2 = {str(key): list_weight[key] for key in range(len(list_weight))}

    # df2 = pd.DataFrame(data2, columns = [str(key) for key in range(len(list_weight))], index = [0])

    # df.to_csv(FILE_RESULT)
    # df2.to_csv(FILE_WEIGHT)

    
    # file_name = '2020-06-28-16-08-45.jpg'


    # print(get_result_as_row(file_name))

    x =  np.array([1,5,6])

    print(np.flip(x))

    



        

    


    

    




