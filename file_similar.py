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
    for i in range( list_ft.shape[1] ):
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
    colors = get_top_file(ft_hist, list_ft_1, list_file, top=15)
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

    result = get_top_file(ft_all, list_ft_all, list_file, top=15)
    # print(result)

    tmp = result or colors
    
    return cluster, colors, tmp

def get_result(path):
    labels, result, tmp = get_list_file_similarity(path)

    # list_img = [path.split('/')[-1]]
    list_img = result
    list_img += tmp

    return labels, list_img

if __name__ == '__main__':

    path = 'F:\Current Project\Landscape Search\dataset/temple/temple_97.jpg'
    labels, result, tmp = get_list_file_similarity(path)

    # list_img = [path.split('/')[-1]]
    list_img = result
    list_img += tmp
    imgs = []
    img = cv2.imread(path)
    imgs.append(img)
    for file in list_img:
        path = os.path.join('dataset', labels, file)
        img = cv2.imread(path)
        img = cv2.resize(img,(128,128))
        imgs.append(img)

    frame = build_montages(imgs, (96, 96), (6, 6))[0]

    cv2.imshow("result", frame)
    cv2.waitKey(0)
        

    


    

    




