from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np #
import pandas as pd #
import pickle
import cv2 #
import os #
from sklearn import tree
from sklearn.externals import joblib 
from tha.extract.histogram import histogram_RGB
from tha.extract.texture import fast_glmc


'''
Build a k-d tree decreas search space

Using decision tree algorithm

'''

class KdTreeDecisionTree:

    def __init__(self, path_model):
        
        self.clf = joblib.load(path_model)  
    

    def get_ft_RGB(self,img_path=None, img =None):

        if img_path is not None:
            img_his =  cv2.imread(img_path)
        
        elif img_path is None and img is None:
            raise Exception("Input a image or path of image")
        
        else:
            img_his = img

        return  histogram_RGB(img_his, 8, 'global', 4)

    def get_ft_texture(self,img_path=None, img= None):

        if img_path is not None:
            img_text = cv2.imread(img_path, 0)
            
        
        elif img_path is None and img is None:
            raise Exception("Input a image or path of image")
        
        else:
            img_text = img
        
        img_text = cv2.resize(img_text,(480,640))
        return fast_glmc(img_text, distances =[30], levels=16)

    def feature_extract(self, feature_hist, feature_text):

        return np.concatenate((feature_hist, feature_text), axis = 0)

    def predict(self,feature_hist,feature_text):
        
        ft = self.feature_extract(feature_hist,feature_text)

        pred = self.clf.predict([ft])
        
        if pred[0] == 0:
            return 'beach'
        elif pred[0] == 1:
            return 'skyscraper' 
        
        elif pred[0] == 2:
            return 'temple'
        
        else:
            return 'terraces'

    def get_model(self):

        return self.clf


def load_feature(filename):

    path = 'utils/extracted'
    X_train, y_train, X_test, y_test = None, None, None, None
    for file in filename:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
     
        X =  df.values[:, 1:-2]
       
        y =  df.values[:, -1]
        print(y.shape)

        if X_train is None:
            X_train = X[:-20]
            y_train = y[:-20]
            X_test  = X[-20:]
            y_test  = y[-20:]

        else:
            X_train = np.concatenate((X_train, X[:-20]), axis = 0)
            y_train  =np.concatenate((y_train, y[:-20]), axis = 0)
            X_test = np.concatenate((X_test,   X[-20:]), axis =0)
            y_test = np.concatenate((y_test,   y[-20:]), axis =0)

    print("[INFOR] : Training shape", X_train.shape)
    print("[INFOR] : Testing shape", X_test.shape)
    print("[INFOR] : Label train shape", y_train.shape)
    print("[INFOR] : Label test shape", y_test.shape)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":

    # load feature from file
    filename1 = ['beach.csv', 'skyscraper.csv', 'temple.csv','terraces.csv']
    filename = ['beach_histogram.csv', 'skyscraper_histogram.csv', 'temple_histogram.csv','terraces_histogram.csv']
    filename2 = ['beach_histogram_rr.csv', 'skyscraper_histogram_rr.csv', 'temple_histogram_rr.csv','terraces_histogram_rr.csv']

    # X_train1, y_train1, X_test1, y_test1 = load_feature(filename)
    # X_train2, y_train2, X_test2, y_test2 = load_feature(filename1)

    # print(X_train1.shape)
    # print(X_train2.shape)
    # X_train = np.concatenate((X_train1, X_train2), axis = 1)
    # X_test = np.concatenate((X_test1, X_test2), axis = 1)
    # y_train = y_train1
    # y_test = y_test1

    X_train, y_train, X_test, y_test = load_feature(filename1)
    print("[INFOR] : Training shape", X_train.shape)
    print("[INFOR] : Testing shape", X_test.shape)
    print("[INFOR] : Label train shape", y_train.shape)
    print("[INFOR] : Label test shape", y_test.shape)

    # encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    # print(y_train[:10])
    file = open('encode.txt', 'w')
    for key,value in enumerate(le.classes_):
        file.write( str(key) + ":" + str(value)+"\n" )
    file.close()

    # print(X_train[0])
    max = 0
    for i in range(159):
        clf = DecisionTreeClassifier(max_features=i+1,random_state=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test) 
        acc = accuracy_score(y_test,y_pred)*100
        print( acc )
        if acc > max:
            max = acc
            # joblib.dump(clf, 'model.pkl')
    
    print('[MAX]:', max)

