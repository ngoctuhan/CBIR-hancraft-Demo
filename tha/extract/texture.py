import cv2 
import numpy as np 
import math
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

def quantization_image(img, n_colors = 8, visualize = False):

    '''
    Define:
            Quantization Image: lượng tử hóa hình ảnh
            Lượng tử hóa màu là quá trình giảm số lượng màu khác biệt trong một hình ảnh.
    Input:

            - n_colors: số màu dùng lượng tử hóa (hằng số) : int , default: 8 

    Output:
            - hình ảnh sau khi lượng tử hóa

    '''
    image = img.copy()
    (h, w) = image.shape[:2]

    # Không gian màu mới được sử dụng cho đầu vào K-Means
    # Khoảng cách được sử dụng là Euclic
    # L*a*b* color space where the euclidean distance implies
        
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Thay đổi kích thước đầu vào mà K-means có thể được áp dụng

    image = image.reshape((image.shape[0] * image.shape[1], 3))
        
    clt = MiniBatchKMeans(n_clusters = n_colors)
    labels = clt.fit_predict(image)

    quant = clt.cluster_centers_.astype("uint8")[labels]

    # print(clt.cluster_centers_)
    centroids = clt.cluster_centers_

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    if visualize:
        
        cv2.imshow("image", np.hstack([image, quant]))
        cv2.waitKey(0)

    return quant, centroids

def glmc_cal(img, n_bins = 16,distance = 50, n_neighbors = 8):

    # print("GLCM cal")

    import time 
    # t1 = time.time()
    h,w = img.shape[0], img.shape[1]

    dx = np.array([0, -1, -1, -1]) * distance
    dy = np.array([1, 1, 0, -1]) * distance

    glmc_maxtrix = np.zeros((n_bins, n_bins))
    
    count = 0
    for i in range(h):
        for j in range(w):

            # 8 hướng là hàm xóm của (i, j) và cách (i,j) một khoảng là d
            for k in range(dx.shape[0]):
                if (i + dx[k]) < h and (j + dy[k]) < w:
                    glmc_maxtrix[img[i,j], img[i+dx[k], j+dy[k]]] +=1
                    count += 1

    
    glmc_maxtrix /= count
    # t2 = time.time()
    # print("Time glmc cal :", t2 - t1)
    # print(glmc_maxtrix.shape)
    return glmc_maxtrix
            
def fast_glmc(image=None, distances= [40], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels = 16):
    
    mi, ma =  0, 255

    # mã hóa
    bins = np.linspace(mi, ma+1, levels+1)
    image = np.digitize(image, bins) - 1 # encode bins[i] < img <= bins[i+1]
    result = greycomatrix(image, distances, angles, levels)

    glcm_0   = result[:, :, 0, 0]
    glcm_45  = result[:, :, 0, 1]
    glcm_90  = result[:, :, 0, 2]
    glcm_135 = result[:, :, 0, 3]
    

    # print( statistical_glmc(glcm_0/ np.sum(glcm_0) ))
    # glcm_0 = np.expand_dims(glcm_0 , axis=-1)
    # glcm_0 = np.expand_dims(glcm_0 , axis=-1)
    # print('Corr: ',greycoprops(glcm_0, prop='correlation'))
    # print('ASM : ',greycoprops(glcm_0, prop='ASM'))
    # print('Diss: ',greycoprops(glcm_0, prop='dissimilarity'))

    glcm =  statistical_glmc(glcm_0/np.sum(glcm_0)) + statistical_glmc(glcm_45/np.sum(glcm_45)) + statistical_glmc(glcm_90/np.sum(glcm_90)) + statistical_glmc(glcm_135/np.sum(glcm_135))
    return glcm

def statistical_glmc(glmc_maxtrix):

    # print(glmc_maxtrix)
    # print("Extract feature")
    feature = []
    # name = ['max', 'contrast']

    # maximum probability: cho biết 2 ảnh có tương tự về số cặp pixel 
    feature.append(np.max(glmc_maxtrix))

    # độ tương phản
    def contrast(glmc_mt):
        h,w = glmc_mt.shape[0],glmc_mt.shape[1]
        contrast_value = 0
        for i in range(h):
            for j in range(w):
                contrast_value += glmc_mt[i,j]*(i - j) ** 2 

        return contrast_value
    feature.append(contrast(glmc_maxtrix))

    # entropy
    glmc_flatten =  glmc_maxtrix.flatten()
    entropy_value = np.sum( [(-1 * i* math.log(i,2)) for i in glmc_flatten if i !=0])
    feature.append(entropy_value)

    # đồng nhất lớn nhất khi chỉ có 1 màu
    uniformity_value = np.sum( [i**2 for i in glmc_flatten ])
    feature.append(uniformity_value)

    # mean đo xs trung bình
    mean_value = np.sum([i for i in glmc_flatten]) / (glmc_maxtrix.shape[0] * glmc_maxtrix.shape[1])
    feature.append(mean_value)

    # độ lệch chuẩn 
    n = glmc_maxtrix.shape[0]
    std_value = np.sqrt((1 / n**2)* np.sum( [(i - mean_value)**2 for i in glmc_flatten ])   )
    feature.append(std_value)

    # correclation
    mean_i = np.mean(glmc_maxtrix, axis=1)
    std_i = np.std(glmc_maxtrix, axis = 1)
    # print(std_i)

    corr = []

    for k  in range(1, len(mean_i)):
        # for i in rang(glmc_maxtrix.shape[1]):
        cov_o_k =  np.sum([((glmc_maxtrix[0, i] - mean_i[0])*(glmc_maxtrix[k, i] - mean_i[k])) for i in range(glmc_maxtrix.shape[1])])
        cov_o_k = cov_o_k / len(mean_i)
        # TH độ lệch chuẩn bằng 0 thì như thế nào
        if (std_i[0] *  std_i[k]) != 0: 
            corr_o_k = cov_o_k / (std_i[0] *  std_i[k])
            corr.append(corr_o_k)
        else:
            if cov_o_k > 0:
                corr_o_k = 1 
                corr.append(corr_o_k)
            elif cov_o_k == 0:
                corr_o_k = 0 
                corr.append(corr_o_k)
            else:
                corr_o_k = -1
                corr.append(corr_o_k)
    feature += corr
    # print('Corr_cal:', np.sum(corr))
    glcm_0 = np.expand_dims(glmc_maxtrix, axis=-1)
    glcm_0 = np.expand_dims(glcm_0 , axis=-1)
    feature.append(greycoprops(glcm_0, prop='correlation')[0][0])
 
    return feature


def glmc(image = None, n_bins = 16,distance = 40, n_neighbors = 8):

    '''
    Tính toán ma trận glmc cho ảnh đầu vào

    Input:

        - image: ảnh đầu vào RGB image

        - distance: khoảng cách tìm hàng xóm láng giềng

        - n_neighbors: số hướng tìm hàng xóm của điểm p bất kì

    Output: 

        - glmc_maxtrix

        - dictionary color encode
    '''
    if image is None:

        raise Exception("Input error!")

    mi, ma =  0, 255

    # mã hóa
    bins = np.linspace(mi, ma+1, n_bins+1)
    tha = np.digitize(image, bins) - 1 # encode bins[i] < img <= bins[i+1]

    if image.ndim == 2:
        glmc_maxtrix =  glmc_cal(tha, distance=distance)
        return statistical_glmc(glmc_maxtrix)

    if image.ndim == 3:
        feature1 =  tha[:,:, 0]
        feature2 =  tha[:,:, 1]
        feature3 =  tha[:,:, 2]
        glmc_maxtrix1 = glmc_cal(feature1, distance=distance)
        glmc_maxtrix2 = glmc_cal(feature2, distance=distance)
        glmc_maxtrix3 = glmc_cal(feature3, distance=distance)

        return statistical_glmc(glmc_maxtrix1) + statistical_glmc(glmc_maxtrix2) + statistical_glmc(glmc_maxtrix3)

if __name__ == "__main__":

    img = np.array([[0, 0, 0, 1, 2], [1, 1, 0, 1, 1]])
    
    print(glmc_cal(img, 3, 1))



