from time import gmtime, strftime
import numpy as np
import base64
def random_name():
    time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    time = str(time) + '.jpg'
    return time

def get_ids_top(arr_scores, top = 3):

    ids_sort = np.argsort(arr_scores)

    return [ids_sort[-1], ids_sort[-2], ids_sort[-3]]

class Params:

    model_dir = 'model'
    imgWidth  = 256
    imgHeight = 256
    CLASSES = { 0: 'Pepper__bell___Bacterial_spot', 
                1: 'Pepper__bell___healthy', 
                2: 'Potato___Early_blight', 
                3: 'Potato___Late_blight', 
                4: 'Potato___healthy', 
                5: 'Tomato_Bacterial_spot', 
                6: 'Tomato_Early_blight', 
                7: 'Tomato_Late_blight', 
                8: 'Tomato_Leaf_Mold', 
                9: 'Tomato_Septoria_leaf_spot', 
                10: 'Tomato_Spider_mites_Two_spotted_spider_mite', 
                11: 'Tomato__Target_Spot', 
                12: 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
                13: 'Tomato__Tomato_mosaic_virus', 
                14: 'Tomato_healthy'}
    

def covert_img2base64(img_path):

    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return 'data:image/jpg;base64,' + str(encoded_string).split("'")[1]



if __name__ == '__main__':

    path = 'F:\Current Project\Landscape Search\dataset/temple/temple_97.jpg'

    print(covert_img2base64(path))
