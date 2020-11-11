import os
from keras.models import load_model
import cv2
from keras import backend as K
import numpy as np
import skimage.io as io
import skimage.transform as trans

def relu6(x):
    return K.relu(x, max_value=6)


def testGenerator(test_path,target_size = (416,416,3),flag_multi_class = False,as_gray = True):
    imgs = os.listdir(test_path)
    for im in imgs:
        # img = io.imread(os.path.join(test_path,im),as_gray = as_gray) # (416, 416)
        img = io.imread(os.path.join(test_path,im))
        img = img / 255.0
        img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape) #  (1, 416, 416 ,3)
        # print(img.shape)

        yield img


def saveResult(save_path,npyfile,names,num_class = 2):

    for i,item in enumerate(npyfile):

        img = item[:,:,0] # 去掉最后一个维度(416, 416, 1)-->(416, 416)

        img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_NEAREST)
        #print(np.max(img), np.min(img))
        img[img>0.5] = 1
        img[img<=0.5] = 0
        print(np.max(img), np.min(img))
        io.imsave(os.path.join(save_path,"%s"%names[i]),img)


if __name__ == '__main__':
    test_img_path = './dataset/build/test/images/'
    save_path = './dataset/result/'
    names = os.listdir(test_img_path)
    test_number = len(names)
    batch_size = 4
    model = load_model('./logs/build_009.h5',custom_objects={'relu6':relu6})
    testGene = testGenerator(test_img_path)
    results = model.predict_generator(testGene,
                                      steps=10,
                                      # steps = test_number,
                                      verbose=1)
    saveResult(save_path, results, names)