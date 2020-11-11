from __future__ import print_function
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau,TensorBoard,EarlyStopping,CSVLogger
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras import backend as K
from nets.unet import mobilenet_unet
import keras
import os

############################################
#             数据处理                      #
############################################
def adjustData(img, mask, flag_multi_class, num_class):
    if(flag_multi_class):
        #img = img /255.
        mask = mask[:, :, :, 0] if (len(mask.shape)==4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,)) # (512, 512, 3, 2)
        print(new_mask.shape)
        for i in range(num_class):
            # index = np.where(mask == i) # 元组
            # index_mask = (index[0], index[1], index[2], np.zeros(len(index[0]), dtype=np.int64) + i) if (
            #             len(mask.shape) == 4) else (index[0], index[1], np.zeros(len(index[0]), dtype=np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1 # 将平面的mask的每类，都单独变成一层
            new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                             new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
            mask = new_mask

    elif(np.max(img) > 1):
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] =1

    return (img,mask)
# adjustData(img, img, 1, 2 )

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,
                    image_save_data_dir, mask_save_data_dir, image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image_",mask_save_prefix  = "mask_",
                    flag_multi_class = False,num_class = 2,target_size = (416,416),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    mask_color_mode = "grayscale"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = image_save_data_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = mask_save_data_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator) # 变成一对一对的啦
    for (img,mask) in train_generator:

        img,mask = adjustData(img,mask, flag_multi_class, num_class)
        # print(img.shape) # (4, 416, 416, 3)
        # print(mask.shape) # 输出的是这个样子呀(4, 416, 416, 1)

        yield (img,mask)

if __name__ == '__main__':
    batch_size = 1
    # img_size = 256
    epochs = 30
    model = mobilenet_unet(n_classes=1, input_height=416, input_width=416)
    model.summary()
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.6/')
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
    weight_path = BASE_WEIGHT_PATH +'\\'+ model_name
    weights_path = keras.utils.get_file(model_name, weight_path)
    print(weight_path)

    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    train_im_path,train_mask_path = './dataset/build/train/images/','./dataset/build/train/labels/'
    val_im_path,val_mask_path = './dataset/build/val/images/','./dataset/build/val/labels/'
    train_set = os.listdir(train_im_path)
    val_set = os.listdir(val_im_path)
    train_number = len(train_set)
    val_number = len(val_set)

    train_root = './dataset/build/train/'
    val_root = './dataset/build/val/'
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    training_generator = trainGenerator(batch_size,train_root,'images','labels',data_gen_args,
                                        image_save_data_dir = './dataset/build_aug/train/images/',
                                        mask_save_data_dir = './dataset/build_aug/train/labels/')
    validation_generator = trainGenerator(batch_size,val_root,'images','labels',data_gen_args,
                                          image_save_data_dir='./dataset/build_aug/val/images/',
                                          mask_save_data_dir='./dataset/build_aug/val/labels/')


    model_path ="./logs/"
    model_name = 'build_{epoch:03d}.h5'
    model_file = os.path.join(model_path, model_name)
    model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # lr_reducer = LearningRateScheduler(scheduler)
    # model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=False)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.5625), cooldown=0, patience=5, min_lr=0.5e-6)
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    callable = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
                model_checkpoint,
                lr_reducer,
                CSVLogger(filename='./logs/log.csv', append=False),  # CSVLoggerb保存训练结果   好好用
                TensorBoard(log_dir='./logs/')]

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=train_number//batch_size,
                        validation_steps=val_number//batch_size,
                        use_multiprocessing=False,
                        epochs=epochs,verbose=1,
                        initial_epoch = 0,
                        callbacks=callable)




























