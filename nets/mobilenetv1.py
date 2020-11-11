from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, Conv2D
import keras
from keras import backend as K

IMAGE_ORDERING = 'channels_last'


################################################################
#                   MobileNetV1——激活函数                       #
################################################################
def relu6(x):
    # 设定max_value=6，鼓励学习稀疏性
    return K.relu(x, max_value=6)


################################################################
#                   MobileNetV1——卷积模块                       #
################################################################
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    # 卷积 + 正则化 + 激活函数(relu6)
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    x = Conv2D(filters, kernel,
               data_format=IMAGE_ORDERING,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


################################################################
#                   MobileNetV1——深度可分离卷积模块              #
################################################################
def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          alpha=1, # 升维、降维控制系数
                          depth_multiplier=1, strides=(1, 1),
                          block_id=1):
    # (深度卷积 + 正则化 +激活) + (卷积 + 正则化 + 激活)
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        data_format=IMAGE_ORDERING,
                        #通道数 = filterss_in * depth_multiplier
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(axis=channel_axis,name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

    return x


################################################################
#                   MobileNet V1的网络部分                      #
################################################################
def mobilenet_encoder(input_height=224,
                      input_width=224,
                      depth_multiplier=1,
                      ):
    img_input = Input(shape=(input_height, input_width, 3))

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))


    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, strides=(1, 1),block_id=1)
    d1 = x
    print('d1:',d1.shape)


    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(1, 1), block_id=3)
    d2 = x
    print('d2:',d2.shape)


    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(1, 1), block_id=5)
    d3 = x
    print('d3:',d3.shape)


    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(1, 1), block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(1, 1), block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(1, 1), block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(1, 1), block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(1, 1), block_id=11)
    d4 = x
    print('d4:',d4.shape)




    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(1, 1), block_id=13)
    d5 = x
    print('d5:', d5.shape)


    return img_input, [d1, d2, d3, d4, d5]
#
#     # 7,7,1024 -> 1,1,1024
#     x = GlobalAveragePooling2D()(x) # 类似于全连接层
#     x = Reshape((1, 1, 1024), name='reshape_1')(x)
#     x = Dropout(dropout, name='dropout')(x)
#     x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
#     x = Activation('softmax', name='act_softmax')(x)
#     x = Reshape((classes,), name='reshape_2')(x)
#
#
#     model = Model(img_input, x, name='mobilenet_1_0_224_tf')
#
#     return model
#
# if __name__ == "__main__":
#     model = mobilenet_encoder()
#     model.summary()