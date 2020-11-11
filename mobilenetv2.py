import math
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Add, ZeroPadding2D, GlobalAveragePooling2D, Dropout, Dense
from keras.layers import MaxPooling2D, Activation, DepthwiseConv2D, Input, GlobalMaxPooling2D
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.utils.data_utils import get_file

BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')

################################################################
#                   MobileNetV2——激活函数                       #
################################################################
def relu6(x):
    return K.relu(x, max_value=6)


################################################################
#                    计算padding的大小                          #
################################################################
def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


################################################################
#        使其结果可以被8整除，因为使用到了膨胀系数α                 #
################################################################
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


################################################################
#                    MobileNetV2——卷积块                       #
################################################################
def _conv_block(inputs, alpha=1,strides=(2, 2)):
    # 卷积 + 正则化 + 激活函数(relu6)
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(inputs, 3),
                      name='Conv1_pad')(inputs)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=strides,
               padding='valid',
               use_bias=False,
               name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999,
                           name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)
    return x




################################################################
#                    MobileNetV2——倒残差块                      #
################################################################
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    # (卷积+正则化+relu6激活)+(可分离卷积+正则化+relu6激活)+(卷积+正则化+relu激活)
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    # part1 升高维度 = expansion * in_channels
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels,
                   kernel_size=1,
                   padding='same',
                   use_bias=False,
                   activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3,
                               momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                          name=prefix + 'pad')(x)

    # part2 可分离卷积
    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        activation=None,
                        use_bias=False,
                        padding='same' if stride == 1 else 'valid',
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # part3 降低维度 = pointwise_filters
    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name=prefix + 'project')(x)

    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


################################################################
#                   MobileNet V2的网络部分                      #
################################################################
def mobilenet_encoder(input_height=224 ,
                      input_width=224,
                      alpha=1,
                      ):
    img_input = Input(shape=(input_height, input_width, 3))


    # stem部分
    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, alpha=1, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=1, block_id=0)
    d1 = x
    print('d1:',d1.shape)


    # 112,112,64 -> 56,56,128
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                            expansion=6, block_id=2)
    d2 = x
    print('d2',d2.shape)


    # 56,56,128 -> 28,28,256
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1,
                            expansion=6, block_id=5)
    d3 = x
    print('d3',d3.shape)


    # 28,28,256 -> 14,14,512
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                            expansion=6, block_id=9)
    d4 = x
    print('d4',d4.shape)


    # 14,14,512 -> 14,14,1024
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=1,
                            expansion=6, block_id=12)
    # 14,14,96 -> 7,7,160
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=1,
                            expansion=6, block_id=15)
    d5 = x
    print('d5',d5.shape)

    return img_input, [d1, d2, d3, d4, d5]

    # # 7,7,160 -> 7,7,320
    # x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
    #                         expansion=6, block_id=16)

    # if alpha > 1.0:
    #     last_block_filters = _make_divisible(1280 * alpha, 8)
    # else:
    #     last_block_filters = 1280
    #
    # # 7,7,320 -> 7,7,1280
    # x = Conv2D(last_block_filters,
    #            kernel_size=1,
    #            use_bias=False,
    #            name='Conv_1')(x)
    # x = BatchNormalization(epsilon=1e-3,
    #                        momentum=0.999,
    #                        name='Conv_1_bn')(x)
    # x = Activation(relu6, name='out_relu')(x)
    #
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(classes, activation='softmax',
    #           use_bias=True, name='Logits')(x)
    #
    # inputs = img_input
    #
    # model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))
    #
    # # Load weights.


    # return model
# if __name__ == "__main__":
#     model = mobilenet_encoder()
#     model.summary()


