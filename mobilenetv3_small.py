from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Input
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras.models import Model
from keras import backend as K

alpha = 1
################################################################
#                   MobileNetV1——激活函数                       #
################################################################
def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)


################################################################
#                   MobileNetV3——激活函数                       #
################################################################
def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


################################################################
#                     激活函数的使用                             #
################################################################
def use_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x


def conv_block(inputs, filters, kernel, strides, nl):
    # 卷积 + 正则化 + 激活函数(relu6/hard_swish)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    x = use_activation(x, nl)
    return x


################################################################
#                   MobileNetV3——注意力模块                     #
################################################################
def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels / 4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x


################################################################
#                   MobileNetV3——基础模块                       #
################################################################
def bottleneck(inputs, filters, kernel, up_dim, stride, sq, nl):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    input_shape = K.int_shape(inputs)

    tchannel = int(up_dim)
    cchannel = int(alpha * filters)

    r = stride == 1 and input_shape[3] == filters
    # 1x1卷积调整通道数，通道数上升
    x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)
    # 进行3x3深度可分离卷积
    x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = use_activation(x, nl)
    # 引入注意力机制
    if sq:
        x = squeeze(x)
    # 下降通道数
    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


################################################################
#                   MobileNetV3——网络编码                       #
################################################################
def mobilenet_encoder(input_height=224,
                      input_width=224,
                      ):
    img_input = Input(shape=(input_height, input_width, 3))
    # 224,224,3 -> 112,112,64
    x = conv_block(img_input, 64, (3, 3), strides=(2, 2), nl='HS')
    d1 = x
    print('d1:',d1.shape)


    # 112,112,64 -> 56,56,128
    x = bottleneck(x, 128, (3, 3), up_dim=16, stride=2, sq=True, nl='RE')
    d2 = x
    print('d2:',d2.shape)


    # 56,56,128 -> 28,28,256
    x = bottleneck(x, 256, (3, 3), up_dim=72, stride=2, sq=False, nl='RE')
    x = bottleneck(x, 256, (3, 3), up_dim=88, stride=1, sq=False, nl='RE')
    d3 = x
    print('d3:',d3.shape)


    # 28,28,256 -> 14,14,500
    x = bottleneck(x, 500, (5, 5), up_dim=96, stride=2, sq=True, nl='HS')
    x = bottleneck(x, 500, (5, 5), up_dim=240, stride=1, sq=True, nl='HS')
    x = bottleneck(x, 500, (5, 5), up_dim=240, stride=1, sq=True, nl='HS')
    # 14,14,40 -> 14,14,512
    x = bottleneck(x, 512, (5, 5), up_dim=120, stride=1, sq=True, nl='HS')
    x = bottleneck(x, 512, (5, 5), up_dim=144, stride=1, sq=True, nl='HS')
    d4 = x
    print('d4:',d4.shape)


    # 14,14,512 -> 7,7,1024
    x = bottleneck(x, 1024, (5, 5), up_dim=288, stride=2, sq=True, nl='HS')
    x = bottleneck(x, 1024, (5, 5), up_dim=576, stride=1, sq=True, nl='HS')
    x = bottleneck(x, 1024, (5, 5), up_dim=576, stride=1, sq=True, nl='HS')
    d5 = x
    print('d5:',d5.shape)


    return img_input, [d1, d2, d3, d4, d5]
    # x = conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, 576))(x)
    #
    # x = Conv2D(1024, (1, 1), padding='same')(x)
    # x = use_activation(x, 'HS')
    #
    # x = Conv2D(n_class, (1, 1), padding='same', activation='softmax')(x)
    # x = Reshape((n_class,))(x)
    #
    # model = Model(img_input, x)
    #
    # return model


# if __name__ == "__main__":
#     model = mobilenet_encoder()
#     model.summary()


