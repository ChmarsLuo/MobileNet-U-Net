from keras.models import *
from keras.layers import *
from nets.mobilenetv3_large import mobilenet_encoder
from keras.activations import *


IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def _unet(n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=416):

	img_input , levels = encoder(input_height=input_height ,  input_width=input_width)
	[d1 , d2 , d3 , d4, d5 ] = levels

	o = d5
	# 14,14,1024
	o = Conv2D(1024, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Dropout(0.5)(o)

	# 28,28,512
	o = UpSampling2D((2,2), data_format=IMAGE_ORDERING)(o)
	o = Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = concatenate([o, d4], axis=MERGE_AXIS)
	o = Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Dropout(0.5)(o)

	# 52,52,256
	o = UpSampling2D((2,2), data_format=IMAGE_ORDERING)(o)
	o = Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = concatenate([o,d3],axis=MERGE_AXIS)
	o = Conv2D(256 , (3, 3), padding='same' , data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Dropout(0.5)(o)


	# 104,104,128
	o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
	o = Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = concatenate([o, d2], axis=MERGE_AXIS)
	o = Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Dropout(0.5)(o)

	# 208,208,64
	o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
	o = Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = concatenate([o, d1], axis=MERGE_AXIS)
	o = Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	o = Dropout(0.5)(o)

	o = Reshape(target_shape=(input_height,input_height,-1))(o)

	o = Conv2D(n_classes, 1, padding='same', activation='sigmoid')(o)

	model = Model(img_input,o)


	return model



def mobilenet_unet(n_classes ,  input_height=224, input_width=224 , encoder_level=3):

	model =  _unet(n_classes , mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_unet"


	return model


# if __name__ == "__main__":
#     model = _unet(n_classes=2, encoder=mobilenet_encoder)
#     model.summary()
