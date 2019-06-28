import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import shutil
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model, load_model , Sequential
from keras import optimizers
from keras import models
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation, Dropout, Add, AveragePooling2D, InputLayer
from keras.initializers import glorot_uniform
from keras.applications.vgg16 import VGG16
from collections import defaultdict, OrderedDict
from keras.layers import  concatenate, UpSampling2D, Cropping2D


vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (512,512, 3))
vgg.trainable = False
layer_size_dict = defaultdict(list)
inputs = []

for lay_idx, c_layer in enumerate(vgg.layers):
    if not c_layer.__class__.__name__ == 'InputLayer':
        layer_size_dict[c_layer.get_output_shape_at(0)[1:4]] += [c_layer]
    else:
        inputs+= [c_layer]

#freeze dict
layer_size_dict = OrderedDict(layer_size_dict.items())
for k, v in layer_size_dict.items():
    print(k, [w.__class__.__name__ for w in v])

pretrained_encoder = Model(inputs = vgg.get_input_at(0),outputs = [v[-1].get_output_at(0) for k, v in layer_size_dict.items()])

pretrained_encoder.trainable = False

x_wid, y_wid = (256, 256)
in_t0 = Input((512, 512,3), name = 'T0_image')
wrap_encoder = lambda i_layer : {k: v for k, v in zip(layer_size_dict.keys(), pretrained_encoder(i_layer))}
t0_outputs = wrap_encoder(in_t0)
lay_dims = sorted(t0_outputs.keys(), key = lambda x :x[0])
print(lay_dims)
skip_layers = 2
last_layer = None
# for k in lay_dims[0:skip_layers+1]:
#     cur_layer = t0_outputs[k]
#     channel_count = cur_layer._keras_shape[-1]
#     print(channel_count)
#     cur_layer = Conv2D(2, kernel_size = (1,1), padding = 'same', activation = 'linear')(cur_layer)
#     cur_layer = BatchNormalization()(cur_layer)
#     cur_layer = Activation('relu')(cur_layer)

#     if last_layer is None:
#         x = cur_layer
#     else:
#         last_channel_count = last_layer._keras_shape[-1]
#         print(last_channel_count)
#         x = Conv2D(2, kernel_size = (1,1), padding = 'same', activation = 'sigmoid')(last_layer)
#         x = UpSampling2D((2,2))(x)
#         x = Add()([last_layer,x])
#     last_layer = x


cur_layer = t0_outputs[lay_dims[0]]
channel_count = cur_layer._keras_shape[-1]
print(channel_count)
cur_layer = Conv2D(2, kernel_size = (1,1), padding = 'same', activation = 'linear')(cur_layer)
cur_layer = BatchNormalization()(cur_layer)
cur_layer = Activation('relu')(cur_layer)
x = cur_layer
last_layer =x
print(last_layer)

cur_layer = t0_outputs[lay_dims[1]]
channel_count = cur_layer._keras_shape[-1]
print(channel_count)
cur_layer = Conv2D(2, kernel_size = (1,1), padding = 'same', activation = 'linear')(cur_layer)
cur_layer = BatchNormalization()(cur_layer)
cur_layer = Activation('relu')(cur_layer)
last_layer = UpSampling2D((2,2), interpolation = 'bilinear')(last_layer)
x = Add()([last_layer, cur_layer])
last_layer = x

cur_layer = t0_outputs[lay_dims[2]]
channel_count = cur_layer._keras_shape[-1]
print(channel_count)
cur_layer = Conv2D(2, kernel_size = (1,1), padding = 'same', activation = 'linear')(cur_layer)
cur_layer = BatchNormalization()(cur_layer)
cur_layer = Activation('relu')(cur_layer)
last_channel_count = last_layer._keras_shape[-1]
last_layer = UpSampling2D((2,2), interpolation = 'bilinear')(last_layer)
x = Add()([last_layer, cur_layer])
last_layer = x

x = UpSampling2D((8,8), interpolation = 'bilinear')(last_layer)
last_layer = x
final_output = Conv2D(1, kernel_size=(1,1), padding = 'same', activation = 'sigmoid')(last_layer)
crop_size = 20
final_output = Cropping2D((crop_size, crop_size))(final_output)
final_output = ZeroPadding2D((crop_size, crop_size))(final_output)
unet_model = Model(inputs = [in_t0],
                  outputs = [final_output])
unet_model.summary()