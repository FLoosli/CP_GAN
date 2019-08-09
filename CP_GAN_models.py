from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, AveragePooling2D, ReLU,\
    LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import concatenate
from keras import backend as K


# Input shape of new background
INPUT_BACKGROUND = (72, 72, 3)
# Input shape of cropped image
INPUT_CROP = (128, 128, 3)
# Input shape of discriminator
INPUT_DISC = (72, 72, 3)
# Number of filters used by the convolution
FILTERS = 64
# At which axis to concatenate and to normalize with batch norm
AXIS = -1
# Size of padding boarder
PADDING = 4


# metrics
def conf_metric(y_true, y_pred):
    return K.mean(K.minimum(y_pred, 1 - y_pred))


def min_metric(y_true, y_pred):
    return K.mean(K.maximum((-1 * K.mean(y_pred, axis=[1, 2, 3]) + 0.2), 0))


# The computed mask should not be smaller than 20% of the bbox
def min_mask_loss(y_true, y_pred):
    return K.mean(K.maximum((-1 * K.mean(y_pred, axis=[1, 2, 3]) + 0.2), 0))


# mask should make confident predictions, each pixel should be either 0 or 1
def confidents_loss(y_true, y_pred):
    return K.mean(K.minimum(y_pred, 1-y_pred))


def mask_loss(y_true, y_pred):
    return confidents_loss(y_true, y_pred) + min_mask_loss(y_true, y_pred)


# Cut and paste function, custom lambda layer
# Creates a new image from a given image, it's mask and given background
def cut_and_paste(x):

    background = x[0]
    crop = x[1]
    mask = x[2]

    mask_to_paste = mask * crop
    mask_to_paste = K.spatial_2d_padding(mask_to_paste,
                                         padding=((PADDING*2, 0), (PADDING, PADDING)),
                                         data_format='channels_last')

    prep_mask = K.spatial_2d_padding(mask,
                                     padding=((PADDING*2, 0), (PADDING, PADDING)),
                                     data_format='channels_last')

    inverted_mask = 1 - prep_mask

    cp_img = (background * inverted_mask) + mask_to_paste

    return cp_img


# Double convolution layer build into the U-NET
def double_conv_layer(inputs, filters):

    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    conv = BatchNormalization(axis=AXIS, momentum=0.5)(conv)
    conv = ReLU()(conv)
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv)
    conv = BatchNormalization(axis=AXIS, momentum=0.5)(conv)
    conv = ReLU()(conv)

    return conv


# Architecture of generator, U-NET
# Input shapes:
# background_img: 72*72*3
# crop: 128*128*3
# Output shapes:
# new_img: 72*72*3
# mask: 64*64*3
def unet_gen_for_gan():

    background = Input(shape=INPUT_BACKGROUND)
    crop = Input(shape=INPUT_CROP)

    conv_256 = double_conv_layer(inputs=crop, filters=FILTERS)
    pool_128 = MaxPooling2D(pool_size=(2, 2))(conv_256)

    conv_128 = double_conv_layer(inputs=pool_128, filters=2*FILTERS)
    pool_64 = MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = double_conv_layer(inputs=pool_64, filters=4*FILTERS)
    pool_32 = MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = double_conv_layer(inputs=pool_32, filters=8*FILTERS)
    pool_16 = MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = double_conv_layer(inputs=pool_16, filters=16*FILTERS)
    pool_8 = MaxPooling2D(pool_size=(2, 2))(conv_16)

    conv_8 = double_conv_layer(inputs=pool_8, filters=32*FILTERS)

    up_16 = concatenate([UpSampling2D(size=(2, 2))(conv_8), conv_16], axis=AXIS)
    up_conv_16 = double_conv_layer(inputs=up_16, filters=16*FILTERS)

    up_32 = concatenate([UpSampling2D(size=(2, 2))(up_conv_16), conv_32], axis=AXIS)
    up_conv_32 = double_conv_layer(inputs=up_32, filters=8*FILTERS)

    up_64 = concatenate([UpSampling2D(size=(2, 2))(up_conv_32), conv_64], axis=AXIS)
    up_conv_64 = double_conv_layer(inputs=up_64, filters=4*FILTERS)

    up_128 = concatenate([UpSampling2D(size=(2, 2))(up_conv_64), conv_128], axis=AXIS)
    up_conv_128 = double_conv_layer(inputs=up_128, filters=2*FILTERS)

    up_256 = concatenate([UpSampling2D(size=(2, 2))(up_conv_128), conv_256], axis=AXIS)
    up_conv_256 = double_conv_layer(inputs=up_256, filters=FILTERS)

    conv_final = Conv2D(filters=1, kernel_size=(2, 2), strides=(2, 2))(up_conv_256)
    mask = Activation('sigmoid')(conv_final)

    org_crop = AveragePooling2D(pool_size=(2, 2), padding='valid')(crop)

    cut_paste_layer = Lambda(cut_and_paste, output_shape=None)
    new_img = cut_paste_layer([background, org_crop, mask])

    model = Model([background, crop], [new_img, mask], name='Gen')

    model.summary()

    return model


# Architecture of generator for supervised learning, U-Net
# Inputshape: 128*128*3
# Outputshape: 64*64*3
def super_unet_gen():

    crop = Input(shape=INPUT_CROP)

    conv_256 = double_conv_layer(inputs=crop, filters=FILTERS)
    pool_128 = MaxPooling2D(pool_size=(2, 2))(conv_256)

    conv_128 = double_conv_layer(inputs=pool_128, filters=2 * FILTERS)
    pool_64 = MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = double_conv_layer(inputs=pool_64, filters=4 * FILTERS)
    pool_32 = MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = double_conv_layer(inputs=pool_32, filters=8 * FILTERS)
    pool_16 = MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = double_conv_layer(inputs=pool_16, filters=16 * FILTERS)
    pool_8 = MaxPooling2D(pool_size=(2, 2))(conv_16)

    conv_8 = double_conv_layer(inputs=pool_8, filters=32 * FILTERS)

    up_16 = concatenate([UpSampling2D(size=(2, 2))(conv_8), conv_16], axis=AXIS)
    up_conv_16 = double_conv_layer(inputs=up_16, filters=16 * FILTERS)

    up_32 = concatenate([UpSampling2D(size=(2, 2))(up_conv_16), conv_32], axis=AXIS)
    up_conv_32 = double_conv_layer(inputs=up_32, filters=8 * FILTERS)

    up_64 = concatenate([UpSampling2D(size=(2, 2))(up_conv_32), conv_64], axis=AXIS)
    up_conv_64 = double_conv_layer(inputs=up_64, filters=4 * FILTERS)

    up_128 = concatenate([UpSampling2D(size=(2, 2))(up_conv_64), conv_128], axis=AXIS)
    up_conv_128 = double_conv_layer(inputs=up_128, filters=2 * FILTERS)

    up_256 = concatenate([UpSampling2D(size=(2, 2))(up_conv_128), conv_256], axis=AXIS)
    up_conv_256 = double_conv_layer(inputs=up_256, filters=FILTERS)

    conv_final = Conv2D(filters=1, kernel_size=(4, 4), strides=(4, 4))(up_conv_256)
    mask = Activation('sigmoid')(conv_final)

    model = Model(crop, mask, name='Super_gen')

    model.summary()

    return model


# Discriminator architecture according to the paper
# Input shape: 72*72*3
# Output shape: scalar
def paper_disc():

    img = Input(shape=INPUT_DISC)

    conv_1 = Conv2D(filters=FILTERS, kernel_size=3, strides=1, padding='valid')(img)
    batch_norm_1 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_1)
    lrelu_1 = LeakyReLU(alpha=0.2)(batch_norm_1)

    conv_2 = Conv2D(filters=2*FILTERS, kernel_size=3, strides=2, padding='valid')(lrelu_1)
    batch_norm_2 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_2)
    lrelu_2 = LeakyReLU(alpha=0.2)(batch_norm_2)

    conv_3 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=2, padding='valid')(lrelu_2)
    batch_norm_3 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_3)
    lrelu_3 = LeakyReLU(alpha=0.2)(batch_norm_3)

    conv_4 = Conv2D(filters=8*FILTERS, kernel_size=3, strides=2, padding='valid')(lrelu_3)
    batch_norm_4 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_4)
    lrelu_4 = LeakyReLU(alpha=0.2)(batch_norm_4)

    flatten = Flatten()(lrelu_4)

    dense = Dense(units=1)(flatten)

    valid = Activation('sigmoid')(dense)

    model = Model(img, valid, name='Disc')

    model.summary()

    return model


# Generator architecture according to the paper
# Input shapes:
# background_img: 72*72*3
# crop: 128*128*3
# Output shapes:
# new_img: 72*72*3
# mask: 64*64*3
def paper_gen():

    background = Input(shape=INPUT_BACKGROUND)
    crop = Input(shape=INPUT_CROP)

    conv_1 = Conv2D(filters=FILTERS, kernel_size=1, strides=2, padding='same')(crop)
    batch_norm_1 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_1)
    relu_1 = ReLU()(batch_norm_1)

    conv_2 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=1, padding='same')(relu_1)
    batch_norm_2 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_2)
    relu_2 = ReLU()(batch_norm_2)

    up_1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(relu_2)

    conv_3 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=1, padding='same')(up_1)
    batch_norm_3 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_3)
    relu_3 = ReLU()(batch_norm_3)

    up_2 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(relu_3)

    conv_4 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=2, padding='same')(up_2)
    batch_norm_4 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_4)
    relu_4 = ReLU()(batch_norm_4)

    conv_5 = Conv2D(filters=1, kernel_size=3, strides=2, padding='same')(relu_4)
    mask = Activation('sigmoid')(conv_5)

    org_crop = AveragePooling2D(pool_size=(2, 2), padding='valid')(crop)

    cut_paste_layer = Lambda(cut_and_paste, output_shape=None)
    new_img = cut_paste_layer([background, org_crop, mask])

    model = Model([background, crop], [new_img, mask], name='Gen')

    model.summary()

    return model


# Generator architecture according to the paper, supervised
# Input shape: 128*128*3
# Output shape: 64*64*3
def super_paper_gen():

    crop = Input(shape=INPUT_CROP)

    conv_1 = Conv2D(filters=FILTERS, kernel_size=1, strides=1, padding='same')(crop)
    batch_norm_1 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_1)
    relu_1 = ReLU()(batch_norm_1)

    conv_2 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=2, padding='same')(relu_1)
    batch_norm_2 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_2)
    relu_2 = ReLU()(batch_norm_2)

    up_1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(relu_2)

    conv_3 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=2, padding='same')(up_1)
    batch_norm_3 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_3)
    relu_3 = ReLU()(batch_norm_3)

    up_2 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(relu_3)

    conv_4 = Conv2D(filters=4*FILTERS, kernel_size=3, strides=2, padding='same')(up_2)
    batch_norm_4 = BatchNormalization(axis=AXIS, momentum=0.5)(conv_4)
    relu_4 = ReLU()(batch_norm_4)

    conv_5 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(relu_4)
    mask = Activation('sigmoid')(conv_5)

    model = Model([crop], [mask, mask], name='Paper_gen')

    model.summary()

    return model
