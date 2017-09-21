from keras.layers import Input, Convolution2D, BatchNormalization, \
    Activation, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, merge
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from model.helpers import dice_coef, dice_coef_loss


class Unet:
    def __init__(self, model_params):
        self.model_params = model_params
        self.n_classes = self.model_params["n_classes"]

    def activation_layer(self, x):
        activation_type = self.model_params["activation"]
        if activation_type == 'relu':
            return Activation("relu")(x)
        elif activation_type == 'prelu':
            return PReLU()(x)
        elif activation_type == 'lrelu':
            return LeakyReLU()(x)

    def conv_bn_relu(self, x, nb_filter, kernel_dim1, kernel_dim2):
        conv = Convolution2D(nb_filter, (kernel_dim1, kernel_dim2),
                             kernel_initializer='he_normal',
                             activation=None,
                             padding='same',
                             kernel_regularizer=l2(self.model_params["l2_penalty"]),
                             bias_regularizer=None,
                             activity_regularizer=None)(x)
        dropout = Dropout(self.model_params["dropout_prob"])(conv)
        norm = BatchNormalization(axis=-1)(dropout)
        x = self.activation_layer(norm)
        return x

    def root_1(self, x):
        filter_factor = self.model_params["filter_factor"]

        conv_0 = self.conv_bn_relu(x, filter_factor * 8, 3, 3)
        conv_0 = self.conv_bn_relu(conv_0, filter_factor * 8, 3, 3)
        pool_0 = MaxPooling2D(pool_size=(2, 2))(conv_0)

        conv0 = self.conv_bn_relu(pool_0, filter_factor * 8, 3, 3)
        conv0 = self.conv_bn_relu(conv0, filter_factor * 8, 3, 3)
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

        conv1 = self.conv_bn_relu(pool0, filter_factor * 16, 3, 3)
        conv1 = self.conv_bn_relu(conv1, filter_factor * 16, 3, 3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_bn_relu(pool1, filter_factor * 32, 3, 3)
        conv2 = self.conv_bn_relu(conv2, filter_factor * 32, 3, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_bn_relu(pool2, filter_factor * 64, 3, 3)
        conv3 = self.conv_bn_relu(conv3, filter_factor * 64, 3, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_bn_relu(pool3, filter_factor * 128, 3, 3)
        conv4 = self.conv_bn_relu(conv4, filter_factor * 128, 3, 3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.conv_bn_relu(pool4, filter_factor * 256, 3, 3)
        conv5 = self.conv_bn_relu(conv5, filter_factor * 256, 3, 3)

        conv4_cropped = Cropping2D(((4, 4), (4, 4)))(conv4)
        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4_cropped], mode='concat', concat_axis=-1)
        conv6 = self.conv_bn_relu(up6, filter_factor * 128, 3, 3)
        conv6 = self.conv_bn_relu(conv6, filter_factor * 128, 3, 3)

        conv3_cropped = Cropping2D(((16, 16), (16, 16)))(conv3)
        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3_cropped], mode='concat', concat_axis=-1)
        conv7 = self.conv_bn_relu(up7, filter_factor * 64, 3, 3)
        conv7 = self.conv_bn_relu(conv7, filter_factor * 64, 3, 3)

        conv2_cropped = Cropping2D(((40, 40), (40, 40)))(conv2)
        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2_cropped], mode='concat', concat_axis=-1)
        conv8 = self.conv_bn_relu(up8, filter_factor * 32, 3, 3)
        conv8 = self.conv_bn_relu(conv8, filter_factor * 32, 3, 3)

        conv1_cropped = Cropping2D(((88, 88), (88, 88)))(conv1)
        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1_cropped], mode='concat', concat_axis=-1)
        conv9 = self.conv_bn_relu(up9, filter_factor * 16, 3, 3)
        conv9 = self.conv_bn_relu(conv9, filter_factor * 16, 3, 3)

        conv0_cropped = Cropping2D(((184, 184), (184, 184)))(conv0)
        up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv0_cropped], mode='concat', concat_axis=-1)
        conv10 = self.conv_bn_relu(up10, filter_factor * 8, 3, 3)
        conv10 = self.conv_bn_relu(conv10, filter_factor * 8, 3, 3)

        conv_0_cropped = Cropping2D(((376, 376), (376, 376)))(conv_0)
        up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv_0_cropped], mode='concat', concat_axis=-1)
        conv11 = self.conv_bn_relu(up11, filter_factor * 8, 3, 3)
        conv11 = self.conv_bn_relu(conv11, filter_factor * 8, 3, 3)

        conv12 = Convolution2D(self.n_classes, (1, 1), activation='sigmoid')(conv11)
        return conv12

    def model(self):
        image = Input((200, 200, 1))
        softmax_output = self.root_1(image)
        model = Model(inputs=image, outputs=softmax_output)

        model.compile(optimizer=Adam(lr=self.model_params["adam_lr"]),
                      loss=dice_coef_loss, metrics=[dice_coef])
        return model
