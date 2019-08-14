from keras.layers.pooling import _Pooling2D
from keras import backend as K


class UniPooling2D(_Pooling2D):
    '''This class implements pooling layer with easily
    variable method of pooling. Tensorflow has only two
    built-in pooling methods, MAX and AVERAGE, while here
    any function, implementing mapping from NxM image patch
    to a scalar is valid, nevertheless the most convenient
    are function tf.math.reduce* which could be used directly'''

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, reduce_function=K.tf.math.reduce_mean, **kwargs):
        ''' majority of parameters are inherited from keras
        _Pooling2D class.
        for padding only 'valid' and 'same' is supported.
        reduce_function - function which implements mapping
        supported values in the current version of tensorflow:
            K.tf.math.reduce_logsumexp
            K.tf.math.reduce_max
            K.tf.math.reduce_meam
            K.tf.math.reduce_min
            K.tf.math.reduce_prod
            K.tf.math.reduce_std - will not work if negative values are present
            K.tf.math.reduce_sum
            K.tf.math.reduce_variance
        '''

        super().__init__(pool_size, strides, padding,
                         data_format, **kwargs)
        self.reduce_function = reduce_function

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        padding = padding.upper()  # keras uses downcase, while tensorflow requires uppercase
        if len(pool_size) != 2:
            raise ValueError("Pool size must be two-dimensional")

        # images initially have shape [batch_size, rows, cols, depth]
        size = [1, pool_size[0], pool_size[1], 1]
        strides_4d = [1, strides[0], strides[1], 1]
        depth = inputs.get_shape()[-1]

        # split the image into patches, resulting in shape [batch size, patch_rows, patch_columns, patch_size*depth]
        # the last dimension has structure [patch1_1, patch1_2, patch1_3..., patch2_1, patch2_2, patch2_3...],
        # where patch1_* are patches from first channel, patch2_* are from second, so on
        image_patches = K.tf.extract_image_patches(images=inputs, ksizes=size, strides=strides_4d, rates=[1, 1, 1, 1], padding=padding)

        # now split the last axis to get shape [batch size, patch_rows, patch_columns, patch_size, depth]
        image_patches = K.tf.stack(K.tf.split(image_patches, depth, axis=3), axis=4)

        # reduce the patch_size dimension to obtain flat pooled image with
        # dimensionality [batch size, patch_rows, patch_columns, 1, depth] and get finally
        # get read of the unnecessary dimension
        output = self.reduce_function(image_patches, axis=3, keepdims=True)
        output = K.tf.squeeze(output, axis=3)
        return output
