'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, Flatten, Dense
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_Projector import Projector

from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
import tensorflow as tf
import math

def ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    '''
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    EARTH_RADIUS = tf.constant(6371000, tf.float32)  # Radius in meters of Earth
    GOOGLE_CAR_CAMERA_HEIGHT = tf.cast(3, tf.float32) # ballpark estimate of the number of meters that camera is off the ground
    MATH_PI = tf.cast(math.pi, tf.float32)
    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    def _atan2(y, x):
        """ My implementation of atan2 in tensorflow.  Returns in -pi .. pi."""
        tan = tf.atan(y / (x + 1e-8))  # this returns in -pi/2 .. pi/2

        one_map = tf.ones_like(tan)

        # correct quadrant error
        correction = tf.where(tf.less(x + 1e-8, 0.0), 3.141592653589793*one_map, 0.0*one_map)
        tan_c = tan + correction  # this returns in -pi/2 .. 3pi/2

        # bring to positive values
        correction = tf.where(tf.less(tan_c, 0.0), 2*3.141592653589793*one_map, 0.0*one_map)
        tan_zero_2pi = tan_c + correction  # this returns in 0 .. 2pi

        # make symmetric
        correction = tf.where(tf.greater(tan_zero_2pi, 3.141592653589793), -2*3.141592653589793*one_map, 0.0*one_map)
        tan_final = tan_zero_2pi + correction  # this returns in -pi .. pi
        return tan_final 


    def world_coordinates_to_streetview_pixel(lat, lng, lat1,lng1, yaw, image_width, image_height,height=0, zoom=None, object_dims=None, method=None):
        image_height = tf.constant(300, dtype=tf.float32)
        image_width = tf.constant(600, dtype=tf.float32)

        EARTH_RADIUS = tf.cast(6371000, tf.float32)  # Radius in meters of Earth
        GOOGLE_CAR_CAMERA_HEIGHT = tf.cast(3, tf.float32) # ballpark estimate of the number of meters that camera is off the ground
        MATH_PI = tf.cast(math.pi, tf.float32)
        pitch = tf.constant(0, dtype=tf.float32)
        dx1 = tf.cos((lat1)* (MATH_PI/180))
        dx11 = lng-lng1
        dxr = tf.sin(dx11 * (MATH_PI/180))
        dx = dx1 * dxr

        dy11 = tf.subtract(lat,lat1)
        dyr = tf.multiply(dy11,(MATH_PI/180))
        dy = tf.sin(dyr)
        look_at_angle = MATH_PI + _atan2(dx, dy) - yaw 

        i = 2*MATH_PI

        c = lambda x : tf.reduce_any(tf.greater(x, i))
        b = lambda x : tf.subtract(x, tf.cast(tf.greater(x, i), tf.float32)*(2*MATH_PI))
        look_at_angle = tf.while_loop(c, b, [look_at_angle])

        t = lambda x : tf.reduce_any(tf.less(x, 0))
        d = lambda x : tf.add(x, tf.cast(tf.less(x, 0), tf.float32)*(2*MATH_PI)) 
        look_at_angle = tf.while_loop(t, d, [look_at_angle])

        inner = dx*dx+dy*dy
        z = tf.multiply(tf.sqrt(tf.add(inner,1e-10)),tf.constant(6371000, tf.float32))
        # z = tf.where(tf.is_nan(z), tf.zeros_like(z), z)

        camhei_ = tf.fill(tf.shape(z), -GOOGLE_CAR_CAMERA_HEIGHT)

        x_ = tf.divide(tf.multiply(image_width,look_at_angle),(2*MATH_PI))

        y_0 = tf.divide(image_height,tf.constant(2.0, dtype=tf.float32))
        y_1 = tf.multiply(image_height,tf.subtract(_atan2(camhei_, z),pitch))
        y_2 = tf.divide(y_1,MATH_PI)
        y_ = tf.subtract(y_0,y_2)

        return x_, y_

    def streetview_pixel_to_world_coordinates(lat1,lng1, yaw, image_width, image_height, x, y):
        EARTH_RADIUS = tf.cast(6371000, tf.float32)  # Radius in meters of Earth
        GOOGLE_CAR_CAMERA_HEIGHT = tf.cast(3, tf.float32) # ballpark estimate of the number of meters that camera is off the ground
        MATH_PI = tf.cast(math.pi, tf.float32)
        pitch = float(0)
        look_at_angle = x*(2*math.pi)/image_width
        height = 0
        tilt_angle = (image_height/2-y)*math.pi/image_height+pitch
        tilt_angle = tf.cast(tilt_angle, tf.float32)
        z_ = K.minimum(np.float32(-1e-2),tilt_angle)        
        z = tf.divide((-GOOGLE_CAR_CAMERA_HEIGHT),tf.tan(z_))
        dx = tf.sin(look_at_angle-MATH_PI+yaw)*z/EARTH_RADIUS
        dy = tf.cos(look_at_angle-MATH_PI+yaw)*z/EARTH_RADIUS
        lat = lat1 + tf.asin(dy) * (180/MATH_PI)
        lng = lng1 + tf.asin(dx/tf.cos(lat1*(MATH_PI/180)))*(180/MATH_PI)
        return lat, lng



    def projector(y_input):
        y_in = y_input[:,:,:14]
        y_geo_1 = y_input[:,:,14:17]
        y_geo_2 = y_input[:,:,17:]

        cx = y_in[...,-12] * y_in[...,-4] * y_in[...,-6] + y_in[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_in[...,-11] * y_in[...,-3] * y_in[...,-5] + y_in[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(y_in[...,-10] * y_in[...,-2]) * y_in[...,-6] # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_in[...,-9] * y_in[...,-1]) * y_in[...,-5] # h = exp(h_pred * variance_h) * h_anchor

        w = y_in[...,-10] * y_in[...,-2] # w = exp(w_pred * variance_w) * w_anchor
        h = y_in[...,-9] * y_in[...,-1] # h = exp(h_pred * variance_h) * h_anchor
        cx = tf.where(tf.is_nan(cx), tf.zeros_like(cx), cx)
        cy = tf.where(tf.is_nan(cy), tf.zeros_like(cy), cy)
        w = tf.where(tf.is_nan(w), tf.zeros_like(w), w)
        h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)

        cx = tf.expand_dims(cx, axis=-1)
        cy = tf.expand_dims(cy, axis=-1)
        w = tf.expand_dims(w, axis=-1)
        h = tf.expand_dims(h, axis=-1)

        tensor= Concatenate(axis=-1, name='y_proj')([cx,cy,w,h])
        ind = 0
        xmin = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        ymin = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        xmax = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        ymax = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax

        normalize_coords=True
        tf_img_height = tf.constant(300, dtype=tf.float32, name='img_height')
        tf_img_width = tf.constant(600, dtype=tf.float32, name='img_width')
        tf_normalize_coords = tf.constant(normalize_coords, name='normalize_coords')

        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1
            
        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(tf_normalize_coords, normalized_coords, non_normalized_coords)

        x = xmax - xmin
        x = x / 2
        x = xmin + x
        y = ymax

        lat, lng = streetview_pixel_to_world_coordinates(y_geo_1[:,0,1][0],y_geo_1[:,0,2][0], y_geo_1[:,0,0][0], 600, 300, x, y)
        x_, y_ = world_coordinates_to_streetview_pixel(lat, lng, y_geo_2[:,0,1][0], y_geo_2[:,0,2][0], y_geo_2[:,0,0][0], 600, 300)

        x_h = (xmax - xmin)/2.0
        y_h = (ymax - ymin)/2.0

        xmin_ = x_ - x_h
        ymin_ = y_ - 2*y_h
        xmax_ = x_ + x_h
        ymax_ = y_ + y_h

        xmin_ = tf.where(tf.is_nan(xmin_), tf.zeros_like(xmin_), xmin_)
        ymin_ = tf.where(tf.is_nan(ymin_), tf.zeros_like(ymin_), ymin_)
        xmax_ = tf.where(tf.is_nan(xmax_), tf.zeros_like(xmax_), xmax_)
        ymax_ = tf.where(tf.is_nan(ymax_), tf.zeros_like(ymax_), ymax_)

        cx_ = tf.divide(tf.add(xmin_, xmax_), tf.constant(2.0, dtype=tf.float32))
        cy_ = tf.divide(tf.add(ymin_, ymax_), tf.constant(2.0, dtype=tf.float32))
        w_ = tf.subtract(xmax_,xmin_)
        h_ = tf.subtract(ymax_,ymin_)

        cx_ = tf.where(tf.is_nan(cx_), tf.zeros_like(cx_), cx_)
        cy_ = tf.where(tf.is_nan(cy_), tf.zeros_like(cy_), cy_)
        w_ = tf.where(tf.is_nan(w_), tf.zeros_like(w_), w_)
        h_ = tf.where(tf.is_nan(h_), tf.zeros_like(h_), h_)

        y_out = tf.concat([cx_/600,cy_/300,w_/600,h_/300], -1)

        return y_out

    ############################################################################
    # Build the network.
    ############################################################################


    def ssdmod(x,geo_1,geo_2,suf):
        x1= Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer'+'_'+suf)(x)
        if not (subtract_mean is None):
            x1= Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization'+'_'+suf)(x1)
        if not (divide_by_stddev is None):
            x1= Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization'+'_'+suf)(x1)
        if swap_channels:
            x1= Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap'+'_'+suf)(x1)
        conv1_1= Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1'+'_'+suf)(x1)
        conv1_2= Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2'+'_'+suf)(conv1_1)
        pool1= MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1'+'_'+suf)(conv1_2)

        conv2_1= Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1'+'_'+suf)(pool1)
        conv2_2= Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2'+'_'+suf)(conv2_1)
        pool2= MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2'+'_'+suf)(conv2_2)

        conv3_1= Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1'+'_'+suf)(pool2)
        conv3_2= Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2'+'_'+suf)(conv3_1)
        conv3_3= Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3'+'_'+suf)(conv3_2)
        pool3= MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3'+'_'+suf)(conv3_3)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1'+'_'+suf)(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2'+'_'+suf)(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3'+'_'+suf)(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4'+'_'+suf)(conv4_3)

        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1'+'_'+suf)(pool4)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2'+'_'+suf)(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3'+'_'+suf)(conv5_2)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5'+'_'+suf)(conv5_3)

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6'+'_'+suf)(pool5)

        fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7'+'_'+suf)(fc6)

        conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1'+'_'+suf)(fc7)
        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6adding'+'_'+suf)(conv6_1)
        conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2'+'_'+suf)(conv6_1)

        conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1'+'_'+suf)(conv6_2)
        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7adding'+'_'+suf)(conv7_1)
        conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2'+'_'+suf)(conv7_1)

        conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1'+'_'+suf)(conv7_2)
        conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2'+'_'+suf)(conv8_1)

        conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1'+'_'+suf)(conv8_2)
        conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2'+'_'+suf)(conv9_1)

        # Feed conv4_3 into the L2 normalization layer
        conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm'+'_'+suf)(conv4_3)

        ### Build the convolutional predictor layers on top of the base network

        # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
        conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf'+'_'+suf)(conv4_3_norm)
        fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf'+'_'+suf)(fc7)
        conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf'+'_'+suf)(conv6_2)
        conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf'+'_'+suf)(conv7_2)
        conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf'+'_'+suf)(conv8_2)
        conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf'+'_'+suf)(conv9_2)
        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc'+'_'+suf)(conv4_3_norm)
        fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc'+'_'+suf)(fc7)
        conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc'+'_'+suf)(conv6_2)
        conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc'+'_'+suf)(conv7_2)
        conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc'+'_'+suf)(conv8_2)
        conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc'+'_'+suf)(conv9_2)

        ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        conv4_3_norm_mbox_priorbox= AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                                 two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                                 variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox'+'_'+suf)(conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox= AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox'+'_'+suf)(fc7_mbox_loc)
        conv6_2_mbox_priorbox= AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox'+'_'+suf)(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox= AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox'+'_'+suf)(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox= AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox'+'_'+suf)(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox= AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox'+'_'+suf)(conv9_2_mbox_loc)

        ### Reshape

        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        conv4_3_norm_mbox_conf_reshape= Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape'+'_'+suf)(conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape= Reshape((-1, n_classes), name='fc7_mbox_conf_reshape'+'_'+suf)(fc7_mbox_conf)
        conv6_2_mbox_conf_reshape= Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape'+'_'+suf)(conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape= Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape'+'_'+suf)(conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape= Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape'+'_'+suf)(conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape= Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape'+'_'+suf)(conv9_2_mbox_conf)
        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        conv4_3_norm_mbox_loc_reshape= Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape'+'_'+suf)(conv4_3_norm_mbox_loc)
        fc7_mbox_loc_reshape= Reshape((-1, 4), name='fc7_mbox_loc_reshape'+'_'+suf)(fc7_mbox_loc)
        conv6_2_mbox_loc_reshape= Reshape((-1, 4), name='conv6_2_mbox_loc_reshape'+'_'+suf)(conv6_2_mbox_loc)
        conv7_2_mbox_loc_reshape= Reshape((-1, 4), name='conv7_2_mbox_loc_reshape'+'_'+suf)(conv7_2_mbox_loc)
        conv8_2_mbox_loc_reshape= Reshape((-1, 4), name='conv8_2_mbox_loc_reshape'+'_'+suf)(conv8_2_mbox_loc)
        conv9_2_mbox_loc_reshape= Reshape((-1, 4), name='conv9_2_mbox_loc_reshape'+'_'+suf)(conv9_2_mbox_loc)
        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        conv4_3_norm_mbox_priorbox_reshape= Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape'+'_'+suf)(conv4_3_norm_mbox_priorbox)
        fc7_mbox_priorbox_reshape= Reshape((-1, 8), name='fc7_mbox_priorbox_reshape'+'_'+suf)(fc7_mbox_priorbox)
        conv6_2_mbox_priorbox_reshape= Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape'+'_'+suf)(conv6_2_mbox_priorbox)
        conv7_2_mbox_priorbox_reshape= Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape'+'_'+suf)(conv7_2_mbox_priorbox)
        conv8_2_mbox_priorbox_reshape= Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape'+'_'+suf)(conv8_2_mbox_priorbox)
        conv9_2_mbox_priorbox_reshape= Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape'+'_'+suf)(conv9_2_mbox_priorbox)

        ### Concatenate the predictions from the different layers

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        mbox_conf= Concatenate(axis=1, name='mbox_conf'+'_'+suf)([conv4_3_norm_mbox_conf_reshape,
                                                           fc7_mbox_conf_reshape,
                                                           conv6_2_mbox_conf_reshape,
                                                           conv7_2_mbox_conf_reshape,
                                                           conv8_2_mbox_conf_reshape,
                                                           conv9_2_mbox_conf_reshape])

        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        mbox_loc= Concatenate(axis=1, name='mbox_loc'+'_'+suf)([conv4_3_norm_mbox_loc_reshape,
                                                         fc7_mbox_loc_reshape,
                                                         conv6_2_mbox_loc_reshape,
                                                         conv7_2_mbox_loc_reshape,
                                                         conv8_2_mbox_loc_reshape,
                                                         conv9_2_mbox_loc_reshape])

        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        mbox_priorbox= Concatenate(axis=1, name='mbox_priorbox'+'_'+suf)([conv4_3_norm_mbox_priorbox_reshape,
                                                                   fc7_mbox_priorbox_reshape,
                                                                   conv6_2_mbox_priorbox_reshape,
                                                                   conv7_2_mbox_priorbox_reshape,
                                                                   conv8_2_mbox_priorbox_reshape,
                                                                   conv9_2_mbox_priorbox_reshape])

        # The box coordinate predictions will go into the loss function just the way they are,
        # but for the class predictions, we'll apply a softmax activation layer first
        mbox_conf_softmax= Activation('softmax', name='mbox_conf_softmax'+'_'+suf)(mbox_conf)

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        mbox_loc_tot = Concatenate(axis=2, name='predictions_tot'+suf)([mbox_conf, mbox_loc, mbox_priorbox, geo_1, geo_2])

        # mbox_proj = Projector(300,600, EARTH_RADIUS, GOOGLE_CAR_CAMERA_HEIGHT, MATH_PI)(mbox_loc_tot)

        mbox_proj = Lambda(projector, name='predictions'+'_'+suf+'_mbox_proj')(mbox_loc_tot)

        # mbox_conf_softmax = 2, mbox_loc = 4, mbox_priorbox = 8, mbox_conf_softmax= 2, mbox_proj= 4, mbox_priorbox = 8
        # predictions = Concatenate(axis=2, name='predictions'+'_'+suf)([mbox_conf_softmax, mbox_loc, mbox_priorbox, mbox_proj])

        predictions = Concatenate(axis=2, name='predictions'+suf)([mbox_conf_softmax, mbox_loc, mbox_priorbox, K.zeros_like(mbox_conf_softmax)])

        mbox_proj = Dense(32, input_dim=13, kernel_initializer='normal', activation='relu')(mbox_proj)
        mbox_proj = Dense(16, input_dim=13, kernel_initializer='normal', activation='relu')(mbox_proj)
        mbox_proj = Dense(8, input_dim=13, kernel_initializer='normal', activation='relu')(mbox_proj)
        mbox_proj = Dense(4, input_dim=13, kernel_initializer='normal', activation='relu')(mbox_proj)
        # # mbox_proj_conv1 = tf.zeros(tf.shape(mbox_proj_conv1), dtype=tf.float32,name=None)

        ### iM HERE JUST LAMBDA LAYER IT
        predictions_proj = Concatenate(axis=2, name='predictions'+suf+'_proj')([mbox_conf_softmax, mbox_proj, mbox_priorbox, K.zeros_like(mbox_conf_softmax)])

        # model = Model(input=[x,geo_1,geo_2],output=predictions)
        model = Model(input=[x,geo_1,geo_2],output=[predictions,predictions_proj])

        return model

    weights_path = 'weights/VGG_ILSVRC_16_layers_fc_reduced.h5'

    X = Input(shape=(img_height, img_width, img_channels))
    Z = Input(shape=(img_height, img_width, img_channels))
    X_geo = Input(shape=(17292,3))
    Z_geo = Input(shape=(17292,3))


    ssd1 = ssdmod(X, X_geo, Z_geo, "_1")
    ssd1.load_weights(weights_path, by_name=True)

    ssd2 = ssdmod(Z, Z_geo, X_geo, "_2")
    ssd2.load_weights(weights_path, by_name=True)


    if mode == 'training':
        model = Model(inputs=[X, Z, X_geo, Z_geo], outputs=[ssd1.get_layer(name="predictions_1").output,ssd2.get_layer(name="predictions_2").output,ssd1.get_layer(name="predictions_1_proj").output,ssd2.get_layer(name="predictions_2_proj").output])
        # model = Model(inputs=[x, x_x, geo_1, geo_2], outputs=[ssd1.get_layer("predictions_1").output, ssd1.get_layer("predictions_1_proj").output])
        # model = Model(inputs=[x, x_x, geo_1, geo_2], outputs=ssd1.get_layer("predictions_1").output)
        # model = Model(inputs=[X, Z, X_geo, Z_geo], outputs=[preds_1, preds_2])


    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                     fc7_mbox_conf._keras_shape[1:3],
                                     conv6_2_mbox_conf._keras_shape[1:3],
                                     conv7_2_mbox_conf._keras_shape[1:3],
                                     conv8_2_mbox_conf._keras_shape[1:3],
                                     conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
