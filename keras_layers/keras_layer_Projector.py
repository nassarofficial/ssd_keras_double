from keras import backend as K
from keras.engine.topology import Layer
import math
import tensorflow as tf
import numpy as np
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from bounding_box_utils.bounding_box_utils import convert_coordinates

class Projector(Layer):

    def __init__(self,                  
                img_height,
                img_width,
                EARTH_RADIUS,
                GOOGLE_CAR_CAMERA_HEIGHT,
                MATH_PI,
                **kwargs):
        self.tf_img_height = K.constant(img_height, dtype='float32')
        self.tf_img_width = K.constant(img_width, dtype='float32')
        self.EARTH_RADIUS = EARTH_RADIUS  # Radius in meters of Earth
        self.GOOGLE_CAR_CAMERA_HEIGHT = GOOGLE_CAR_CAMERA_HEIGHT # ballpark estimate of the number of meters that camera is off the ground
        self.MATH_PI = MATH_PI
        super(Projector, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Projector, self).build(input_shape)  # Be sure to call this at the end

    def world_coordinates_to_streetview_pixel(self, lat, lng, lat1,lng1, yaw, image_width, image_height,height=0, zoom=None, object_dims=None, method=None):
        camera_height = self.GOOGLE_CAR_CAMERA_HEIGHT
        pitch = 0
        dx1 = tf.cos((lat1)* (self.MATH_PI/180))
        dx11 = lng-lng1
        dxr = tf.sin(dx11 * (self.MATH_PI/180))
        dx = dx1 * dxr

        dy11 = lat-lat1
        dyr = dy11 * (self.MATH_PI/180)
        dy = tf.sin(dyr)

        look_at_angle = tf.subtract(tf.add(self.MATH_PI,tf.atan2(dx, dy, name='test_atan2_1')),yaw)

        i = tf.constant(np.float32(2*math.pi))

        c = lambda x : tf.reduce_any(tf.greater(x, i))
        b = lambda x : tf.subtract(x, tf.cast(tf.greater(x, i), tf.float32)*2*self.MATH_PI)
        look_at_angle = tf.while_loop(c, b, [look_at_angle])

        t = lambda x : tf.reduce_any(tf.less(x, 0))
        d = lambda x : tf.add(x, tf.cast(tf.less(x, 0), tf.float32)*2*self.MATH_PI) 
        look_at_angle = tf.while_loop(t, d, [look_at_angle])

        z = tf.sqrt(dx*dx+dy*dy)*self.EARTH_RADIUS
        camhei_ = tf.fill(tf.shape(z), -self.GOOGLE_CAR_CAMERA_HEIGHT)

        x_o = (image_width*look_at_angle)/(2*self.MATH_PI)
        y_o = image_height/2 - image_height*(tf.atan2(camhei_, z)-pitch)/(self.MATH_PI) 
        return x_o, y_o


    def streetview_pixel_to_world_coordinates(self, lat1,lng1, yaw, image_width, image_height, x, y):
        camera_height = self.GOOGLE_CAR_CAMERA_HEIGHT  # ballpark estimate of the number of meters that camera is off the ground
        pitch = float(0)
        look_at_angle = x*(2*math.pi)/image_width
        height = 0
        tilt_angle = (image_height/2-y)*math.pi/image_height+pitch
        tilt_angle = tf.cast(tilt_angle, tf.float32)
        z_ = K.minimum(np.float32(-1e-2),tilt_angle)
        z = (-camera_height) / tf.tan(z_)
        dx = tf.sin(look_at_angle-math.pi+yaw)*z/self.EARTH_RADIUS
        dy = tf.cos(look_at_angle-math.pi+yaw)*z/self.EARTH_RADIUS
        lat = lat1 + tf.asin(dy) * (180/self.MATH_PI)
        lng = lng1 + tf.asin(dx/tf.cos(lat1*(self.MATH_PI/180)))*(180/self.MATH_PI)
        return lat, lng

    def call(self, y_input):
        y_in = y_input[:,:,:14]
        y_geo_1 = y_input[:,:,14:17]
        y_geo_2 = y_input[:,:,17:]

        cx = y_in[...,-12] * y_in[...,-4] * y_in[...,-6] + y_in[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_in[...,-11] * y_in[...,-3] * y_in[...,-5] + y_in[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        # w = tf.exp(y_in[...,-10] * y_in[...,-2]) * y_in[...,-6] # w = exp(w_pred * variance_w) * w_anchor
        # h = tf.exp(y_in[...,-9] * y_in[...,-1]) * y_in[...,-5] # h = exp(h_pred * variance_h) * h_anchor
        w = tf.exp(y_in[...,-10] * y_in[...,-2]) * y_in[...,-6] # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_in[...,-9] * y_in[...,-1]) * y_in[...,-5] # h = exp(h_pred * variance_h) * h_anchor

        cx = tf.where(tf.is_nan(cx), tf.zeros_like(cx), cx)
        cy = tf.where(tf.is_nan(cy), tf.zeros_like(cy), cy)
        w = tf.where(tf.is_nan(w), tf.zeros_like(w), w)
        h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)

        xmin = tf.multiply(tf.subtract(cx,0.5),w)
        ymin = tf.multiply(tf.subtract(cy,0.5),h)
        xmax = tf.multiply(tf.subtract(cx,0.5),w)
        ymax = tf.multiply(tf.subtract(cy,0.5),h)

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


        # # x = tf.cast(x, tf.float32)
        # # y = tf.cast(y, tf.float32)
        lat, lng = self.streetview_pixel_to_world_coordinates(y_geo_1[:,0,1],y_geo_1[:,0,2], y_geo_1[:,0,0], 600, 300, x, y)        

        x_, y_ = self.world_coordinates_to_streetview_pixel(lat, lng, y_geo_2[:,0,1], y_geo_2[:,0,2], y_geo_2[:,0,0], 600, 300)
        # print("x_: ", x_.shape)

        x_h = (xmax - xmin)/2.0
        y_h = (ymax - ymin)/2.0

        xmin_ = x_ - x_h
        ymin_ = y_ - 2*y_h
        xmax_ = x_ + x_h
        ymax_ = y_ + y_h

        cx_ = tf.divide(tf.add(xmin_, xmax_),2.0)
        cy_ = tf.divide(tf.add(ymin_,ymax_), 2.0)
        w_ = tf.subtract(xmax_,xmin)
        h_ = tf.subtract(ymax_,ymin_)



        # tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        # tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        # tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        # tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h

        y_out = tf.concat([cx_, cy_, w_, h_], -1)
        return y_out

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 17292, 4)
