'''
Includes:
* Function to compute the IoU similarity for axis-aligned, rectangular, 2D bounding boxes
* Function for coordinate conversion for axis-aligned, rectangular, 2D bounding boxes

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
import tensorflow as tf
import numpy as np
import math

def tf_intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    m = tf.shape(boxes1)[0]
    n = tf.shape(boxes2)[0]
    
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    d = 0

    box1xminxmax = tf.slice(boxes1, [0, 0], [m, xmax])
    box2xminxmax = tf.slice(boxes2, [0, 0], [n, xmax])                      
                            
    min_xy = tf.maximum(tf.tile(tf.expand_dims(box1xminxmax,1),[1, n, 1]),
                        tf.tile(tf.expand_dims(box2xminxmax,0),[m, 1, 1]))
    
    box1xmaxymax = tf.slice(boxes1, [0, 2], [m, 2])
    box2xmaxymax = tf.slice(boxes2, [0, 2], [n, 2])                      

    max_xy = tf.minimum(tf.tile(tf.expand_dims(box1xmaxymax,1),[1, n, 1]),
                    tf.tile(tf.expand_dims(box2xmaxymax,0),[m, 1, 1]))

    side_lengths = tf.maximum(tf.cast(0, dtype=tf.float32), tf.add(tf.subtract(max_xy,min_xy),d))

    return tf.multiply(side_lengths[:,:,0],side_lengths[:,:,1])
    
def tf_iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    intersection_areas = tf_intersection_area_(boxes1, boxes2, coords=coords, mode=mode)
    m = tf.shape(boxes1)[0]
    n = tf.shape(boxes2)[0]
    
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    d = 0
    
    boxes1_areas = tf.tile(tf.expand_dims(tf.multiply(tf.add(tf.subtract(boxes1[:,xmax],boxes1[:,xmin]),d),tf.add(tf.subtract(boxes1[:,ymax],boxes1[:,ymin]),d)), 1), [1,n])
    boxes2_areas = tf.tile(tf.expand_dims(tf.multiply(tf.add(tf.subtract(boxes2[:,xmax],boxes2[:,xmin]),d),tf.add(tf.subtract(boxes2[:,ymax],boxes2[:,ymin]),d)), 0), [m,1])
    
    union_areas = tf.subtract(tf.add(boxes1_areas,boxes2_areas),intersection_areas)
    op = tf.divide(intersection_areas,union_areas)
    return op