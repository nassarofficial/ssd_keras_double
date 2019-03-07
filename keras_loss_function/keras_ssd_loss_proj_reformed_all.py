'''
The Keras-compatible loss function for the SSD model. Currently supports TensorFlow only.

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
import math
import keras.backend as K
import numpy as np
from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi



class SSDLoss_proj:
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss


    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        y_true_1 = y_true[:,:,:18]
        y_pred_1 = y_pred[:,:,:18]
        y_true_2 = y_true[:,:,18:]
        y_pred_2 = y_pred[:,:,18:]

        def gt_rem(pred, gt):
            val = tf.subtract(tf.shape(pred)[1], tf.shape(gt)[1],name="gt_rem_subtract")
            gt = tf.slice(gt, [0, 0, 0], [1, tf.shape(pred)[1], 18],name="rem_slice")
            return gt

        def gt_add(pred, gt):
            #add to gt
            val = tf.subtract(tf.shape(pred)[1], tf.shape(gt)[1],name="gt_add_subtract")
            ext = tf.slice(gt, [0, 0, 0], [1, val, 18], name="add_slice")
            gt = K.concatenate([ext,gt], axis=1)
            return gt

        def equalalready(gt, pred): return pred

        def make_equal(pred, gt):
            equal_tensor = tf.cond(tf.shape(pred)[1] < tf.shape(gt)[1], lambda: gt_rem(pred, gt), lambda: gt_add(pred, gt), name="make_equal_cond")
            return equal_tensor


        def matcher(y_true_1,y_pred_1,y_true_2,y_pred_2, bsz):
            pred = 0
            gt = 0

            for i in range(bsz):
                
                filterer = tf.where(tf.not_equal(y_true_1[i,:,-4],99))
                filterer_2 = tf.where(tf.not_equal(y_true_2[i,:,-4],99))

                y_true_new = tf.gather_nd(y_true_1[i,:,:],filterer)            
                y_true_new = tf.expand_dims(y_true_new, 0)
                
                y_true_2_new = tf.gather_nd(y_true_2[i,:,:],filterer_2)
                y_true_2_new = tf.expand_dims(y_true_2_new, 0)

                set1 = tf.cast(y_true_new[i,:,-4],dtype=tf.int32)
                set2 = tf.cast(y_true_2_new[i,:,-4],dtype=tf.int32)
                
                id_pick = tf.sets.set_intersection(set1[None,:], set2[None, :])
                id_pick = tf.cast(id_pick.values[0],dtype=tf.float32)
                            
                filterer = tf.where(tf.equal(y_true_1[i,:,-4],id_pick))
                filterer_2 = tf.where(tf.equal(y_true_2[i,:,-4],id_pick))

                y_true_new = tf.gather_nd(y_true_1[i,:,:],filterer)            
                y_true_new = tf.expand_dims(y_true_new, 0)
                
                y_true_2_new = tf.gather_nd(y_true_2[i,:,:],filterer_2)
                y_true_2_new = tf.expand_dims(y_true_2_new, 0)
                
                iou_out = tf.py_func(iou, [y_pred_1[i,:,-16:-12],tf.convert_to_tensor(y_true_new[i,:,-16:-12])], tf.float64, name="iou_out")
                bipartite_matches = tf.py_func(match_bipartite_greedy, [iou_out], tf.int64, name="bipartite_matches")
                out = tf.gather(y_pred_2[i,:,:], [bipartite_matches], axis=0, name="out")
                


                box_comparer = tf.reduce_all(tf.equal(tf.shape(out)[1], tf.shape(y_true_2_new)[1]), name="box_comparer")
                y_true_2_equal = tf.cond(box_comparer, lambda: equalalready(out, y_true_2_new), lambda: make_equal(out, y_true_2_new), name="y_true_cond")

                if i != 0:
                    pred = K.concatenate([pred,out], axis=-1)
                    gt = K.concatenate([gt,y_true_2_equal], axis=0)
                else:
                    pred = out
                    gt = y_true_2_equal    
            return pred, gt

        y_pred, y_true = matcher(y_true_1,y_pred_1,y_true_2,y_pred_2,1)


        # print("y_true: ", y_true)
        # print("y_pred: ", y_pred)      

        y_pred1 = y_pred_1
        t_true1 = y_true_1

        batch_size = tf.shape(y_pred1)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(t_true1)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

        # 1: Compute the losses for class and box predictions for every box.

        classification_loss = tf.to_float(self.log_loss(t_true1[:,:,:-16], y_pred1[:,:,:-16])) # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(t_true1[:,:,-16:-12], y_pred1[:,:,-16:-12])) # Output shape: (batch_size, n_boxes)
        # print("classification_loss: ", classification_loss)
        # return localization_loss
        

        # 2: Compute the classification losses for the positive and negative targets.

        # Create masks for the positive and negative ground truth classes.
        negatives = t_true1[:,:,0] # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(t_true1[:,:,1:-16], axis=-1)) # Tensor of shape (batch_size, n_boxes)
        # # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any).

        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False) # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: Compute the total loss.

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        geoloss = keras.losses.mean_squared_error(y_true_2[:,:,-2:], y_pred_2[:,:,-2:])

        distloss = keras.losses.mean_squared_error(y_true_2[:,:,-3], y_pred_2[:,:,-3])


        total_loss = (total_loss * tf.to_float(batch_size))+geoloss+distloss
        return total_loss
