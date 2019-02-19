from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, EarlyStopping, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from matplotlib import pyplot as plt
from keras.preprocessing import image
from imageio import imread

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss_mod import SSDLoss
from keras_loss_function.keras_ssd_loss_proj import SSDLoss_proj

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder_mod import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize_Modified
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels_Modified
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation_modified
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import random
np.set_printoptions(precision=20)
import tensorflow as tf
np.random.seed(1337)


img_height = 300 # Height of the model input images
img_width = 600 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True            # print(y_encoded)

steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

# 2: Load some weights into the model.

# TODO: Set the path to the weights you want to load.
weights_path = 'weights/VGG_ILSVRC_16_layers_fc_reduced.h5'

model.load_weights(weights_path, by_name=True)

def Accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    return K.mean(K.equal(K.argmax(y_true[:,:,:-4], axis=-1),
                  K.argmax(y_pred[:,:,:-4], axis=-1)))

def proj(y_pred, y_true):
    #add to gt
    acc = tf.constant(0)
    acc = K.mean(K.equal(K.argmax(y_true[:,:,18:-4], axis=-1), K.argmax(y_pred[:,:,18:-4], axis=-1)))
    return acc

def notproj(y_pred, y_true):
    #add to gt
    acc = tf.constant(0,dtype=tf.float32)
    return acc

def Accuracy_proj(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    result = tf.cond(tf.equal(tf.shape(y_true)[2],36), lambda: proj(y_pred, y_true), lambda: notproj(y_pred, y_true))
    return result

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss1 = SSDLoss(neg_pos_ratio=3, alpha=1.0)
ssd_loss2 = SSDLoss(neg_pos_ratio=3, alpha=1.0)
ssd_loss3 = SSDLoss_proj(neg_pos_ratio=3, alpha=1.0)
ssd_loss4 = SSDLoss_proj(neg_pos_ratio=3, alpha=1.0)

losses = {
    "predictions_1": ssd_loss1.compute_loss,
    "predictions_2": ssd_loss2.compute_loss,
    "predictions_1_proj": ssd_loss3.compute_loss,
    "predictions_2_proj": ssd_loss4.compute_loss

}
lossWeights = {"predictions_1": 1.0,"predictions_2": 1.0,"predictions_1_proj": 1.0,"predictions_2_proj": 1.0}

model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights, metrics=[Accuracy, Accuracy_proj]) 

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset_1 = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)


VOC_2007_images_dir      = '../datasets/Images/'

# The directories that contain the annotations.
VOC_2007_annotations_dir      = '../datasets/VOC/Pasadena/Annotations/'

# The paths to the image sets.
VOC_2007_trainval_image_set_filename = '../datasets/VOC/Pasadena/ImageSets/Main/siamese/trainval_sia.txt'
VOC_2007_val_image_set_filename      = '../datasets/VOC/Pasadena/ImageSets/Main/siamese/val_sia.txt'
VOC_2007_test_image_set_filename     = '../datasets/VOC/Pasadena/ImageSets/Main/siamese/test_sia.txt'


# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'tree']

train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)


val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                      image_set_filenames=[VOC_2007_val_image_set_filename],
                      annotations_dirs=[VOC_2007_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

batch_size = 1 # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation_modified(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels_Modified()        # w_ = tf.where(tf.is_nan(w_), tf.zeros_like(w_), w_)
        # h_ = tf.where(tf.is_nan(h_), tf.zeros_like(h_), h_)

resize = Resize_Modified(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf__1').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf__1').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf__1').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf__1').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf__1').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf__1').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()

val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size)) 

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='checkpoints/double_ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best = 
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)

early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

terminate_on_nan = TerminateOnNaN()
# printer_callback = prediction_history()
# custom_los = custom_loss()
callbacks = [
            model_checkpoint,
#             csv_logger,
#             custom_los,
            learning_rate_scheduler,
            early_stopping,
            terminate_on_nan,
#             printer_callback,
            tbCallBack]
            
initial_epoch   = 0
final_epoch     = 500
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(train_dataset_size/batch_size),
                              epochs=final_epoch,
                              callbacks=callbacks,
                              verbose=1,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)