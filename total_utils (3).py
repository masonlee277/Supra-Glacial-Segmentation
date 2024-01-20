from keras.optimizers import Adam, SGD
#from image_utils import *
import keras.backend as K
from keras.layers import UpSampling2D
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import Model
from google.cloud import storage
from keras.layers import concatenate
from keras.layers import Flatten, Dense, Reshape, Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
import keras as keras
from tensorflow.keras.metrics import binary_crossentropy
from pathlib import Path
from google.colab import auth
from rasterio.crs import CRS
import os
from rasterio.plot import show
import copy
import imageio
import pandas as pd
from tqdm import tqdm
from typing import List
from skimage import morphology
from rasterio import Affine
import random
import gc
from tensorflow.compat.v1.keras.backend import set_session
import sys
import warnings
from copy import copy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from skimage.morphology import remove_small_objects
import rasterio
from tensorflow.keras.applications import VGG16
from keras.layers import Conv2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import binary_crossentropy
import keras
from keras.layers import LeakyReLU
from keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from google.colab import drive
from PIL import Image
from skimage.transform import resize
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import diplib as dip
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tifffile import tifffile
from sklearn.model_selection import train_test_split
from keras.layers import  Dropout, Activation
from random import shuffle
from PIL import Image, ImageOps
import math
from tensorflow.keras.preprocessing import image
import re
from matplotlib import pyplot as plt
import timeit
import tracemalloc
import io
from sklearn.preprocessing import LabelEncoder
import tensorflow
from keras.models import Model
from io import BytesIO
import tensorflow as tf
from matplotlib.colors import ListedColormap
from copy import deepcopy
from matplotlib import gridspec
import time
import traceback
import cv2
from tensorflow.keras.utils import to_categorical
from matplotlib import cm
import glob


def shape(x):
    print(np.shape(x))

def encode_label(fpath,s=512,buffer=False):
    if type(fpath) == str:
        img = Image.open(fpath)
    else:
        img = fpath
    shape = np.shape(img)
    if shape[0] != s or shape[0]!=s:
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        img = resize_with_padding(img, (s,s))
    label = np.array(img)
    del img # delete the img object from memory after use

    if len(label.shape) == 2:
        label = np.where(label == 0, 0.0, 1.0)
        en_label = np.expand_dims(label,axis=-1)
    elif len(label.shape) == 3 and label.shape[-1] == 4:
        l_4 = label[:, :, 3]
        en_label = np.where(l_4 == 0, 0.0, 1.0)
        en_label = np.expand_dims(en_label,axis=2)
    elif len(label.shape) == 3 and label.shape[-1] == 3:  # Check for RGB shape
        # Convert RGB to binary (0, 1) image
        #print('handling weird shape')
        label = np.mean(label, axis=2)  # Convert RGB to grayscale
        en_label = np.where(label == 0, 0.0, 1.0)
        en_label = np.expand_dims(en_label, axis=2)
    elif len(label.shape) == 3 and label.shape[-1] == 1:
        en_label = label
    else:
        raise Exception(f'Error Encoding Label: {np.shape(label)}')
    
    if buffer:
        print('Increasing Line Width on River Mask')
        kernel = np.ones((12,12), np.uint8)
        en_label = np.array(cv2.dilate(en_label, kernel, iterations=1))
        if en_label.ndim == 2: en_label= np.expand_dims(en_label,axis=-1)
    del label # delete the label object from memory after use
    
    return en_label

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=0)

def plot_image(num, X_train, y_train):
    X = X_train[num]
    Y = y_train[num]
    print(Y.shape)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X, interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(Y[:, :, 0])
    plt.show()

def plot_image_df(num):
    row = df.iloc[num]
    X = row['IMG_Padded']
    Y = row['Label']
    print(Y.shape)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X, interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(Y[:, :, 0])
    plt.show()

def crop_center(img, cropx, cropy):
    (w, h, c) = img.shape
    if c == 3:
        (y, x, c) = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx, :]
    else:
        img = np.squeeze(img)
        (y, x) = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]

def slices_to_map(y_pred, size=10, row_len=10, offset=0, overlay=False, images=None, threshold=0.001):
    """
Make Unified Map of Slices
  slices_to_map
    input:
        ds: contigous dataset slice -- a set of squares
        row_len: how many images are supposed to go in each row
        threshold: lower the threshold the more values that are plotted
    output:
        a graph of all the connected images
        possibly export to JPEG / PNG

    ideation:
      preprocess: return images to 100x100 format -- squeeze + crop

      1. we must sort training and test before we train the model else there will be gaps
          if we are only looking at the test set. Else we can make predictions
          on the entire dataset, sort it, and then plot it
      2. we also see a lot of noise on the sides of the images, we need to remove
          this noise along boundaries, it will create a grid like effect
      3. the shape of the dataset will be (num_img, 100,100). we need to plot
          row_len images and then move to the next row

  ### WILL BE REPLACED BY TILE_STITCH
"""
    scale = 20
    plt.figure(figsize=(row_len, size))
    gs1 = gridspec.GridSpec(row_len, size)
    gs1.update(wspace=0, hspace=0)
    for i in range(0, row_len * size):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        img = y_pred[i + offset]
        img = crop_center(img, 100, 100)
        if not overlay:
            ax1.imshow(img)
        else:
            baseline = images[i + offset]
            baseline = crop_center(baseline, 100, 100)
            img = np.expand_dims(img, axis=2)
            img[img[:, :, 0] < threshold] = 0
            img = np.squeeze(img)
            ax1.imshow(baseline, interpolation='nearest')
            ax1.imshow(img, cmap='jet', alpha=0.5)

def encode_dataset(images_path, labels_path=None, coords=False, height=512, width=512):
    """
  TODO:
    1. Make it so that images coordinates are preserved
        parse from the name of the file
  """
    print('encoding image')
    cols = ['num', 'IMG_Padded', 'Label', 'predicted_network', 'X', 'Y']
    df = pd.DataFrame(columns=cols)
    print(df.head())
    print('df initalized')
    filenames_image = os.listdir(images_path)
    if labels_path is not None:
        filenames_mask = os.listdir(labels_path)
        filenames = [item for item in filenames_image if item in filenames_mask]
    else:
        filenames = filenames_image
    for filename in tqdm(filenames):
        if filename.endswith('.png'):
            name = re.findall('\\d+', filename)
            fname = name[0]
            print(f'found file name: {fname}')
            if coords:
                regex = '\\(([^\\)]+)\\)'
                p = re.compile(regex)
                result = p.search(filename)
                found = result.group(0).strip('()').split(',')
                x_cord = int(found[0])
                y_cord = int(found[-1])
            else:
                x_cord = None
                y_cord = None
            fpath_img = os.path.join(images_path, filename)
            np_img = encode_image(None, True, fpath_img)
            label = None
            if labels_path is not None:
                fpath = os.path.join(labels_path, filename)
                en_label = encode_label(fpath, height)
                label = [en_label]
            row = {'num': fname, 'IMG_Padded': np_img, 'Label': label, 'predicted_network': None, 'X': x_cord, 'Y': y_cord}
            df = df.append(row, ignore_index=True)
    return df

def encode_image(img, png=False, filepath=None, s=512):
    """
  if working with PNG there will be 4 channels, we need to open as a
  if working from how images should be tiled when inputting a geotiff,
    we will have a 1 chanel image

  """
    if png:
        img = np.array(Image.open(filepath).convert('RGB'))
    img = Image.fromarray(img, 'RGB')
    img = np.array(img)
    shp = np.shape(img)
    if shp[2] != 3:
        raise Exception('Image Needs To Have 3 channels')
    if shp[0] != s or shp[1] != s:
        img = resize_with_padding(img, (s, s))
    img = np.array(img)
    img = img / 255
    img = img.astype(np.float32)
    return img

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    """ Input """
    (w, h, c) = input_shape
    inputs = Input(input_shape)
    if c == 1:
        img_input = Input(shape=(w, h, 1))
        inputs = Concatenate()([img_input, img_input, img_input])
    ' Pre-trained VGG16 Model '
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg16.trainable = False
    ' Encoder '
    s1 = vgg16.get_layer('block1_conv2').output
    s2 = vgg16.get_layer('block2_conv2').output
    s3 = vgg16.get_layer('block3_conv3').output
    s4 = vgg16.get_layer('block4_conv3').output
    ' Bridge '
    b1 = vgg16.get_layer('block5_conv3').output
    ' Decoder '
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    ' Output: Binary Segmentation '
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
    model = Model(inputs, outputs, name='VGG16_U-Net')
    return model

def info_df(df):
    print(df.isna().sum())
    print(df.shape)
    print(df.dtypes)

def align_meta_data(fp, save_path):
    with rasterio.open(fp) as src0:
        print('original meta data: ', src0.meta)
        meta1 = src0.meta
    with rasterio.open(save_path, 'r+') as src0:
        meta = src0.meta
        src0.transform = meta1['transform']
        src0.crs = meta1['crs']
        t = src0.crs
#######################################
# inheritance for training process plot
class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

      images_path = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/tile'
      labels_path = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/predicted'

      resize_shape = 512
      batch_size = 1
      X_train_filenames, X_val_filenames = get_file_names(images_path,labels_path)

      my_training_batch_generator = My_Custom_Generator(images_path,labels_path, X_train_filenames, batch_size,resize_shape,aug=False)
      x_val,y_val = my_training_batch_generator.__getitem__(1)

      self.lbl = np.squeeze(y_val[0])
      self.img = np.squeeze(x_val[0])
      self.i = 0

    def on_epoch_end(self, epoch, logs={}):

      #choose a random test image and preprocess
      print('i=',self.i,'loss=',logs.get('loss'))
      self.i+=1

      pred = model.predict(np.expand_dims(self.img,axis=0))

      pred = np.squeeze(pred)
      img = self.img
      lbl = self.lbl

      print(np.shape(img),np.shape(pred), np.shape(lbl))
      fig, axs = plt.subplots(1, 3)

      axs[0].imshow(img)
      plt.axis('off')

      axs[1].imshow(pred)
      plt.axis('off')

      axs[2].imshow(lbl)
      plt.axis('off')

      plt.show()
#######################################
def masked_loss(y_true, y_pred):
    """Defines a masked loss that ignores border/unlabeled pixels (represented as -1).

    Args:
      y_true: Ground truth tensor of shape [B, H, W, 1].
      y_pred: Prediction tensor of shape [B, H, W, N_CLASSES].
    """
    gt_validity_mask = tf.cast(tf.greater_equal(y_true[:, :, :, 0], 0), dtype=tf.float32)
    y_true = K.abs(y_true)
    raw_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    masked = gt_validity_mask * raw_loss
    return tf.reduce_mean(masked)

def s(targets, inputs, smooth=1e-06):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    BCE = binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    return Dice_BCE

def IoULoss(targets, inputs, smooth=1e-06):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def plot_results(y_pred, y_test, X_test, offset=500):
    size = 20
    (fig, axs) = plt.subplots(size, 3, figsize=(30, 300))
    for i in tqdm(range(0, size)):
        ex = np.squeeze(y_pred[i + offset], axis=2)
        truth = np.squeeze(y_test[i + offset], axis=2)
        img = X_test[i + offset]
        axs[i, 0].imshow(truth)
        axs[i, 1].imshow(ex)
        axs[i, 2].imshow(img, interpolation='nearest')
    fig.subplots_adjust(wspace=None, hspace=None)
    fig.show()

def plot_images(*images_lists, titles=None):
    """Plots the images in the lists side by side.

  Args:
    *images_lists (list of lists): A variable number of lists of images to be plotted. Each list should contain 2D images.
    titles (list of strings, optional): A list of strings to be used as the title for each image list. If not provided, no titles will be displayed.

  """
    num_lists = len(images_lists)
    num_images = [len(images) for images in images_lists]
    shapes = [np.shape(images) for images in images_lists]
    print(shapes)
    max_images = max(num_images)
    (fig, axs) = plt.subplots(max_images, num_lists, figsize=(20, 300))
    if titles:
        if len(titles) != num_lists:
            raise ValueError('Number of titles must match number of image lists')
    for i in range(num_lists):
        images = images_lists[i]
        if images.ndim == 4 and np.shape(images)[-1] == 1:
            images = np.squeeze(images, axis=-1)
        for j in range(len(images)):
            image = images[j]
            axs[j, i].imshow(image)
            if titles:
                axs[j, i].set_title(titles[i])

def predict_from_dataframe(image_df, model):
    """
    Input: image_df
    Output: image_df with the imaged prediction map for each tile

    each image within image_df is an unprocessed numpy array (100x100)
      psuedo code:
        for each image in image_df
          process the image into 128,128,3
          make a prediction on the image
          save the prediction into
   cols = ['num', 'IMG_Padded', 'Label', 'predicted_network', 'X', 'Y']

  """
    for (index, row) in df.iterrows():
        each_image = row['IMG_Padded']
        if index % 50 == 0:
            print(f'Images Predicted: {index}')
        img_processed = each_image
        prediction_map = model.predict(np.expand_dims(img_processed, axis=0), verbose=0)
        pred_map = prediction_map.squeeze(axis=0).squeeze(axis=2)
        row['predicted_network'] = pred_map
    return image_df

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = (I - mn) / mx * 255
    return I.astype(np.uint8)

def get_examples(images_path, labels_path):
    """
    Returns tuple of (images, labels)
    Where each image is of size 100x100
    and each label is 100x100x2 (one-hot encoded, "is this pixel a stream or no?")

    :param images_path: path to DIRECTORY where images are kept
    :param labels_path: see above but for labels
    """
    images_lst = []
    for filename in os.listdir(images_path):
        if filename.endswith('.png'):
            img_with_chan = np.reshape(imageio.imread(images_path + filename), (100, 100, 1))
            images_lst.append(img_with_chan)
    labels_lst = []
    for filename in os.listdir(labels_path):
        if filename.endswith('.png'):
            labels_lst.append(iio.imread(labels_path + filename))
    encoded_labels = [np.zeros((100, 100)) for l in labels_lst]
    for i in range(len(labels_lst)):
        l = labels_lst[i]
        l_4 = l[:, :, 3]
        encoded_labels[i] = np.where(l_4 == 0, 0.0, 1.0)
    return (images_lst, encoded_labels)

def lut_display(image, display_min, display_max):
    lut = np.arange(2 ** 16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)

from typing import List

def get_file_names(images_path: str, labels_path: str) -> List[str]:
    """
    Returns the file names of the images and labels that match in both directories and are shuffled.
    :param images_path: the directory path where the images are located
    :param labels_path: the directory path where the labels are located
    :return: a list of file names for the images and labels that match in both directories and are shuffled
    """
    filenames_image = os.listdir(images_path)
    filenames_mask = os.listdir(labels_path)
    files_tile = filenames_image
    files_mask = filenames_mask
    assert len(list(set(files_mask).difference(files_tile))) == 0
    assert len(list(set(files_tile).difference(files_mask))) == 0
    assert len(files_tile) == len(np.unique(files_tile))
    assert len(files_mask) == len(np.unique(files_mask))
    from sklearn.utils import shuffle
    filenames = [item for item in filenames_image if item in filenames_mask]
    (filenames_image, filenames_mask) = shuffle(filenames, filenames)
    (X_train_filenames, X_val_filenames, y_train, y_val) = train_test_split(filenames_image, filenames_mask, test_size=0.2, random_state=1)
    return (X_train_filenames, X_val_filenames)

def augment_image(img, num_lines=30):
    uniform_pos = np.random.uniform(1, 512, num_lines)
    width = 3
    for x in uniform_pos:
        width = int(np.random.normal(loc=5, scale=2, size=1)[0])
        x = int(x)
        if bool(random.getrandbits(1)):
            img[x - width:x + width, :] = 0
        else:
            img[:, x - width:x + width] = 0
    return img

def gan_stuff():
    working_dir = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/GAN_TILES'
    mask_path = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/dataset_corrected/aug_masks'
    i = 0
    for fp in os.listdir(mask_path):
        i += 1
        image_path = os.path.join(mask_path, fp)
        img = np.array(Image.open(image_path))
        img = augment_image(img)
        save_path = os.path.join(working_dir, fp)
        imageio.imwrite(save_path, img)
        if i % 250 == 0:
            print(save_path, image_path)

def raster_bands(rasterout):
    with rasterio.open(rasterout) as src0:
        print(src0.meta)
        m = src0.meta
        print()
        h = int(m['height'])
        w = int(m['width'])
        print(h, w)
        map_recon = np.zeros(shape=(h, w))
        print(map_recon.shape)
        for b in range(m['count'] - 1):
            band = src0.read(b + 1)
            map_recon = np.dstack((map_recon, band))
        plot_bands(map_recon)

def display(map_im):
    (fig, ax) = plt.subplots(figsize=(50, 50))
    ax.imshow(map_im, interpolation='nearest', cmap='viridis')
    plt.tight_layout()

def aug_batch(batch_x, batch_y):
    """
  "The 'aug_batch' function applies image augmentations using the 'albumentations'
  library to input image and mask lists, 'batch_x' and 'batch_y', respectively.
  It returns the augmented image and mask lists. The function continues applying
  augmentations until the mask sum exceeds a threshold value, and can resample
  the image if the threshold is not met after a certain number of iterations."

  """
    (oh, ow) = (512, 512)
    aug = A.Compose([A.RandomSizedCrop(min_max_height=(100, 356), height=oh, width=ow, p=0.8), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.OneOf([A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=1), A.Sharpen(p=1), A.Solarize(threshold=0.1, p=1)], p=0.8), A.ColorJitter(brightness=0.7, contrast=0.4, saturation=0.4, hue=0.3, always_apply=False, p=1), A.GaussNoise(var_limit=(0, 0.1), p=0.6)])
    xn = []
    yn = []
    for (i, (image, mask)) in enumerate(zip(batch_x, batch_y)):
        mask_sum = 0
        it = 0
        thresh = 100
        sum_list = np.empty(shape=len(batch_x))
        while mask_sum <= thresh:
            augmented = aug(image=image, mask=mask)
            im_aug = augmented['image']
            m_aug = augmented['mask']
            mask_sum = int(np.sum(m_aug))
            if it > 5:
                if np.max(sum_list) > thresh:
                    index = np.argmax(sum_list)
                    image = batch_x[index]
                    mask = batch_y[index]
                else:
                    break
            sum_list[i] = mask_sum
            it += 1
        xn.append(im_aug)
        yn.append(m_aug)
        del augmented
        del image
        del mask
    return (xn, yn)



def dice_loss_with_weights(y_true, y_pred, w_tp=2.0, w_fp=0.01, w_tn=0.01, w_fn=1):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    tp = w_tp * tf.reduce_sum(y_true * y_pred)
    fp = w_fp * tf.reduce_sum((1 - y_true) * y_pred)
    tn = w_tn * tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fn = w_fn * tf.reduce_sum(y_true * (1 - y_pred))
    dice_loss = 1 - (2 * tp + 1) / (2 * tp + fp + fn + 1)
    return dice_loss

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_with_cross_entropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dice = dice_loss(y_true, y_pred)
    y_true = tf.cast(y_true, tf.int32)
    print(y_true.dtype)
    ce = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=tf.constant(2))
    return dice + ce

def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, 'float32'))
    return iou

def on_train_begin(self, logs={}):
    images_path = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/tile'
    labels_path = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/DB512v6/predicted'
    resize_shape = 512
    batch_size = 1
    (X_train_filenames, X_val_filenames) = get_file_names(images_path, labels_path)
    my_training_batch_generator = My_Custom_Generator(images_path, labels_path, X_train_filenames, batch_size, resize_shape, aug=False)
    (x_val, y_val) = my_training_batch_generator.__getitem__(1)
    self.lbl = np.squeeze(y_val[0])
    self.img = np.squeeze(x_val[0])
    self.i = 0

def on_epoch_end(self, epoch, logs={}):
    print('i=', self.i, 'loss=', logs.get('loss'))
    self.i += 1
    pred = model.predict(np.expand_dims(self.img, axis=0))
    pred = np.squeeze(pred)
    img = self.img
    lbl = self.lbl
    print(np.shape(img), np.shape(pred), np.shape(lbl))
    (fig, axs) = plt.subplots(1, 3)
    axs[0].imshow(img)
    plt.axis('off')
    axs[1].imshow(pred)
    plt.axis('off')
    axs[2].imshow(lbl)
    plt.axis('off')
    plt.show()

def get_image(images_path, display=True):
    d = os.listdir(images_path)
    name = random.choice(d)
    fp = os.path.join(images_path, name)
    img = Image.open(fp)
    if display:
        plt.imshow(img)
    return (name, img)

def print_examples(x_val, y_val, model, dilate_im, step=None, num_examples=2):
    # Ensure x_val, y_val, and model are not None and have expected shapes
    if x_val is None or y_val is None or model is None or len(x_val) == 0 or len(y_val) == 0:
        print("Error: Invalid inputs provided.")
        return
    
    num_examples = min(num_examples, len(x_val), len(y_val))
    
    y_pred = model.predict(x_val[:num_examples])

    if step is not None:
        print(f'Step {step}:')
    else:
        print('End of epoch:')
    
    (fig, axs) = plt.subplots(num_examples, 5 if dilate_im else 3, figsize=(15, 3*num_examples))

    for i in range(num_examples):
        lbl = np.squeeze(y_val[i])
        img = np.squeeze(x_val[i])
        pred = np.squeeze(y_pred[i])

        if dilate_im:
            lbl_dilated = dilate_batch(np.expand_dims(y_val[i], axis=0))[0]
            lbl_dilated = np.squeeze(lbl_dilated)
            pred_masked = np.multiply(pred, lbl_dilated)
        else:
            lbl_dilated = None
            pred_masked = None

        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Input Image')
        
        axs[i, 1].imshow(pred)
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Predicted')
        
        axs[i, 2].imshow(lbl)
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Ground Truth')

        if dilate_im:
            axs[i, 3].imshow(pred_masked)
            axs[i, 3].axis('off')
            axs[i, 3].set_title('Predicted Masked')
            
            axs[i, 4].imshow(lbl_dilated)
            axs[i, 4].axis('off')
            axs[i, 4].set_title('Ground Truth Dilated')

    plt.tight_layout()
    plt.show()

# Assuming the function `dilate_batch` is defined elsewhere in your code.


def masked_dice_loss(y_true, y_pred, mask, cont_loss=False, continuity_weight=0.15):
    smooth = 1.0
    y_true = tf.multiply(y_true, mask)
    y_pred = tf.multiply(y_pred, mask)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    if cont_loss:
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        line_length_penalty_scalar = line_length_penalty(y_pred_binary)
        total_loss = dice_loss + continuity_weight * line_length_penalty_scalar
    else:
        total_loss = dice_loss
    del y_true_f
    del y_pred_f
    del y_pred
    del y_true
    del mask
    return total_loss

def remove_small_objects_np(images, min_size=500, threshold=0.5):
    """
    Remove small objects from a batch of images.
    Args:
        images: numpy array of shape (n, 512, 512, 1)
        min_size: minimum size of connected components to keep
    Returns:
        numpy array of images with small objects removed
    """
    processed_images = []
    for img in images:
        img = img.squeeze()
        img_binary = np.where(img > threshold, True, False)
        img_cleaned = morphology.remove_small_objects(img_binary, min_size)
        processed_images.append(np.where(img_cleaned[..., np.newaxis], 1, 0))
    return np.array(processed_images)
####################################################
import tensorflow as tf  # Import TensorFlow library

def connection_nn(y_pred, model2, threshold=0.5, min_size=100):
    """
    Thresholds the images in the batch to make them binary, removes small objects,
    and calls model2.predict(y_pred) to connect discontinuous lines.

    Args:
        y_pred: tensor of shape (n, 512, 512, 1) - A batch of images with shape (n, 512, 512, 1).
                This typically represents predicted values, such as image segmentation masks.
        model2: a TensorFlow model - A neural network model used for prediction.
        threshold: float (default: 0.5) - A threshold value used to convert image values to binary (0 or 1).
                   Pixels with values above this threshold become 1, and those below become 0.
        min_size: int (default: 100) - Minimum size for connected objects to be retained in the output.
                  Smaller objects are removed from the binary images.

    Returns:
        tensor of shape (n, 512, 512, 1) with connected lines - A binary tensor representing connected lines or objects
        in the input images after applying the threshold and removing small objects.
    """
    # Convert pixel values in y_pred to binary based on the threshold
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    # Convert the binary tensor to a NumPy array
    y_pred_np = y_pred_binary.numpy()
    
    # Remove small objects from the binary image using a custom function (remove_small_objects_np)
    y_pred_cleaned_np = remove_small_objects_np(y_pred_np, min_size)
    
    # Convert the cleaned NumPy array back to a TensorFlow tensor
    y_pred_cleaned = tf.convert_to_tensor(y_pred_cleaned_np, dtype=tf.float32)
    
    # Use model2 to predict the connected lines or objects in the cleaned binary image
    connected_lines = model2.predict(y_pred_cleaned)
    
    # Apply the threshold again to the model's predictions and convert them to binary
    connected_lines = tf.where(connected_lines > threshold, 1, 0)
    
    # Add an extra dimension to the connected_lines tensor if it's missing
    if len(connected_lines.shape) == 3:
        connected_lines = tf.expand_dims(connected_lines, axis=-1)
    
    # Return the final binary tensor with connected lines or objects
    return connected_lines



def masked_dice_loss_test(y_true, y_pred, mask):
    """
    Calculate the Dice Loss for two tensors with optional masking.

    Parameters:
    - y_true (tf.Tensor): The ground truth tensor.
    - y_pred (tf.Tensor): The predicted tensor.
    - mask (tf.Tensor): A binary mask that can be used to focus the loss calculation on specific areas.

    Returns:
    - loss (tf.Tensor): The Dice Loss value.
    - y_true_f (tf.Tensor): Flattened version of y_true after applying the mask.
    - y_pred_f (tf.Tensor): Flattened version of y_pred after applying the mask.

    The Dice Loss is a metric used in image segmentation tasks to measure the similarity
    between two binary masks (y_true and y_pred). It ranges between 0 and 1, with 0 indicating
    no similarity and 1 indicating a perfect match.

    The function applies the given binary mask to both y_true and y_pred to focus the loss
    calculation on specific regions of interest. It then calculates the Dice Loss based on
    the masked tensors.

    The formula for Dice Loss is as follows:
    Dice Loss = 1 - (2 * intersection + smooth) / (sum(y_true) + sum(y_pred) + smooth)

    - `intersection` is the sum of element-wise multiplication between y_true and y_pred.
    - `smooth` is a smoothing constant to avoid division by zero errors.
    - `sum(y_true)` is the sum of all elements in y_true.
    - `sum(y_pred)` is the sum of all elements in y_pred.

    The function returns the calculated loss along with the flattened versions of y_true and y_pred
    after applying the mask.
    """
    smooth = 1.0
    y_true = tf.multiply(y_true, mask)  # Apply the mask to y_true.
    y_pred = tf.multiply(y_pred, mask)  # Apply the mask to y_pred.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    loss = 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return (loss, y_true_f, y_pred_f)

def dilate_batch(batch, kernel_size=12, iterations=1):
    """
    Dilate a batch of images using OpenCV.

    Args:
    batch (list of numpy arrays): List of input images to be dilated.
    kernel_size (int, optional): Size of the kernel used for dilation. Default is 12.
    iterations (int, optional): Number of times dilation is applied. Default is 1.

    Returns:
    numpy array: An array containing the dilated images.

    Dependencies:
    - This function relies on the OpenCV library (cv2) for image dilation operations.
    - You need to have 'import numpy as np' at the beginning of your script for numpy arrays.

    Example:
    >>> input_images = [image1, image2, ...]  # List of input images
    >>> dilated_images = dilate_batch(input_images, kernel_size=8, iterations=2)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_batch = []
    for image in batch:
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        dilated_batch.append(dilated_image)
    return np.array(dilated_batch)
    
def iterative_connection_nn(y_pred, segconnector, num_iterations=3, threshold=0.5, min_size=100):
    """
    Iteratively calls the connection_nn function, using the output of one call as 
    the input to the next.

    Args:
        y_pred: tensor of shape (n, 512, 512, 1) - A batch of images with shape (n, 512, 512, 1).
                This typically represents predicted values, such as image segmentation masks.
        segconnector: a TensorFlow model - A neural network model used for refining the connections.
        num_iterations: int - The number of times to call the connection_nn function.
        threshold: float (default: 0.5) - A threshold value used to convert image values to binary (0 or 1).
                   Pixels with values above this threshold become 1, and those below become 0.
        min_size: int (default: 100) - Minimum size for connected objects to be retained in the output.
                  Smaller objects are removed from the binary images.

    Returns:
        tensor of shape (n, 512, 512, 1) with iteratively refined images.
    """
    
    current_result = y_pred

    for _ in range(num_iterations):
        current_result = connection_nn(current_result, segconnector, threshold=threshold, min_size=min_size)
    
    # Ensure the output shape is consistent
    if len(current_result.shape) == 3:
        current_result = tf.expand_dims(current_result, axis=-1)
    
    current_result = tf.cast(current_result > threshold, tf.float32)
    return current_result

####################################################
import os
import tensorflow as tf

def custom_fit(model,             # The neural network model to train.
               train_generator,   # The training data generator.
               steps_per_epoch,   # The number of steps (batches) to run in each epoch.
               epochs,            # The total number of training epochs.
               images_path_b,     # Path to the directory containing images.
               labels_path_b,     # Path to the directory containing labels.
               checkpoint_dir=None,  # Path to the directory for saving model checkpoints (optional).
               callbacks=None,    # List of Keras callbacks for monitoring and saving models (optional).
               dilate_im=False,   # Flag to enable image dilation during training (default: False).
               optimizer=None,    # The optimizer to use for training (optional, default: Adam).
               loss_fn=None,      # The loss function to use (optional, default: masked_dice_loss).
               fine_tune=False,   # Flag to enable fine-tuning mode (default: False).
               model_lines=None
              ):
    """
    Train a neural network model using custom settings.

    Args:
        model (tf.keras.Model): The neural network model to train.
        train_generator (object): The training data generator.
        steps_per_epoch (int): The number of steps (batches) to run in each epoch.
        epochs (int): The total number of training epochs.
        images_path_b (str): Path to the directory containing images.
        labels_path_b (str): Path to the directory containing labels.
        checkpoint_dir (str, optional): Path to the directory for saving model checkpoints.
        callbacks (list, optional): List of Keras callbacks for monitoring and saving models.
        dilate_im (bool, optional): Flag to enable image dilation during training (default: False).
        optimizer (tf.keras.optimizers.Optimizer, optional): The optimizer to use for training (default: Adam).
        loss_fn (function, optional): The loss function to use (default: masked_dice_loss).
        fine_tune (bool, optional): Flag to enable fine-tuning mode (default: False).
    
    Description:
        This function trains a neural network model with customizable settings. It allows for fine-tuning and custom loss functions.
        The training process is controlled by specifying the number of training steps per epoch and the total number of epochs.
        The model is trained on image data located in the specified 'images_path_b' directory and their corresponding labels in 'labels_path_b'.
        Checkpoints of the model's weights can be saved in 'checkpoint_dir' at specified epochs for later use.
        Additionally, custom callbacks can be provided for monitoring and saving the model during training.
        Image dilation can be enabled with 'dilate_im', and fine-tuning mode can be activated with 'fine_tune', which allows for loading pre-trained weights and using a different loss function.
        The training history, including loss and accuracy, is returned as a dictionary.
        
    Returns:
        dict: A dictionary containing training history (loss and accuracy).
    """
    print(model.optimizer)
    print(checkpoint_dir)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    model.optimizer = optimizer
    print(model.optimizer)
    print(model.loss)
    
    resize_shape = 512
    batch_size = 1
    (X_train_filenames, X_val_filenames) = get_file_names(images_path_b, labels_path_b)
    my_training_batch_generator = My_Custom_Generator(images_path_b, labels_path_b, X_train_filenames, batch_size, resize_shape, aug=False)
    
    if loss_fn is None:
        loss_fn = masked_dice_loss
        dilate_im = True
        print('Dilation Activated by Default')
        print('Masked Loss Activated by Default')
    
    if fine_tune:
        # model_lines = unet(sz=(512, 512, 1))
        # checkpoint_path = '/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/model_folder/VAE/training_no_loops/cp-{epoch:04d}.ckpt'
        # model_lines.load_weights(checkpoint_path.format(epoch=45))
        loss_fn = dice_loss
        print(f'Fine Tune Mode Activated: {loss_fn}')
    
    history = {'loss': [], 'accuracy': []}
    
    # The outer loop runs for each epoch (an entire pass through the dataset).
    for epoch in range(epochs):
        
        # Every 4 epochs, the model weights are saved.
        # This helps in periodically saving the progress of the model so that you can recover it if needed.
        if (epoch + 1) % 4 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'model_weights_epoch_{epoch + 1}.h5'))
            print(f'Model weights saved for epoch {epoch + 1} in {checkpoint_dir}')
        
        # Initialize metrics for the current epoch.
        epoch_loss = 0
        epoch_accuracy = 0
        
        # The inner loop runs for each batch of data in the current epoch.
        for step in range(steps_per_epoch):
            
            # Fetch a batch of data from the training generator.
            (x_batch, y_batch) = train_generator.__getitem__(step)
            
            # Every 10 steps, print the current loss value and display some prediction examples.
            # This gives a regular update on the progress during training.
            if step % 10 == 0:
                print(f'Step: {step + 1}/{steps_per_epoch} - Loss: {epoch_loss}')
                print_examples(x_batch, y_batch, model, dilate_im, step=step)
            
            # TensorFlow's GradientTape is used to monitor operations for which gradients should be computed.
            with tf.GradientTape() as tape:
                
                # Make predictions for the current batch.
                y_pred = model(x_batch, training=True)
                
                # If fine-tuning is enabled, use the 'connection_nn' function.
                if fine_tune:
                    batch_connected = connection_nn(y_pred, model_lines)
                    
                    # If dilation is also enabled, perform dilation.
                    # Then, calculate the loss using both the connected and dilated versions.
                    if dilate_im: #Make sure the chosen loss function takes in a mask
                        batch_dilated = dilate_batch(y_batch)
                        step_loss = loss_fn(y_pred, batch_connected, batch_dilated)
                        del batch_dilated
                    else:
                        step_loss = loss_fn(y_pred, batch_connected)
                
                # If only dilation is enabled (without fine-tuning), just dilate and calculate the loss.
                elif dilate_im:
                    batch_dilated = dilate_batch(y_batch)
                    step_loss = loss_fn(y_batch, y_pred, batch_dilated)
                    del batch_dilated
                    
                # If neither fine-tuning nor dilation is enabled, calculate the loss directly.
                else:
                    step_loss = loss_fn(y_batch, y_pred)
            
            # Compute the gradients for the current step.
            gradients = tape.gradient(step_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Update the model's weights using the computed gradients.
            step_accuracy = tf.keras.metrics.binary_accuracy(y_batch, y_pred) # Calculate the accuracy for the current batch.
            
            # Accumulate the loss and accuracy values for the epoch.
            epoch_loss += step_loss
            epoch_accuracy += step_accuracy
            
            # Free up memory by deleting variables that are no longer needed.
            del y_pred
            del y_batch

            
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss}')
        print(f'Epoch completion: {100 * (epoch + 1) / epochs:.2f}%')
        
        (x_val, y_val) = my_training_batch_generator.__getitem__(1)
        print_examples(x_val, y_val, model, False)
        
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                    callback.on_epoch_end(epoch)
        
        epoch_loss /= steps_per_epoch
        epoch_accuracy /= steps_per_epoch
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
    
    return history

def compile_model(height: int, width: int, segmentation: bool=True) -> tf.keras.Model:
    """
  compile the model to be trained

  :param height: image height
  :param width: image width
  :param segmentation: bool flag to decide if to use binary classification or image segmentation. Default is segmentation
  :return: Compiled model
  """
    IMG_HEIGHT = height
    IMG_WIDTH = width
    IMG_CHANNELS = 3
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    print(input_shape)
    model_vgg16 = build_vgg16_unet(input_shape)
    model = model_vgg16
    loss = masked_dice_loss
    iou1 = tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)
    iou2 = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
    auc = tf.keras.metrics.AUC()
    aucLog = tf.keras.metrics.AUC(from_logits=True)
    metric = [mean_iou]
    opt = tf.keras.optimizers.legacy.Adam()
    model.compile(optimizer=opt, loss=loss, metrics=metric)
    print('total number of model parameters:', model.count_params())
    return model

  
  ##############################


  ##############################

def train_model(model: tf.keras.Model, images_path: str, labels_path: str, working_dir: str, epochs=20, batch_size: int=32, pretrained_weights: str=None, resize_shape=512, fine_tune=False):
    """
  This function trains a given model using image and label data from specified path, saves the model's weights to a specified directory, and saves the training history.
  :param model: Tensorflow model object to be trained
  :param images_path: path to directory where images are stored
  :param labels_path: path to directory where labels are stored
  :param working_dir: path to directory where model weights and training history will be saved
  :param epochs: number of training epochs
  :param batch_size: batch size for training
  :param pretrained_weights: path to pretrained weights, if any
  """
    (X_train_filenames, X_val_filenames) = get_file_names(images_path, labels_path)
    epochs = epochs
    batch_size = 10
    STEPS_PER_EPOCH = len(X_train_filenames) / batch_size
    SAVE_PERIOD = 2
    m = int(SAVE_PERIOD * STEPS_PER_EPOCH)
    checkpoint_dir = os.path.join(working_dir, 'checkpoint_dir')
    try:
        os.mkdir(checkpoint_dir)
        print('constructed checkpoint')
    except:
        print('checkpoint dir made')
    my_training_batch_generator = My_Custom_Generator(images_path, labels_path, X_train_filenames, batch_size, resize_shape, aug=True)
    my_validation_batch_generator = My_Custom_Generator(images_path, labels_path, X_val_filenames, batch_size, resize_shape, aug=False)
    (x, y) = my_training_batch_generator.__getitem__(2)
    print('x_batch shape: ', x.shape, np.max(x[1]), np.min(x[1]))
    print('y_batch shape: ', y.shape)
    im = y[1]
    print(f'Unique Labels: {len(np.unique(im))}')
    assert len(np.unique(im)) == 2
    assert np.max(np.unique(im)) <= 1.0
    im = x[1]
    print(np.max(im))
    assert np.max(np.unique(im)) <= 1.0
    os.path.exists(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=m)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-05, verbose=1)
    model.save_weights(checkpoint_path.format(epoch=0))
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print('pre-trained weights loaded')
    try: 
      print("Custom Fit Arguments:")
      print(f"model: {model}")
      print(f"train_generator: {my_training_batch_generator}")
      print(f"steps_per_epoch: {int(STEPS_PER_EPOCH)}")
      print(f"epochs: {epochs}")
      print(f"images_path_b: {images_path}")
      print(f"labels_path_b: {labels_path}")
      print(f"checkpoint_dir: {checkpoint_dir}")
      print(f"callbacks: [PlotLearning(), early_stopping, cp_callback, reduce_lr]")
      print(f"dilate_im: {False}")
      print(f"optimizer: {None}")
      print(f"loss_fn: {masked_dice_loss}")
      print(f"fine_tune: {fine_tune}")

    except Exception as e: 
      print('Error Printing')

    history = custom_fit(model=model,
                        train_generator=my_training_batch_generator,
                        steps_per_epoch=int(STEPS_PER_EPOCH),
                        epochs=epochs,
                        images_path_b=images_path,  # Provide the actual images_path value
                        labels_path_b=labels_path,  # Provide the actual labels_path value
                        checkpoint_dir=checkpoint_dir,
                        callbacks=[PlotLearning(), early_stopping, cp_callback, reduce_lr],
                        dilate_im=False,
                        optimizer=None,
                        loss_fn=masked_dice_loss,
                        fine_tune=fine_tune)

    
    hist_df = pd.DataFrame(history)
    save_path_his = os.path.join(working_dir, 'history2.csv')
    hist_df.to_csv(save_path_his)

def get_batch(images_path, labels_path, model, model_lines, b=1):
    batch_size = 16
    (X_train_filenames, X_val_filenames) = get_file_names(images_path, labels_path)
    my_validation_batch_generator = My_Custom_Generator(images_path, labels_path, X_val_filenames, batch_size)
    (x_val, y_val) = my_validation_batch_generator.__getitem__(b)
    print(x_val.shape)
    print(y_val.shape)
    if model is not None:
        y_pred = model.predict(x_val, verbose=1)
        if model_lines is not None:
            ds = tf.image.resize(deepcopy(y_pred), [256, 256])
            y_fixed = model_lines.predict(ds)
            skel_im = []
            for im in y_fixed:
                skel_im.append(morphology.skeletonize(np.where(im > 0.001, 1, 0)))
            skel_im_2 = []
            print('skel images shape:  ', np.shape(skel_im))
            pred_skel = model_lines.predict(np.expand_dims(skel_im, axis=-1))
            for im in pred_skel:
                skel_im_2.append(morphology.skeletonize(np.where(im > 0.001, 1, 0)))
            skel_im_3 = []
            print('skel images shape:  ', np.shape(skel_im_2))
            pred_skel = model_lines.predict(np.expand_dims(skel_im_2, axis=-1))
            for im in pred_skel:
                skel_im_3.append(morphology.skeletonize(np.where(im > 0.001, 1, 0)))
            skel_im_4 = []
            print('skel images shape:  ', np.shape(skel_im))
            pred_skel = model_lines.predict(np.expand_dims(skel_im_3, axis=-1))
            for im in pred_skel:
                skel_im_4.append(morphology.skeletonize(np.where(im > 0.001, 1, 0)))
            plot_images([y_pred, y_fixed, skel_im, skel_im_2, skel_im_3, skel_im_4, x_val, y_val])
        else:
            plot_images([y_pred, x_val, y_val])
    return (x_val, y_val)

def show(image):
    if np.ndim(image) == 3 and np.shape(image)[-1] == 1:
        image = np.squeeze(image, axis=-1)
    print(np.shape(image), np.min(image), np.max(image))
    plt.imshow(image)

def plot_images(images_list):
    """
  Takes a variable number of lists of images and plots them side by side.
  Each image in the list of images is plotted in the same row.

  Parameters:
  images_list (list of lists of numpy arrays): A list of lists of images. Each inner list should contain images of the same size (x, 512, 512, 3) or (x, 512, 512, 1).

  Returns:
  None
  """
    n_rows = len(images_list)
    n_cols = max([len(images) for images in images_list])
    (fig, axs) = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    for (i, images) in enumerate(images_list):
        for (j, image) in enumerate(images):
            if image.ndim == 3 and image.shape[-1] == 1:
                image = np.squeeze(image)
                cmap = 'gray'
            else:
                cmap = None
            axs[i, j].imshow(image, cmap=cmap)
            axs[i, j].axis('off')
    fig.tight_layout()
    plt.show()

def predict_from_dataframe_v2(df: pd.DataFrame, model, batch_size: int=64, resize: bool=False, single_channel: bool=False) -> pd.DataFrame:
    """
    This function predicts on image tiles stored in a dataframe using a pre-trained model
    df : pd.DataFrame : dataframe containing image tiles
    model : keras.model : pre-trained model
    batch_size : int : number of images to be predicted at once
    resize : bool : whether to resize the images
    single_channel : bool : whether the images are single channel or not
    """
    if single_channel:
        print('single channel tiff')
        images = df['tile'].apply(lambda x: encode_label(x)).tolist()
    else:
        print('3 channel tiff')
        images = df['tile'].apply(lambda x: encode_image(x)).tolist()
    print('Images From Dataframe Are Properly Encoded')
    res = int(model.input.shape[1])
    if resize:
        print('resizing images to model input')
        images = tf.image.resize(images, [res, res])
        print('Resizing Before Prediction: ', np.shape(images))
    print('images are proccesed: ', np.shape(images))
    images = np.stack(images, axis=0)
    assert images.ndim == 4
    image = images[0]
    print('tiled image data: ', np.shape(images), np.min(image), np.max(image), image.dtype)
    predictions = model.predict(images, verbose=1)
    gc.collect()
    print('prediction data shape: ', np.shape(predictions))
    pred = []
    for im in predictions:
        pred.append([im])
    del predictions
    ex_df = pd.DataFrame(data=pred, columns=['tile'])
    ex_df['x'] = df['x']
    ex_df['y'] = df['y']
    del df
    return ex_df

def apply_function_to_tiff(img: np.ndarray, func: callable, single_channel: bool, resize: bool=False, tile_size: int=512):
    """
  Applies a function to a tiff image and returns the processed image.
  :param img: 2D tiff (height, width)
  :param func: a function that takes in a 2/3 channel image and outputs a 3 channel image
  :param single_channel: a flag indicating if the input image is single channel
  :param resize: a flag indicating if the image should be resized before processing
  :param tile_size: the size of the tiles to split the image into
  :return: the processed image
  """
    assert np.ndim(img) == 2
    (width, height) = np.shape(img)[0:2]
    print(width, height)
    if single_channel:
        tile_df = tile_image(img, tile_size, single_channel=True)
    else:
        tile_df = tile_image(img, tile_size)
    pred = []
    for im in tile_df['tile']:
        pred.append([func(im)])
    print(f'recon preds: {np.shape(pred)},raw preds: {np.shape(pred)}')
    ex_df = pd.DataFrame(data=pred, columns=['tile'])
    ex_df['x'] = tile_df['x']
    ex_df['y'] = tile_df['y']
    recon = reconstruct_image(ex_df)
    return np.array(recon)


def erase_loops(img: np.ndarray):
    """
    Given an image, this function erases loops (small closed shapes) that appear within the image.
    The function returns the processed image, and the count of loops that were erased.
    :param img: The image to process.
    :return: A tuple containing the processed image and the count of erased loops.
    """
    if np.ndim(img) == 3:
        img = np.squeeze(img)
    (hh, ww) = img.shape
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = contours[1] if len(contours) == 2 else contours[2]
    contours = contours[0] if len(contours) == 2 else contours[1]
    hierarchy = hierarchy[0]
    count = 0
    result = np.zeros_like(img)
    for component in zip(contours, hierarchy):
        cntr = component[0]
        hier = component[1]
        if (hier[3] > -1) & (hier[2] < 0):
            count += 1
            cv2.drawContours(result, [cntr], -1, (255, 255, 255), 7)
    return (np.expand_dims(result, axis=-1) / 255.0, count)

    
class My_Custom_Generator(keras.utils.Sequence) :
  '''
  returns
  images: (batchsize, tile_size, tile_size, 3)
  labels: (batchsize, tile_size, tile_size, 1)

  '''
  def __init__(self, image_dir, label_dir, image_filenames,  batch_size, resize_shape, aug=False) :
    self.image_filenames = image_filenames
    self.batch_size = batch_size
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.resize_shape = resize_shape
    self.aug = aug


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    lbl = [encode_label(os.path.join(self.label_dir, filename),s=self.resize_shape,buffer=False) for filename in batch_y]
    img = [encode_image(np.array(Image.open(os.path.join(self.image_dir,file_name)).convert('RGB')), s=self.resize_shape) for file_name in batch_x]

    if self.aug: img,lbl =  aug_batch(img,lbl)
    return np.array(img), np.array(lbl)


def plot_predictions(df: pd.DataFrame, model: Model) -> None:
    """
  Visualize the predictions of a model on a sample of tiles from a dataframe.
  The function plots the original tile and the predicted label.
  :param df: a dataframe containing the original tiles and labels
  :param model: a model to predict the labels of the tiles
  :return: None
  """
    tiles = df.sample(30)
    (fig, axes) = plt.subplots(15, 2, figsize=(30, 300))
    for (i, (_, row)) in enumerate(tiles.iterrows()):
        tile = np.array(Image.open(BytesIO(row['tile'])))
        r = i // 2
        x = np.expand_dims(tile, axis=0)
        x = x / 255.0
        pred = model.predict(x, verbose=0)
        pred = np.squeeze(pred, axis=(0, -1))
        axes[r, 0].imshow(tile)
        axes[r, 0].set_title('Original')
        axes[r, 1].imshow(pred)
        axes[r, 1].set_title('Predicted')
    plt.tight_layout()
    plt.show()

def reconstruct_image(df: pd.DataFrame, shape) -> Image:
    """
  Reconstructs an image from its tiled version using data from a dataframe
  :param df: DataFrame containing the tiled image information (x,y,tile)
  :return: Reconstructed image as a PIL Image object
  """
    row = df.iloc[0]
    (w, h, c) = np.shape(row['tile'])
    width = shape[0]
    height = shape[1]
    img = np.zeros(shape=(width, height))
    for (_, row) in tqdm(df.iterrows()):
        try:
            tile = Image.open(BytesIO(row['tile']))
            tile = np.array(tile)
        except:
            tile = np.array(row['tile'])
        x = row['x']
        y = row['y']
        tile = np.array(tile)
        if tile.ndim == 3:
            tile = np.squeeze(tile)
        (tx, ty) = np.shape(img[x:x + w, y:y + h])
        if tx != w or ty != h:
            tile = tile[:tx, :ty]
        img[x:x + w, y:y + h] += tile
        img[x:x + w, y:y + h] = np.clip(img[x:x + w, y:y + h], 0.0, 1.0)
    return img

def tile_image(img: np.ndarray, 
               tile_size: int = 512, 
               single_channel: bool = False, 
               overlap: int = 0) -> pd.DataFrame:
    """
    Tiles a large image into smaller images of a specified size while providing options for customization.

    This function takes a large image and divides it into smaller tiles of a specified size, 
    with optional overlap between tiles. It can also handle single-channel images and 
    provides detailed information about the resulting tiles.

    Args:
        img (np.ndarray): The input image as a NumPy ndarray.
        tile_size (int, optional): The size of the tiles in pixels (default is 512).
        single_channel (bool, optional): Indicates whether the image is single-channel or not (default is False).
        overlap (int, optional): The overlap between tiles in pixels (default is 0).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the tiles and their coordinates.

    Warnings:
        This function suppresses FutureWarnings generated by NumPy to maintain clean output.

    Raises:
        AssertionError: If any tile has dimensions different from the specified tile_size.

    Prints:
        - Information about the input image dimensions, data type, and intensity range.
        - Progress updates for tile processing, indicating the number of tiles processed.
        - Summary statistics including the total number of tiles, blank tiles, and tiles with incorrect dimensions.
        - A message if all tiles are blank, in which case a blank tile is created and included in the output.

    Example:
        To tile a color image 'input_image' with a tile size of 256 pixels and overlap of 64 pixels:
        >>> result_df = tile_image(input_image, tile_size=256, overlap=64)

    Note:
        - Ensure that 'img' is a valid NumPy ndarray representing an image.
        - 'tile_size' should be a positive integer representing the desired tile size.
        - 'single_channel' should be set to True if working with single-channel images (e.g., grayscale).
        - 'overlap' allows for overlapping tiles, useful for applications such as image stitching.

    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    img = np.array(img)
    img = img.astype(np.uint8)
    (width, height) = np.shape(img)[:2]
    min_val = np.min(img)
    max_val = np.max(img)
    print(f'tiling images: {(width, height)}, {(img.dtype, max_val, min_val)} tile size: {tile_size}')
    df = pd.DataFrame(columns=['x', 'y', 'tile'])
    i = 0
    blanks = 0
    d = 0
    for x in range(0, width, tile_size - overlap):
        for y in range(0, height, tile_size - overlap):
            i += 1
            if i % 500 == 0:
                print(f'{i}')
            # tile = img[x:x + tile_size, y:y + tile_size]
            # (h, w) = np.shape(tile)[:2]
            # if h != tile_size or w != tile_size:
            #     new_h = x - (tile_size - h)
            #     new_w = y - (tile_size - w)
            #     tile = img[new_h:new_h + tile_size, new_w:new_w + tile_size]
            #     (h, w) = np.shape(tile)[:2]
            #     assert h == tile_size
            #     assert w == tile_size

            # First, determine if you're on the right boundary for width
            if (x + tile_size) > width: 
                x = width - tile_size

            # Now, determine if you're on the bottom boundary for height
            if (y + tile_size) > height: 
                y = height - tile_size

            # Now extract the tile with the potentially adjusted x and y values
            tile = img[x:x + tile_size, y:y + tile_size]

            # Verify the tile dimensions as a sanity check
            (h, w) = np.shape(tile)[:2]
            assert h == tile_size
            assert w == tile_size
            d += 1
            
            if np.sum(tile) == 0:
                blanks += 1
                continue
            if single_channel:
                if np.ndim(tile) != 3:
                    img_bytes = np.expand_dims(tile, axis=-1)
                else:
                    img_bytes = tile
            else:
                tile = Image.fromarray(tile)
                if tile.mode == 'L' or tile.mode == 'l':
                    tile = tile.convert('RGB')
                img_bytes = np.array(tile)
            df = df.append({'x': x, 'y': y, 'tile': img_bytes}, ignore_index=True)
            del tile
    j = df.shape[0]
    print(f'total images: {j}, total reg images: {j}, blank images: {blanks}, wrong dimensions: {d}')
    if j == 0:
        print('all blanks')
        img_bytes = np.zeros(shape=(tile_size, tile_size))
        df = df.append({'x': x, 'y': y, 'tile': img_bytes}, ignore_index=True)
    return df

def createMask(img: np.ndarray) -> np.ndarray:
    """
    Creates a mask by drawing random lines on a white background and performing a bitwise
    AND operation on the original image with the mask
    :param img: The original image as a numpy array
    :return: The masked image as a numpy array
    """
    (h, w) = np.shape(img)
    mask = np.full((h, h), 0, np.uint8)
    for _ in range(np.random.randint(20, 45)):
        (x1, x2) = (np.random.randint(1, h), np.random.randint(1, h))
        (y1, y2) = (np.random.randint(1, h), np.random.randint(1, h))
        th = int(h / 30)
        if th <= 1:
            th = 2
        thickness = np.random.randint(1, th + 5)
        cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)
    idx = mask > 0
    img[idx] = 0
    return np.array(img)

def predict_from_tiff(img: np.ndarray, model: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
    """
    This function takes in a tiff image and a trained model, and returns the predicted image.

    Parameters:
    img : np.ndarray - The tiff image to be predicted
    model : Model - The trained model to be used for prediction
    fix_lines : bool - A flag indicating whether to fix lines in the image (default: False)
    resize : bool - A flag indicating whether to resize the image (default: False)
    single_channel : bool - A flag indicating whether to use single channel or not (default: True)
    tile_size : int - The size of each tile to be used for prediction (default: 512)

    Returns:
    np.ndarray : The predicted image
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    (width_im, height_im, channels) = np.shape(img)
    print(width_im, height_im)
    if fix_lines:
        tile_df = tile_image(img, tile_size, single_channel=True, overlap=overlap)
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=True, resize=resize)
    else:
        tile_df = tile_image(img, tile_size, overlap=overlap)
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=False, resize=resize)
    recon = reconstruct_image(pred_df, (width_im, height_im))
    del pred_df
    del tile_df
    return np.array(recon)

def ensemble_predict_from_tiff(img: np.ndarray, model_list: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
    """
    This function takes in a tiff image and a trained model, and returns the predicted image.

    Parameters:
    img : np.ndarray - The tiff image to be predicted
    model : Model - The trained model to be used for prediction
    fix_lines : bool - A flag indicating whether to fix lines in the image (default: False)
    resize : bool - A flag indicating whether to resize the image (default: False)
    single_channel : bool - A flag indicating whether to use single channel or not (default: True)
    tile_size : int - The size of each tile to be used for prediction (default: 512)

    Returns:
    np.ndarray : The predicted image
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    (width_im, height_im) = np.shape(img)[:2]
    print(width_im, height_im)
    recon_fin = np.zeros(shape=np.shape(img)[:2], dtype=np.float64)
    if fix_lines:
        single_channel = True
    else:
        single_channel = False
    tile_df = tile_image(img, tile_size, single_channel=single_channel, overlap=overlap)
    num_model = len(model_list)
    scaling_factors = np.linspace(0, 1, num_model)
    for (i, model) in enumerate(model_list):
        print(f'MODEL PREDICTION: {i}')
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=single_channel, resize=resize)
        recon = reconstruct_image(pred_df, (width_im, height_im))
        recon = recon * scaling_factors[i]
        recon_fin += recon.astype(np.float64)
    del pred_df
    del tile_df
    return np.array(recon_fin / np.max(recon_fin))

def parallel_ensemble_predict_from_tiff(img: np.ndarray, model_list: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
    """
    This function takes in a tiff image and a trained model, and returns the predicted image.

    Parameters:
    img : np.ndarray - The tiff image to be predicted
    model : Model - The trained model to be used for prediction
    fix_lines : bool - A flag indicating whether to fix lines in the image (default: False)
    resize : bool - A flag indicating whether to resize the image (default: False)
    single_channel : bool - A flag indicating whether to use single channel or not (default: True)
    tile_size : int - The size of each tile to be used for prediction (default: 512)

    Returns:
    np.ndarray : The predicted image
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    (width_im, height_im) = np.shape(img)
    print(width_im, height_im)
    recon_fin = np.zeros_like(img, dtype=np.float64)
    if fix_lines:
        single_channel = True
    else:
        single_channel = False
    tile_df = tile_image(img, tile_size, single_channel=single_channel, overlap=overlap)
    num_model = len(model_list)
    scaling_factors = np.linspace(0, 1, num_model)
    for (i, model) in enumerate(model_list):
        print(f'MODEL PREDICTION: {i}')
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=single_channel, resize=resize)
        recon = reconstruct_image(pred_df, (width_im, height_im))
        recon = recon * scaling_factors[i]
        recon_fin += recon.astype(np.float64)
    del pred_df
    del tile_df
    return np.array(recon_fin / np.max(recon_fin))

def overlay_positive_values(image, binary_image, c=None):
    overlay = np.zeros(image.shape)
    overlay[binary_image == 1] = image[binary_image == 1]
    overlay = np.where(overlay > 0.2, 1, 0)
    (fig, ax) = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    ax.contour(overlay, cmap='afmhot', alpha=0.7)
    if c is not None:
        plt.title(c)
    del image
    del binary_image

def find_checkpoints(checkpoint_dir, step, h5=False):
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if h5:
            if filename.endswith('.h5'):
                try:
                    checkpoint_number = int(filename.split('_epoch_')[1].split('.h5')[0])
                    if checkpoint_number % step == 0:
                        checkpoints.append(os.path.join(checkpoint_dir, filename))
                except (IndexError, ValueError):
                    continue
        elif filename.endswith('.ckpt.index'):
            try:
                checkpoint_number = int(filename.split('-')[1].split('.ckpt')[0])
                if checkpoint_number % step == 0:
                    checkpoints.append(os.path.join(checkpoint_dir, filename[:-6]))
            except (IndexError, ValueError):
                continue
    return checkpoints

def predict_checkpoints(img, model, checkpoint_dir, step):
    display(img)
    checkpoints = find_checkpoints(checkpoint_dir, step)
    p_maps = []
    for c in checkpoints:
        model.load_weights(c)
        pred_map = predict_from_tiff(img, model, fix_lines=False)
        binary_image = np.where(pred_map > 0.2, 1, 0)
        p_maps.append(binary_image)
    return p_maps

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def unet(sz=(512, 512, 1)):
    x = Input(sz)
    inputs = x
    f = 8
    layers = []
    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(), loss=dice_loss, metrics=[mean_iou])
    return model

def downsize_array(arr: np.ndarray) -> np.ndarray:
    (rows, cols) = arr.shape
    new_rows = rows // 2
    new_cols = cols // 2
    return resize(arr, (new_rows, new_cols))

def crop_to_tile_image(image, target_size):
    (height, width) = np.shape(image)[:2]
    num_tiles_x = width // target_size
    num_tiles_y = height // target_size
    crop_size_x = num_tiles_x * target_size
    crop_size_y = num_tiles_y * target_size
    cropped_image = image[:crop_size_y, :crop_size_x]
    return cropped_image

def rotate_image(image, angle):
    image = tf.convert_to_tensor(image)
    (height, width) = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def full_prediction_tiff(map, save_path, model_list, model_lines):
    """
  Arguments:

  map: a 2D array representing an image to be segmented
  save_path: a filepath to the directory to which the final tiff will be saved
  model_list: a list of ensemble learners to be run on satelliete data
  model_lines: morphological neural network to be run over output of model_list

  Summary:
  The full_prediction_tiff function performs segmentation on an input image by
  breaking it down into smaller chunks and applying a pre-trained segmentation
  model to each chunk. It then reassembles the predicted segmentation maps for each
  chunk into a single output segmentation map for the entire image. The function
  uses a pre-defined checkpoint directory containing pre-trained models to perform
  the segmentation. It prints the dimensions of the input image before and after
  cropping, as well as the total number of chunks to be processed. The function
  returns the output segmentation map.

  """
    image = map
    chunk_size = 512 * 10
    print(f'Image Before Crop: {np.shape(image)}')
    print(f'Image After Crop: {np.shape(image)}')
    image_height = image.shape[0]
    image_width = image.shape[1]
    total_chunks = image_height * image_width / (chunk_size * chunk_size)
    print(f'TOTAL CHUNKS TO BE PROCESSED: {total_chunks}')
    pred_map_full = np.zeros((image_height, image_width))
    print('pred_map intialized')
    cur_chunk = 0
    for i in range(0, image_height, chunk_size):
        for j in range(0, image_width, chunk_size):
            print(f'*****************CHUNKS PROCESSED: {cur_chunk}*****************')
            i_min = i
            i_max = min(i + chunk_size, image_height)
            j_min = j
            j_max = min(j + chunk_size, image_width)
            chunk = image[i_min:i_max, j_min:j_max]
            shape(chunk)
            seg_map = ensemble_predict_from_tiff(chunk, model_list, resize=False, fix_lines=False, overlap=20)
            seg_map = np.where(seg_map > 0.1, 1, 0)
            seg_map = predict_from_tiff(seg_map, model_lines, fix_lines=True, resize=False, overlap=50)
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')
            if np.shape(seg_map) == np.shape(pred_map_full[i_min:i_max, j_min:j_max]):
                print('correct dimensions pasting')
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map
            else:
                print('incorrect dimensions pasting')
                (h, w) = np.shape(pred_map_full[i_min:i_max, j_min:j_max])
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map[:h, :w]
            del chunk
            del seg_map
            cur_chunk += 1
    if save_path is not None:
        try:
            tifffile.imsave(save_path, pred_map_full)
        except:
            print('saving failed')
    print('Retreiving Final Prediction')
    return pred_map_full

def align_meta_data(fp, save_path):
    with rasterio.open(fp) as src0:
        print('original meta data: ', src0.meta)
        meta1 = src0.meta
    with rasterio.open(save_path, 'r+') as src0:
        meta = src0.meta
        src0.transform = meta1['transform']
        src0.crs = meta1['crs']
        t = src0.crs

def line_lengths(binary_image):
    (contours, _) = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths = [cv2.arcLength(contour, closed=False) for contour in contours]
    return lengths

def average_line_length_tf(batch_images):
    batch_images_np = batch_images
    line_lengths_batch = [np.mean(line_lengths(img.squeeze().astype(np.uint8))) for img in batch_images_np]
    avg_line_length = np.mean(line_lengths_batch)
    return tf.constant(avg_line_length, dtype=tf.float32)

def remove_small_objects_np(images, min_size=500, threshold=0.5):
    """
    Remove small objects from a batch of images.
    Args:
        images: numpy array of shape (n, 512, 512, 1)
        min_size: minimum size of connected components to keep
    Returns:
        numpy array of images with small objects removed
    """
    processed_images = []
    for img in images:
        img = img.squeeze()
        img_binary = np.where(img > threshold, True, False)
        img_cleaned = morphology.remove_small_objects(img_binary, min_size)
        processed_images.append(np.where(img_cleaned[..., np.newaxis], 1, 0))
    return np.array(processed_images)

def open_tiff(rasterorig,display_im=True):
  with rasterio.open(rasterorig) as src0:
    print('original meta data: ', src0.meta)
    meta = src0.meta
    if meta['count'] >=3:
      band1= src0.read(1)
      band2= src0.read(2)
      band3= src0.read(3)
      print('3 band tiff')


      map_im = np.dstack((band1,band2,band3))

    elif meta['count'] == 1:
      map_im= src0.read(1)
      print('1 band tiff')
  if display_im:
    display(map_im)
  return map_im



def overlay_positive_values(image, binary_image, c=None):
    overlay = binary_image
    (fig, ax) = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    ax.contour(overlay, cmap='afmhot', alpha=0.7)
    if c is not None:
        plt.title(c)
    del image
    del binary_image

def remove_small_objects_np(images, min_size=500, threshold=0.5):
    """
    Remove small objects from a batch of images.
    Args:
        images: numpy array of shape (n, 512, 512, 1)
        min_size: minimum size of connected components to keep
    Returns:
        numpy array of images with small objects removed
    """
    processed_images = []
    for img in images:
        img = img.squeeze()
        img_binary = np.where(img > threshold, True, False)
        img_cleaned = morphology.remove_small_objects(img_binary, min_size)
        processed_images.append(np.where(img_cleaned[..., np.newaxis], 1, 0))
    return np.array(processed_images)




def full_prediction_tiff_single_model(map, save_path, model):
    """
  Arguments:

  map: a 2D array representing an image to be segmented
  save_path: a filepath to the directory to which the final tiff will be saved
  model_list: a list of ensemble learners to be run on satelliete data
  model_lines: morphological neural network to be run over output of model_list

  Summary:
  The full_prediction_tiff function performs segmentation on an input image by
  breaking it down into smaller chunks and applying a pre-trained segmentation
  model to each chunk. It then reassembles the predicted segmentation maps for each
  chunk into a single output segmentation map for the entire image. The function
  uses a pre-defined checkpoint directory containing pre-trained models to perform
  the segmentation. It prints the dimensions of the input image before and after
  cropping, as well as the total number of chunks to be processed. The function
  returns the output segmentation map.

  """
    image = map
    chunk_size = 512 * 10
    print(f'Image Before Crop: {np.shape(image)}')
    print(f'Image After Crop: {np.shape(image)}')
    image_height = image.shape[0]
    image_width = image.shape[1]
    total_chunks = image_height * image_width / (chunk_size * chunk_size)
    print(f'TOTAL CHUNKS TO BE PROCESSED: {total_chunks}')
    pred_map_full = np.zeros((image_height, image_width))
    print('pred_map intialized')
    cur_chunk = 0
    for i in range(0, image_height, chunk_size):
        for j in range(0, image_width, chunk_size):
            print(f'*****************CHUNKS PROCESSED: {cur_chunk}*****************')
            i_min = i
            i_max = min(i + chunk_size, image_height)
            j_min = j
            j_max = min(j + chunk_size, image_width)
            chunk = image[i_min:i_max, j_min:j_max]
            shape(chunk)
            seg_map = predict_from_tiff(chunk, model, resize=False, fix_lines=False, overlap=20)
            #seg_map = np.where(seg_map > 0.1, 1, 0)
            #seg_map = predict_from_tiff(seg_map, model_lines, fix_lines=True, resize=False, overlap=50)
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')
            if np.shape(seg_map) == np.shape(pred_map_full[i_min:i_max, j_min:j_max]):
                print('correct dimensions pasting')
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map
            else:
                print('incorrect dimensions pasting')
                (h, w) = np.shape(pred_map_full[i_min:i_max, j_min:j_max])
                pred_map_full[i_min:i_max, j_min:j_max] = seg_map[:h, :w]
            del chunk
            del seg_map
            cur_chunk += 1
    if save_path is not None:
        try:
            tifffile.imsave(save_path, pred_map_full)
        except:
            print('saving failed')
    print('Retreiving Final Prediction')
    return pred_map_full



