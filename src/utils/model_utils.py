from image_utils import *
from training_utils import *
from config_utils import *
from visualization import *
from file_utils import *
from common_imports import *

def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, 'float32'))
    return iou

def unetV1(sz=(512, 512, 1)):
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
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(), loss=dice_loss, metrics=mean_iou)
    return model

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


def dice_lossV1(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), dtype=tf.float64)  # Cast to double tensor
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), dtype=tf.float64)  # Cast to double tensor
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

from tensorflow.keras import backend as K
import tensorflow as tf

def mean_iouV1(y_true, y_pred):
    yt0 = tf.cast(y_true[:, :, :, 0], 'float32')
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)), dtype='float32')
    union = tf.math.count_nonzero(tf.add(yt0, yp0), dtype='float32')
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, 'float32'))
    return iou


def predict_from_tiffV1(img: np.ndarray, model: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
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
        tile_df = tile_imageV1(img, tile_size, single_channel=True, overlap=overlap)
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=True, resize=resize)
    else:
        tile_df = tile_image(img, tile_size, overlap=overlap)
        pred_df = predict_from_dataframe_v2(tile_df, model, single_channel=False, resize=resize)
    recon = reconstruct_image(pred_df, (width_im, height_im))
    del pred_df
    del tile_df
    return np.array(recon)

def ensemble_predict_from_tiffV1(img: np.ndarray, model_list: Model, fix_lines: bool=False, resize: bool=False, tile_size: int=512, overlap: int=0) -> np.ndarray:
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
    tile_df = tile_imageV1(img, tile_size, single_channel=single_channel, overlap=overlap)
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

from skimage.transform import resize

def full_prediction_tiff(map, save_path, RiverNet_list, seg_conncector):
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
            print(f'*****************************************************************')
            print(f'*****************CHUNKS PROCESSED: {cur_chunk}*****************')
            i_min = i
            i_max = min(i + chunk_size, image_height)
            j_min = j
            j_max = min(j + chunk_size, image_width)

            if i_max == image_height:  # this is the last row of chunks
                i_min = image_height - chunk_size
                i_max = image_height

            if j_max == image_width:  # this is the last column of chunks
                j_min = image_width - chunk_size
                j_max = image_width

            chunk = image[i_min:i_max, j_min:j_max]
            shape(chunk)
            #seg_map = np.zeros(shape=(chunk_size,chunk_size))
            seg_map = ensemble_predict_from_tiffV1(chunk, RiverNet_list, resize=False, fix_lines=False, overlap=50)
            print(' ********************* Finished Ensemble Prediction with RiverNet for Chunk ********************')
            seg_map = np.where(seg_map > 0.1, 1, 0)
            check_segmap_zeroes(seg_map)

            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            print(f'seg_map: {np.shape(seg_map)}, pred_map: {np.shape(pred_map_full[i_min:i_max, j_min:j_max])}')

            #One Pass At Low Resolution
            seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)
            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            seg_map = np.where(seg_map > 0.1, 1, 0)
            if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            check_segmap_zeroes(seg_map)

            # for i in range(6):
            #   offset=50
            #   print(f'^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            #   print(f'SegConnector Prediction: {i}')
            #   if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)
            #   seg_map = predict_from_tiffV1(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=50+offset*i)
            #   check_segmap_zeroes(seg_map)

            ##Higher Overlap
            #seg_map = predict_from_tiff(seg_map, seg_conncector, fix_lines=True, resize=False, tile_size = 512, overlap=150)

            #####################################3
            #if seg_map.ndim == 2: seg_map = np.expand_dims(seg_map, axis=-1)

            # print(f"Original shape: {np.shape(seg_map)}")  # Debug print
            # original_shape = seg_map.shape  # Storing original shape for later

            # # Downscale seg_map by 2 on each dimension using scikit-image
            # seg_map_downscaled = resize(seg_map, (original_shape[0]//2, original_shape[1]//2, original_shape[2]), anti_aliasing=True)
            # print(f"Shape after downscaling: {np.shape(seg_map_downscaled)}")  # Debug print

            # # Make prediction on the downscaled image
            # seg_map_downscaled = predict_from_tiff(seg_map_downscaled, seg_connector, fix_lines=True, resize=False, tile_size=512, overlap=50)
            # print(f"Shape after prediction: {np.shape(seg_map_downscaled)}")  # Debug print

            # if seg_map_downscaled.ndim == 2: seg_map_downscaled = np.expand_dims(seg_map_downscaled, axis=-1)

            # # Upscale seg_map back to its original size using scikit-image
            # seg_map = np.squeeze(resize(seg_map_downscaled, original_shape, anti_aliasing=True))

            # print(f"Shape after upscaling: {np.shape(seg_map)}")  # Debug print

            # # Make sure the rescaled shape is the same as the original shape
            # #assert seg_map.shape == original_shape, "Shapes are not equal"


            #################################
            # Adjusting for overlap on the boundary
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


            # chunk_border_thickness = 30  # You can adjust this if you want thicker borders
            # pred_map_full[i_min:i_min+chunk_border_thickness, j_min:j_max] = 1
            # pred_map_full[i_max-chunk_border_thickness:i_max, j_min:j_max] = 1
            # pred_map_full[i_min:i_max, j_min:j_min+chunk_border_thickness] = 1
            # pred_map_full[i_min:i_max, j_max-chunk_border_thickness:j_max] = 1
    if save_path is not None:
        try:
            tifffile.imsave(save_path, pred_map_full)
        except:
            print('saving failed')
    print('Retreiving Final Prediction')
    return pred_map_full



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
