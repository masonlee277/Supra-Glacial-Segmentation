from tensorflow import keras
import numpy as np
import os
from PIL import Image
import copy as cp
import gc  # Import garbage collector
from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf


class My_Custom_Generator1(keras.utils.Sequence):
    '''
    returns
    images: (batchsize, tile_size, tile_size, 3)
    labels: (batchsize, tile_size, tile_size, 1)
    '''

    def __init__(self, image_dir, label_dir, image_filenames, batch_size, resize_shape, aug=False, seg_connector=False, use_parallel=True, tfrecord_path=None):
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.resize_shape = resize_shape
        self.aug = aug
        self.seg_connector = seg_connector
        self.print_time = False
        self.use_parallel = use_parallel
        self.tfrecord_path = tfrecord_path

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def _parse_tfrecord(self, record):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        image = tf.image.decode_png(example['image'], channels=3)
        label = tf.image.decode_png(example['label'], channels=1)
        return image, label

    def track_time(self, start_time, message):
        if self.print_time:
            elapsed_time = time.time() - start_time
            print(f'{message}: {elapsed_time:.4f} seconds')
            return time.time()  # Return the current time for the next tracking
        return start_time  # If print_time is False, just return the original start_time

    def process_image_func(self, file_name, image_dir, resize_shape):
        image = np.array(Image.open(os.path.join(image_dir, file_name)).convert('RGB'))
        return encode_image(image, s=resize_shape)

    def process_label_func(self, file_path, resize_shape=512):
        return encode_label(file_path, s=resize_shape, buffer=False)

    def process_seg_connector_func(self, i, augmented):
        return encode_label(createMaskv1(i, add_noise=True, thickness=4), s=512, buffer=False)

    def process_seg_connector_func_no_aug(self, i, augmented):
        return encode_label(i, s=512, buffer=False)

    def __getitem__(self, idx):
      ## When using tf_record we return the img, labels without processing. (not fully functional processing yet                                                                                )
      if self.tfrecord_path:
          raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
          parsed_dataset = raw_dataset.map(self._parse_tfrecord)
          batch_data = list(parsed_dataset.skip(idx*self.batch_size).take(self.batch_size))
          img_tensor = tf.stack([item[0] for item in batch_data])
          lbl_tensor = tf.stack([item[1] for item in batch_data])
          return img_tensor.numpy(), lbl_tensor.numpy()
      start_time = time.time() if self.print_time else None  # Record the start time if print_time is True

      batch_y = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
      if not batch_y or not isinstance(batch_y[0], str):
          raise ValueError(f"Unexpected batch_y value: {batch_y}")

      start_time = self.track_time(start_time, 'Time taken to get batch_y')

      lbl = [self.process_label_func(os.path.join(self.label_dir, filename), self.resize_shape) for filename in batch_y]
      start_time = self.track_time(start_time, 'Time taken to get lbl')

      if self.seg_connector:
          if self.aug:
              start_time_augmentation = self.track_time(time.time(), 'Starting Augmentation') if self.print_time else None  # Record the start time for augmentation
              augmented, lbl = aug_batch_segconnect(cp.deepcopy(lbl))

              if self.use_parallel:
                  img = Parallel(n_jobs=-1)(
                      delayed(self.process_seg_connector_func)(i, augmented) for i in augmented)
              else:
                  img = [self.process_seg_connector_func(i, augmented) for i in augmented]

              del augmented  # Explicitly delete temporary variables
              self.track_time(start_time_augmentation, 'Time taken for augmentation')
          else:
              if self.use_parallel:
                  img = Parallel(n_jobs=-1)(
                      delayed(self.process_seg_connector_func)(i, lbl) for i in lbl)
              else:
                  img = [self.process_seg_connector_func_no_aug(i, lbl) for i in lbl]

          if self.use_parallel:
              lbl = Parallel(n_jobs=-1)(
                  delayed(encode_label)(im, s=512) for im in lbl)
          else:
              img = [self.process_label_func(os.path.join(self.image_dir, filename), self.resize_shape) for filename in batch_y]
              lbl = [encode_label(im, s=512) for im in lbl]

          self.track_time(start_time, 'Time taken for seg_connector processing')
      else:
          if self.use_parallel:
              img = Parallel(n_jobs=-1)(
                  delayed(self.process_image_func)(file_name, self.image_dir, self.resize_shape) for file_name in batch_y)
          else:
              img = [self.process_image_func(file_name, self.image_dir, self.resize_shape) for file_name in batch_y]

          if self.aug:
              start_time_augmentation = self.track_time(time.time(), 'Starting Augmentation') if self.print_time else None  # Record the start time for augmentation
              img, lbl = aug_batchV1(cp.deepcopy(img), cp.deepcopy(lbl))
              self.track_time(start_time_augmentation, 'Time taken for augmentation')
          # else:
          #     #print('Regular call')

          self.track_time(start_time, 'Time taken for regular processing')

      start_time_tensor_conversion = time.time() if self.print_time else None  # Record the start time for tensor conversion

      img_tensor = np.array(img)
      lbl_tensor = np.array(lbl)

      self.track_time(start_time_tensor_conversion, 'Time taken for tensor conversion')
      del img, lbl

      return img_tensor, lbl_tensor




# inheritance for training process plot
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from tensorflow import keras

class PlotLearningV1(keras.callbacks.Callback):

    def __init__(self, model_instance, generator):
        super().__init__()
        self.model_instance = model_instance
        self.generator = generator

    def on_train_begin(self, logs={}):
        x_val, y_val = self.generator.__getitem__(1)
        self.lbl = np.squeeze(y_val[0])
        self.img = np.squeeze(x_val[0])
        self.i = 0

    def on_epoch_end(self, epoch, logs={}):
        try:
            pred = self.model_instance.predict(np.expand_dims(cp.deepcopy(self.img), axis=0))
            pred = np.squeeze(pred)
            img = cp.deepcopy(self.img)
            lbl = cp.deepcopy(self.lbl)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Add titles for each subplot
            axs[0].set_title("Original Image")
            axs[1].set_title("Predicted Mask")
            axs[2].set_title("Ground Truth Mask")

            # Display the images
            axs[0].imshow(img)
            axs[0].axis('off')

            axs[1].imshow(pred)
            axs[1].axis('off')

            axs[2].imshow(lbl)
            axs[2].axis('off')

            if epoch is not None and logs is not None and "loss" in logs:
                # Check if "loss" exists in logs before accessing it
                print('i=', self.i, 'loss=', logs["loss"])
                self.i += 1
                fig.suptitle(f'Epoch: {epoch + 1}, Loss: {logs["loss"]:.4f}', fontsize=16)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred in on_epoch_end: {str(e)}")
def bezier_curve(points, n=1000):
    N = len(points)
    x_vals, y_vals = np.array([0.0] * n), np.array([0.0] * n)
    t = np.linspace(0, 1, n)

    for i in range(N):
        x_vals += binom(N - 1, i) * ((1 - t) ** (N - 1 - i)) * (t ** i) * points[i][0]
        y_vals += binom(N - 1, i) * ((1 - t) ** (N - 1 - i)) * (t ** i) * points[i][1]

    return list(zip(x_vals.astype(int), y_vals.astype(int)))

def binom(n, k):
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))

def createMaskv1(img, add_noise=True, thickness=2):
    if np.ndim(img) == 3:
        img = np.squeeze(img)

    h, w = np.shape(img)
    mask = np.full((h, w), 0, np.uint8)
    if np.sum(img) < 300 or np.sum(img) > 202144.0:
       #print('zero')
       return np.array(img)
    else:
      #print('edge')

      for _ in range(int(np.clip(np.random.normal(20, 7), 0, 40))):  # Mean is 10, standard deviation is 7
          control_points = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(4)]
          points = bezier_curve(control_points)

          for point in points:
              x, y = point
              if 0 <= x < h and 0 <= y < w:
                  cv2.circle(mask, (y, x), thickness, (1), -1)

      # if add_noise:
      #     noise = np.random.normal(0, 0.2, mask.shape)
      #     mask = np.clip(mask + noise, 0, 1)
      idx = (mask > 0)
      img[idx] = 0
      if add_noise:
        if np.random.rand() < 0.65:

          # Adjust Gaussian noise std based on image range
          range_value = np.max(img) - np.min(img)
          #print(range_value)
          scalar = random.uniform(0.01, 0.2)
          adjusted_std = scalar * range_value
          img = add_gaussian_noise(img, std=adjusted_std)

      return np.array(img)



def add_artifacts(image, num_dots=30, dot_size=3, blur_size=3, **kwargs):
    """
    Add small white artifacts on the image.

    Parameters:
    - image: The input image to modify.
    - num_dots: Number of dots/artifacts to add.
    - dot_size: Size of the dots.
    - blur_size: Size of Gaussian blur kernel to blur the artifacts.

    Returns:
    - Modified image with artifacts.
    """

    modified_image = image.copy()  # Work on a copy to preserve the original image
    height, width = modified_image.shape[:2]
    max_val = np.max(modified_image)  # Get the maximum pixel value in the image
    max_val = 1
    for _ in range(num_dots):
        # Randomly select a center for the dot
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        # Draw the dot with the intensity set to the maximum pixel value of the image
        cv2.circle(modified_image, (center_x, center_y), dot_size, (max_val, max_val, max_val), -1)

    # Blur the image slightly to make the artifacts softer
    blurred_image = cv2.GaussianBlur(modified_image, (blur_size, blur_size), 0)

    # Ensure the image has 3 dimensions
    if len(blurred_image.shape) == 2:
        blurred_image = np.expand_dims(blurred_image, -1)

    return blurred_image


def aug_batch_segconnect(batch_y):
    """
    The 'aug_batch_segconnect' function applies simplified image augmentations
    specifically flipping and Gaussian noise to input mask list, 'batch_y'.
    It returns the augmented and original mask lists.
    """

    yn_original = []
    yn_augmented = []

    for mask in batch_y:

        alpha = random.uniform(30, 60)
        sigma = alpha * random.uniform(0.02, 0.07)
        alpha_affine = alpha * random.uniform(0.02, 0.05)

        original_height, original_width = mask.shape[:2]

        # Decide random crop size
        crop_height = int(original_height * random.uniform(0.6, 1))
        crop_width = int(original_width * random.uniform(0.6, 1))

        # Calculate random start coordinates for the cropping to ensure we get the same crop
        start_x = np.random.randint(0, original_width - crop_width + 1)
        start_y = np.random.randint(0, original_height - crop_height + 1)

        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Apply cropping based on calculated coordinates
        mask_cropped = mask[start_y:end_y, start_x:end_x]
        mask_cropped_resized = cv2.resize(mask_cropped, (original_width, original_height))

        m_aug = mask.copy()
        dots = int(random.uniform(10, 50))
        if np.random.rand() < 0.9:
            m_aug = add_artifacts(m_aug, num_dots=dots, dot_size=3, blur_size=3)

        m_aug_cropped = m_aug[start_y:end_y, start_x:end_x]
        m_aug_cropped_resized = cv2.resize(m_aug_cropped, (original_width, original_height))

        # Apply the random ElasticTransform
        elastic_transform = A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=1)

        if np.random.rand() < 0.4:
          elastic_mask = mask_cropped_resized
          elastic_aug = elastic_transform(image=m_aug_cropped_resized)["image"]
        else:
          elastic_mask = mask_cropped_resized
          elastic_aug = m_aug_cropped_resized
        # Adjust Gaussian noise std based on image range
        range_value = np.max(mask) - np.min(mask)
        scalar = random.uniform(0.1, 0.3)
        adjusted_std = scalar * range_value


        ##Optionally apply horizontal flip to both original and augmented
        if np.random.rand() < 0.5:
            elastic_mask = np.flip(elastic_mask, axis=1)
            elastic_aug = np.flip(elastic_aug, axis=1)

        ## Optionally apply vertical flip to both original and augmented
        if np.random.rand() < 0.5:
            elastic_mask = np.flip(elastic_mask, axis=0)
            elastic_aug = np.flip(elastic_aug, axis=0)

        yn_original.append(elastic_mask)
        yn_augmented.append(elastic_aug)

    return np.array(yn_augmented), np.array(yn_original)



def add_gaussian_noise(image, mean=0., std=0.01):
    """Add gaussian noise to a image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, np.min(image), np.max(image))  # clip based on the original image range
    return noisy_image



def augment_image(image, mask, aug, thresh):
    mask_sum = 0
    it = 0
    while mask_sum <= thresh and it <= 5:
        augmented = aug(image=image, mask=mask)
        im_aug = augmented['image']
        m_aug = augmented['mask']
        mask_sum = int(np.sum(m_aug))
        it += 1
    return im_aug, m_aug

def aug_batchV1(batch_x, batch_y):
    (oh, ow) = (512, 512)
    aug = A.Compose([A.RandomSizedCrop(min_max_height=(200, 456), height=oh, width=ow, p=0.4),
                     A.VerticalFlip(p=0.5),
                     A.RandomRotate90(p=0.5),
                     A.OneOf([A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.3, p=.1),
                              A.Sharpen(p=.4),
                              A.Solarize(threshold=0.05, p=1)], p=0.3),
                     A.ColorJitter(brightness=0.7, contrast=0.4, saturation=0.4, hue=0.3, always_apply=False, p=0.4),
                     A.GaussNoise(var_limit=(0, 0.05), p=0.4)])

    thresh = 100
    results = Parallel(n_jobs=-1)(delayed(augment_image)(image, mask, aug, thresh) for image, mask in zip(batch_x, batch_y))
    xn, yn = zip(*results)

    return list(xn), list(yn)


