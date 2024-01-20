def open_tiff(rasterorig, display_im=True):
    with rasterio.open(rasterorig) as src0:
        print('Original meta data: ', src0.meta)
        meta = src0.meta
        if meta['count'] >= 3:
            band1 = src0.read(1)
            band2 = src0.read(2)
            band3 = src0.read(3)
            print('3 band tiff')
            map_im = np.dstack((band1, band2, band3))
        elif meta['count'] == 1:
            map_im = src0.read(1)
            print('1 band tiff')
    return map_im if isinstance(map_im, np.ndarray) else None

def resize_tile_df(df, num):
    # Function to apply OpenCV resize on each tile
    def resize_tile(tile):
        return cv2.resize(tile, (num, num))

    # Update 'tile' column with resized tiles
    df['tile'] = df['tile'].map(resize_tile)

    return df

def retile_labels(label_dir, save_dir, list_sizes, num_tiles_each_size=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in os.listdir(label_dir):
        if file_name.endswith('.png'):
            full_file_path = os.path.join(label_dir, file_name)
            img = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)

            h, w = img.shape
            original_size = (h, w)

            print(f"Processing image: {file_name}")
            print(f"Original image size: {original_size}")

            for size in list_sizes:
                for i in range(num_tiles_each_size):
                    # Choose a random top-left corner for the square
                    y = randint(0, h - size)
                    x = randint(0, w - size)

                    # Crop the square from the image
                    cropped_img = img[y:y+size, x:x+size]

                    # Print crop range
                    print(f"Crop range: x({x}, {x+size}) y({y}, {y+size})")

                    # Upscale the cropped image back to original size
                    upscaled_img = cv2.resize(cropped_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

                    # Save the upscaled image
                    save_path = os.path.join(save_dir, f"{file_name.split('.')[0]}_size{size}_upscaled_{i}.png")
                    cv2.imwrite(save_path, upscaled_img)

                    print(f"Saving upscaled image with tile size {size} as {save_path}")

def convert_rgb_to_grayscale(rgb_image):
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    grayscale_image = 0.299*r + 0.587*g + 0.114*b
    return grayscale_image

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

def lut_display(image, display_min, display_max):
    lut = np.arange(2 ** 16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)

def display(map_im):
    (fig, ax) = plt.subplots(figsize=(50, 50))
    ax.imshow(map_im, interpolation='nearest', cmap='viridis')
    plt.tight_layout()

from typing import List
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

def tile_image(img: np.ndarray, tile_size: int=512, single_channel: bool=False, overlap=0) -> pd.DataFrame:
    """
  Tiles a large image into smaller images of a specified size.
  :param img: the image to be tiled
  :param tile_size: the size of the tiles in pixels
  :param single_channel: whether the image is single channel or not.
  :return: a dataframe containing the tiles and their coordinates
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
            tile = img[x:x + tile_size, y:y + tile_size]
            (h, w) = np.shape(tile)[:2]
            if h != tile_size or w != tile_size:
                new_h = x - (tile_size - h)
                new_w = y - (tile_size - w)
                tile = img[new_h:new_h + tile_size, new_w:new_w + tile_size]
                (h, w) = np.shape(tile)[:2]
                assert h == tile_size
                assert w == tile_size
                d += 1
            # if np.sum(tile) == 0:
            #     blanks += 1
            #     continue
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

    # Additional print statements for debugging
    unique_shapes = df['tile'].apply(lambda x: x.shape).unique()
    unique_dtypes = df['tile'].apply(lambda x: x.dtype).unique()
    min_max_values = df['tile'].apply(lambda x: (np.min(x), np.max(x))).unique()

    print(f"After resizing, the tiles in the DataFrame have shapes: {unique_shapes}")
    print(f"After resizing, the tiles in the DataFrame have data types: {unique_dtypes}")
    print(f"After resizing, the tiles in the DataFrame have min-max values: {min_max_values}")
    return df

def tile_imageV1(img: np.ndarray,
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

def dilate_batchV1(batch, kernel_size=12, iterations=4):
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

    dilated_batch = np.array(dilated_batch)

    # Check and correct dimensions if necessary
    if len(dilated_batch.shape) == 3:
        dilated_batch = np.expand_dims(dilated_batch, axis=-1)

    return dilated_batch



def normalize_to_8bit(image_array):
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    #max_val = 15000

    normalized = (image_array - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    eight_bit = (normalized * 255).astype(np.uint8)  # Scale to [0, 255]

    return eight_bit
    return eight_bit

import numpy as np

def stats(arr):
    """
    Print statistics about a numpy array.

    Parameters:
    - arr (numpy array): The input numpy array.

    Returns:
    - None
    """
    # Basic Information
    print("Array Statistics:")
    print("-" * 50)
    print(f"Number of Dimensions : {arr.ndim}")
    print(f"Shape                : {arr.shape}")
    print(f"Size                 : {arr.size}")
    print(f"Data Type            : {arr.dtype}")

    # Checking if the array has numeric data
    if np.issubdtype(arr.dtype, np.number):
        print(f"Minimum Value        : {np.min(arr)}")
        print(f"Maximum Value        : {np.max(arr)}")
        print(f"Mean Value           : {np.mean(arr)}")
        print(f"Standard Deviation   : {np.std(arr)}")
        # For complex numbers, print real and imaginary parts separately
        if np.iscomplexobj(arr):
            print(f"Real Part - Mean     : {np.mean(arr.real)}")
            print(f"Real Part - Std Dev  : {np.std(arr.real)}")
            print(f"Imaginary Part - Mean: {np.mean(arr.imag)}")
            print(f"Imaginary Part - Std Dev: {np.std(arr.imag)}")

    print("-" * 50)

def check_segmap_zeroes(seg_map):
    if np.sum(seg_map) == 0:
        raise ValueError("Noisy error: seg_map is completely zero after transformation!")