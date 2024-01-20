from image_utils import *
from model_utils import *
from training_utils import *
from config_utils import *
from visualization import *
from common_imports import *

# Function to download a TIFF file from a Google Cloud Storage bucket
def download_tiff_from_bucket(file_path, bucket_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    local_file_path = file_path.split('/')[-1]
    blob.download_to_filename(local_file_path)
    return local_file_path

def process_tiff_file_from_bucket(file_path, bucket_name='greenland_delin_imagery'):
    local_file_path = download_tiff_from_bucket(file_path, bucket_name)
    return open_tiff(local_file_path)



def process_and_upload_predictions(bucket_name, bucket_directory, model_list, seg_connector):
    """
    Download TIFF files from a Google Cloud Storage bucket, perform predictions on them, and upload the results back
    to the same bucket.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        bucket_directory (str): The directory within the bucket where the processed images will be stored.
        model_list: (list) List of models for prediction.
        seg_connector: (str) Connector for segmentation.

    Returns:
        None
    """
    # Instantiate a Google Cloud Storage client
    client = storage.Client()

    # Specify your bucket
    bucket = client.get_bucket(bucket_name)

    # List all the blobs in the bucket
    blobs = bucket.list_blobs()
    local_path = ''
    tiff_filepath = 'temp_download.tif'

    for blob in blobs:
        if blob.name.endswith('.tif'):
            try:
                print(f"Processing {blob.name}")
                blob.download_to_filename(tiff_filepath)

                # Open the .tif file and extract the metadata
                with rasterio.open(tiff_filepath) as src:
                    original_meta = src.meta

                print(original_meta)
                # Perform prediction
                m = open_tiff(tiff_filepath, display_im=False)
                m = normalize_to_8bit(m)
                # Assuming m is normalized to [0, 1] range
                pred_map_full = full_prediction_tiff(m, None, model_list, seg_connector)

                # Convert predictions to binary (0 or 1) and then cast to int8
                pred_map_full = (pred_map_full > 0.5).astype('int8')

                try:
                    mask = (m == 0)
                    pred_map_full = pred_map_full * ~mask
                except:
                    mask = (m[:,:,0] == 0)
                    pred_map_full = pred_map_full * ~mask

                # Add a new dimension to represent single band if needed
                if pred_map_full.ndim == 2:
                    pred_map_full = np.expand_dims(pred_map_full, axis=0)

                # Update metadata for the new file
                new_meta = original_meta.copy()
                new_meta['dtype'] = 'int8'
                new_meta['count'] = pred_map_full.shape[0]
                new_meta['compress'] = 'lzw'

                # Create a new file name for prediction
                new_file_name = local_path + blob.name.replace('.tif', '-pred-v1.tif')

                # Ensure the directory exists before attempting to write the file
                os.makedirs(os.path.dirname(new_file_name), exist_ok=True)

                # Write the new file with updated metadata and prediction data
                with rasterio.open(new_file_name, 'w', **new_meta) as dest:
                    dest.write(pred_map_full)

                # Upload the prediction back to the bucket
                pred_blob = bucket.blob(bucket_directory + blob.name.replace('.tif', '-pred-v1.tif'))
                pred_blob.upload_from_filename(new_file_name)

            except Exception as e:
                print(f"Prediction failed for file: {blob.name}. Error: {str(e)}")
                traceback.print_exc()

            finally:
                # Delete the local files to free up memory
                if os.path.isfile(tiff_filepath):
                    os.remove(tiff_filepath)
                if os.path.isfile(new_file_name):
                    os.remove(new_file_name)

            print(f"Processing of {blob.name} complete.")

import numpy as np
import rasterio
from rasterio.transform import from_origin

def download_tiff(array, filepath='/content/', filename='output.tif', transform=None, crs=None):
    """
    Saves a NumPy array as a compressed TIFF file.

    Parameters:
    array (np.array): The NumPy array to be saved.
    filepath (str): The directory path where the TIFF file will be saved. Defaults to '/content/'.
    filename (str): The name of the TIFF file. Defaults to 'output.tif'.
    transform (rasterio.transform.Affine): Optional. Affine transform for the TIFF file. Defaults to None.
    crs (rasterio.crs.CRS): Optional. Coordinate Reference System for the TIFF file. Defaults to None.
    """
    # Ensure the directory path ends with a slash
    if not filepath.endswith('/'):
        filepath += '/'

    # Create full path
    full_path = filepath + filename

    # If transform is not provided, use a default one
    if transform is None:
        transform = from_origin(0, 0, 1, 1)

    # If crs is not provided, use a default one
    if crs is None:
        crs = rasterio.crs.CRS.from_epsg(4326)  # WGS84

    # Add a new dimension to represent single band if needed
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)

    # Prepare metadata for the TIFF file
    meta = {
        'driver': 'GTiff',
        'height': array.shape[1],
        'width': array.shape[2],
        'count': array.shape[0],
        'dtype': array.dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    # Write the TIFF file
    with rasterio.open(full_path, 'w', **meta) as dst:
        dst.write(array)

# Example usage
# array = np.random.rand(100, 100)  # Example array
# download_tiff(pred_map, filename='my_tiff_file.tif')


def count_files_in_directory(directory_path):
    with os.scandir(directory_path) as entries:
        return sum(1 for entry in entries if entry.is_file())

def shape(x):
    print(np.shape(x))

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
