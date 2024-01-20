from image_utils import *
from model_utils import *
from training_utils import *
from config_utils import *
from file_utils import *
from common_imports import *

def plot_imagesV1(river_labels, river_images):
    batch_size, height, width, *_ = river_labels.shape  # Get shape info, ignore additional dimensions
    num_images = min(len(river_labels), len(river_images))  # Get the minimum number of paired images

    for i in range(num_images):
        plt.figure(figsize=(12, 6))

        # Handle grayscale and color images for river_labels
        plt.subplot(1, 2, 1)
        label_data = river_labels[i]

        # If the range is 0-1, scale it
        if np.min(label_data) >= 0 and np.max(label_data) <= 1:
            label_data = label_data * 255

        if river_labels.shape[-1] == 1:
            plt.imshow(np.squeeze(label_data), cmap='jet')
        else:
            plt.imshow(label_data.astype(np.uint8), cmap='jet')

        # Set title with range for river_labels
        label_range = np.max(label_data) - np.min(label_data)
        plt.title(f"Label {i+1} (Range: {label_range:.2f})")
        plt.axis("off")

        # Handle grayscale and color images for river_images
        plt.subplot(1, 2, 2)
        image_data = river_images[i]

        # If the range is 0-1, scale it
        if np.min(image_data) >= 0 and np.max(image_data) <= 1:
            image_data = image_data * 255

        if river_images.shape[-1] == 1:
            plt.imshow(np.squeeze(image_data), cmap='jet')
        else:
            plt.imshow(image_data.astype(np.uint8), cmap='jet')

        # Set title with range for river_images
        image_range = np.max(image_data) - np.min(image_data)
        plt.title(f"Image {i+1} (Range: {image_range:.2f})")
        plt.axis("off")

        plt.show()


def plot_imagesV2(*args):
    num_lists = len(args)
    num_images = min(len(arg) for arg in args)  # Get the minimum number of images across all lists

    for i in range(num_images):
        plt.figure(figsize=(12 * num_lists, 6))
        for j, image_list in enumerate(args):
            image_data = image_list[i]

            # Plot image data
            plt.subplot(1, num_lists, j + 1)
            plot_data(image_data, i, f'List {j + 1}')

        plt.show()

def plot_data(data, index, title_prefix):
    # If the range is 0-1, scale it
    if np.min(data) >= 0 and np.max(data) <= 1:
        data = data * 255

    if data.shape[-1] == 1:
        plt.imshow(np.squeeze(data), cmap='gray')
    else:
        plt.imshow(data.astype(np.uint8))

    # Set title with range for data
    data_range = np.max(data) - np.min(data)
    plt.title(f"{title_prefix} Image {index + 1} (Range: {data_range:.2f})")
    plt.axis("off")

def display_overlay(map, pred_map):

    """
    display_overlay Function
    -------------------------
    This function is used to overlay a prediction map (pred_map) on top of a base map.
    The function also transforms zero values in the pred_map to NaNs.

    Parameters:
      map (np.array): A 2D numpy array that represents the base map.
      pred_map (np.array): A 2D numpy array that represents the prediction map. The zero values in this map will be replaced by NaNs.

    Returns:
      This function does not return a value. It displays a figure with the overlay of pred_map on top of the map.

    Example:
      display_overlay(map_array, pred_map_array)
    """

    # Convert 0s to NaNs in pred_map
    pred_map = np.where(pred_map == 0, np.nan, pred_map)

    plt.figure(figsize=(20, 20))

    # Display the base map
    plt.imshow(map, cmap='gray')

    # Overlay pred_map onto the base map
    plt.imshow(pred_map, cmap='jet', alpha=0.5)

    # Display the result
    plt.show()

def print_examplesV1(x_val, y_val, model, dilate_im, step=None, num_examples=2):
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
            lbl_dilated = dilate_batchV1(np.expand_dims(y_val[i], axis=0))[0]
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
