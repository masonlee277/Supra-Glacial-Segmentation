from image_utils import *
from model_utils import *
from config_utils import *
from visualization import *
from file_utils import *
from common_imports import *

def custom_fitV1(model,             # The neural network model to train.
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
               model_lines=None,
               validation_generator=None

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
    my_training_batch_generator = My_Custom_Generator1(images_path_b, labels_path_b, X_train_filenames, batch_size, resize_shape, aug=False)

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
        if (epoch + 1) % 10 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'model_weights_epoch_{epoch + 1}.h5'))
            print(f'Model weights saved for epoch {epoch + 1} in {checkpoint_dir}')

        # Initialize metrics for the current epoch.
        epoch_loss = 0
        epoch_accuracy = 0

        # The inner loop runs for each batch of data in the current epoch.
        step_break = random.randint(50, 60)
        for step in range(steps_per_epoch):

            # Fetch a batch of data from the training generator.
            (x_batch, y_batch) = train_generator.__getitem__(step)


            # Every 10 steps, print the current loss value and display some prediction examples.
            # This gives a regular update on the progress during training.

            if step % step_break == 0:
              try:
                print(f'Step: {step + 1}/{steps_per_epoch} - Loss: {epoch_loss}')
                print_examplesV1(x_batch, y_batch, model, dilate_im, step=step)
              except Exception as e:
                print(f"An error occurred Printing: {e}")
            # TensorFlow's GradientTape is used to monitor operations for which gradients should be computed.
            with tf.GradientTape() as tape:

                # Make predictions for the current batch.
                y_pred = model(x_batch, training=True)

                # If fine-tuning is enabled, use the 'connection_nn' function.
                if fine_tune:
                    #batch_connected = connection_nn(y_pred, model_lines)
                    batch_connected = tf.stop_gradient(connection_nn(y_pred, model_lines))

                    # If dilation is also enabled, perform dilation.
                    # Then, calculate the loss using both the connected and dilated versions.
                    if dilate_im: #Make sure the chosen loss function takes in a mask
                        batch_dilated = dilate_batchV1(y_batch,kernel_size=12, iterations=4)
                        step_loss = loss_fn(y_pred, batch_connected, batch_dilated)
                        del batch_dilated
                    else:
                        step_loss = loss_fn(y_pred, batch_connected)

                # If only dilation is enabled (without fine-tuning), just dilate and calculate the loss.
                elif dilate_im:

                    batch_dilated = dilate_batchV1(y_batch,kernel_size=12, iterations=4)

                    # print("y_true shape:", tf.shape(y_batch))
                    # print("y_pred shape:", tf.shape(y_pred))
                    # print("mask shape:", tf.shape(batch_dilated))

                    step_loss = loss_fn(y_batch, y_pred, batch_dilated)
                    del batch_dilated

                # If neither fine-tuning nor dilation is enabled, calculate the loss directly.
                else:
                    step_loss = loss_fn(y_batch, y_pred)



            # Compute the gradients for the current step.
            gradients = tape.gradient(step_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Update the model's weights using the computed gradients.
            step_accuracy = tf.keras.metrics.binary_accuracy(y_batch, y_pred) # Calculate the accuracy for the current batch.
            #Loss the Batch Loss
            if (step + 1) % 25 == 0:
              try:
                iou_metric = mean_iouV1(y_batch, y_pred)
                wandb.log({"batch_loss": step_loss, "batch_accuracy": step_accuracy, "mean_iou": iou_metric})
                print(f"Successfully logged batch_loss, step_accuracy, mean_iou {step}")

              except Exception as e:
                print(f"An error occurred Logging batch Loss: {e}")
            # Accumulate the loss and accuracy values for the epoch.
            epoch_loss += step_loss
            epoch_accuracy += step_accuracy

            # Free up memory by deleting variables that are no longer needed.
            del y_pred
            del y_batch


        print(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss}')
        print(f'Epoch completion: {100 * (epoch + 1) / epochs:.2f}%')

        # After you've completed training on the current epoch, you can evaluate on the validation set.
        try:
            val_loss = 0
            val_steps = 0
            for val_step in range(len(validation_generator)):
                (x_val_batch, y_val_batch) = validation_generator.__getitem__(val_step)
                y_val_pred = model(x_val_batch, training=False)

                # Calculate validation loss
                val_step_loss = loss_fn(y_val_batch, y_val_pred)
                val_loss += val_step_loss
                val_steps += 1

            val_loss /= val_steps
            wandb.log({"val_loss": val_loss})
            print(f"Validation Loss for epoch {epoch + 1}: {val_loss}")

        except Exception as e:
            print


        try:
            (x_val, y_val) = my_training_batch_generator.__getitem__(1)
            print_examplesV1(x_val, y_val, model, False)
        except Exception as e:
            print(f"An error occurred: {e}")

        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                    callback.on_epoch_end(epoch)


        # At the end of each epoch, check if WandbCallback is in the list of callbacks
        if callbacks is not None and any(isinstance(cb, WandbCallback) for cb in callbacks):
            try:
                # Log the metrics to WandB
                wandb.log({"epoch": epoch, "loss": epoch_loss.numpy(), "accuracy": epoch_accuracy.numpy()})
                print("Successfully logged metrics to WandB.")
            except Exception as e:
                print(f"An error occurred while logging metrics to WandB: {e}")

        epoch_loss /= steps_per_epoch
        epoch_accuracy /= steps_per_epoch
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

    return history

import wandb
from wandb.keras import WandbCallback
def train_modelV1(model: tf.keras.Model,
                  images_path: str,
                  labels_path: str,
                  working_dir: str,
                  epochs: int = 20,
                  batch_size: int = 32,
                  pretrained_weights: str = None,
                  resize_shape: int = 512,
                  fine_tune: bool = False):
    """
    Train a TensorFlow model using the specified image and label data. The model's weights
    and training history are saved to the provided working directory. This function also
    integrates with the Weights & Biases (WandB) platform for real-time monitoring and logging.

    Parameters:
    - model: TensorFlow model to be trained.
    - images_path: Directory path containing training images.
    - labels_path: Directory path containing corresponding label data.
    - working_dir: Directory path to save the model weights and training history.
    - epochs: Number of epochs for training (default is 20).
    - batch_size: Batch size for training (default is 32).
    - pretrained_weights: Path to any pretrained model weights to initialize with (default is None).
    - resize_shape: Resize shape for input images (default is 512).
    - fine_tune: Boolean flag indicating if fine-tuning is to be performed (default is False).

    Returns:
    - None: The function saves model weights, logs metrics to WandB, and saves a training history file.
    """

    # WandB Initialization
    wandb.init(project="RiverNetV1", name="maksed_loss_augmentation")

    # Data Preparation
    X_train_filenames, X_val_filenames = get_file_names(images_path, labels_path)
    STEPS_PER_EPOCH = len(X_train_filenames) / batch_size
    SAVE_PERIOD = 2
    m = int(SAVE_PERIOD * STEPS_PER_EPOCH)

    # Checkpoint Directory Setup
    checkpoint_dir = os.path.join(working_dir, 'checkpoint_dir')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        print('constructed checkpoint')
    else:
        print('checkpoint dir exists')

    # Batch Generators
    my_training_batch_generator = My_Custom_Generator1(images_path, labels_path, X_train_filenames, batch_size, resize_shape, aug=True, seg_connector=False)
    my_validation_batch_generator = My_Custom_Generator1(images_path, labels_path, X_val_filenames, batch_size, resize_shape, aug=False, seg_connector=False)

    # Debugging Outputs
    x, y = my_training_batch_generator.__getitem__(2)
    print(f'x_batch shape: {x.shape}, max: {np.max(x[1])}, min: {np.min(x[1])}')
    print(f'y_batch shape: {y.shape}, Unique Labels: {len(np.unique(y[1]))}')

    # Check y-values
    unique_y_values = np.unique(y)
    if len(unique_y_values) != 2 or np.min(unique_y_values) < 0 or np.max(unique_y_values) > 1:
        print(f"Warning: Found incorrect values in y. Unique values: {unique_y_values}")

        # Clip all values of y between 0 and 1
        y = np.clip(y, 0, 1)

        # Set values less than zero to zero (this is redundant after clipping but keeping for clarity)
        y[y < 0] = 0

    # Check x-values
    if np.max(x) > 1:
        print(f"Warning: Found values greater than 1 in x. Clipping values...")
        # Clip all values of x between 0 and 1
        x = np.clip(x, 0, 1)


    # Model Preparations
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq=m)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-05, verbose=1)
    model.save_weights(checkpoint_path.format(epoch=0))
    plotLearning = PlotLearningV1(model, my_validation_batch_generator)

    if pretrained_weights:
        model.load_weights(pretrained_weights)
        print('pre-trained weights loaded')

    #This will turn of the segconnector reconstruction <-- autodiff difficulties right now
    ####################################
    dilate_im=False
    fine_tune = False
    loss_fn = dice_lossV1

    # Callbacks
    callbacks = [plotLearning, early_stopping, cp_callback, reduce_lr, WandbCallback()]

    # Load Segmentation Connector
    # seg_connector = unetV1()
    # seg_connector_checkpoint_path = "/content/drive/My Drive/Projects/Mapping Glacial Rivers/Data/New_Data/training_dir/seg_connector_checkpoint_dir/large_training_1/epoch_13.h5"
    # seg_connector.load_weights(seg_connector_checkpoint_path)

    ####################################
    # Print Custom Fit Arguments (Debugging)
    custom_fit_args = {
        "model": model,
        "train_generator": my_training_batch_generator,
        "steps_per_epoch": int(STEPS_PER_EPOCH),
        "epochs": epochs,
        "images_path_b": images_path,
        "labels_path_b": labels_path,
        "checkpoint_dir": checkpoint_dir,
        "callbacks": callbacks,
        "dilate_im": dilate_im,
        "optimizer": None,
        "loss_fn": loss_fn,
        "fine_tune": fine_tune
    }

    model.loss = loss_fn
    wandb.config.update(custom_fit_args)

    for key, value in custom_fit_args.items():
        try:
          print(f"{key}: {value}")
        except Exception as e:
            print('Error Printing Custom Fit Arguments')


    # Training
    history = custom_fitV1(
        model=model,
        train_generator=my_training_batch_generator,
        steps_per_epoch=int(STEPS_PER_EPOCH),
        epochs=epochs,
        images_path_b=images_path,
        labels_path_b=labels_path,
        checkpoint_dir=checkpoint_dir,
        callbacks=callbacks,
        dilate_im=dilate_im,
        optimizer=None,
        loss_fn=loss_fn,
        fine_tune=fine_tune,
        model_lines=None,
        validation_generator=my_validation_batch_generator
    )

    # Save Training History
    hist_df = pd.DataFrame(history)
    save_path_history = os.path.join(working_dir, 'history2.csv')
    hist_df.to_csv(save_path_history)