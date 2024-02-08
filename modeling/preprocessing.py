import tensorflow as tf
RSEED = 42

def img_dataset_from_dir_and_split_train_val(data_path):
    ''' 
    Input: Path to parent directory where the images are stored in sufolders/classes
    Return training and validation data set at 80:20 ratio. Crops and resizes images to 224x224px. RSEED = 42
    '''
    datasets = tf.keras.utils.image_dataset_from_directory(
        data_path, 
        validation_split = 0.2,
        subset = "both", 
        seed = RSEED,
        image_size = (224, 224),
        crop_to_aspect_ratio = True,
        label_mode = 'categorical'
    )
    return datasets