import os
import cv2
import tensorflow as tf
import segmentation_models as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataset import modify_mask
import requests
import io
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from dataset import create_dataset, map_function
import pathlib
import tensorflow_model_optimization as tfmot
from tensorflow import keras

# Recreate the exact same model, including its weights and the optimizer
basedir = os.getcwd()
model_path = os.path.join(basedir, 'result', 'tf_full.h5')
history_path = os.path.join(basedir, 'result', 'history.csv')
# with tf.device('/cpu:0'):
unet_model = tf.keras.models.load_model(
    model_path, custom_objects={"iou_score": sm.metrics.iou_score})

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
basedir = os.getcwd()
dpath = os.path.join(basedir, "input/")
df = pd.read_csv(dpath + 'metadata.csv')
df['Image'] = df['Image'].map(lambda x: dpath + 'Image/' + x)
df['Mask'] = df['Mask'].map(lambda x: dpath + 'Mask/' + x)
df_train, df_test = train_test_split(df, test_size=0.1)

batch_size = 1
epochs = 3
validation_split = 0.1
img_size = 64

num_images = int(len(df_train) * (1 - validation_split))

end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                             final_sparsity=0.80,
                                                             begin_step=0,
                                                             end_step=end_step)
}

model_for_pruning = prune_low_magnitude(unet_model, **pruning_params)

model_for_pruning.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[sm.metrics.iou_score],
)


logdir = os.path.join(basedir, 'logs')
os.makedirs(logdir, exist_ok=True)
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

train_df = df_train[:int(len(df_train) * (1 - validation_split))]
val_df = df_train[int(len(df_train) * (1 - validation_split)):]

train_dataset = create_dataset(
    train_df, batch_size, (img_size, img_size), training=True)
val_dataset = create_dataset(
    val_df, batch_size, (img_size, img_size), training=True)


history = model_for_pruning.fit(train_dataset,
                                batch_size=batch_size, epochs=epochs, validation_data=val_dataset,
                                callbacks=callbacks)

spath = os.path.join(basedir, 'result/')
hist_df = pd.DataFrame(history.history)
with open(os.path.join(spath, 'history_pruning.csv'), mode='w') as f:
    hist_df.to_csv(f)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

pruned_keras_file = os.path.join(spath, 'tf_pruning.h5')
tf.keras.models.save_model(
    model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)
