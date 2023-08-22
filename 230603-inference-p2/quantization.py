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

basedir = os.getcwd()
model_path = os.path.join(basedir, 'result', 'tf_full.h5')
history_path = os.path.join(basedir, 'result', 'history.csv')
with tf.device('/cpu:0'):
    unet_model = tf.keras.models.load_model(
        model_path, custom_objects={"iou_score": sm.metrics.iou_score})


basedir = os.getcwd()
dpath = os.path.join(basedir, "input/")
df = pd.read_csv(dpath + 'metadata.csv')
df['Image'] = df['Image'].map(lambda x: dpath + 'Image/' + x)
df['Mask'] = df['Mask'].map(lambda x: dpath + 'Mask/' + x)
df_train, df_test = train_test_split(df, test_size=0.1)


def representative_data_gen():
    img_size, bs, num = 64, 1, 10
    training = True
    dataset = tf.data.Dataset.from_tensor_slices(
        (df_train['Image'], df_train['Mask']))
    dataset = dataset.map(lambda img, mask: tf.numpy_function(
        map_function, [img, mask, (img_size, img_size), training], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(bs)

    for input_img, input_mask in dataset.take(num):
        yield [input_img]


def get_default_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def get_quantized_tflite_model(model, repr_data, quant_type):
    converter = tf.lite.TFLiteConverter.from_keras_model(unet_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = repr_data

    if quant_type == 'int':
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quant_type == 'float':
        converter.target_spec.supported_types = [tf.float16]

    tflite_model_quant = converter.convert()

    return tflite_model_quant


tflite_models_dir = pathlib.Path(os.path.join(basedir, "tflite_models"))
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model = get_default_tflite_model(unet_model)
tflite_model_file = tflite_models_dir/"unet.tflite"
tflite_model_file.write_bytes(tflite_model)

tflite_model_float = get_quantized_tflite_model(
    unet_model, representative_data_gen, quant_type='float')
tflite_model_float_file = tflite_models_dir/"unet_fp16.tflite"
tflite_model_float_file.write_bytes(tflite_model_float)

tflite_model_int = get_quantized_tflite_model(
    unet_model, representative_data_gen, quant_type='int')
tflite_model_int_file = tflite_models_dir/"unet_int8.tflite"
tflite_model_int_file.write_bytes(tflite_model_int)


def get_file_sizes(model_file):
    print("model in Mb:", os.path.getsize(model_file) / float(2**20))


def get_io_types(model_file):
    interpreter = tf.lite.Interpreter(model_content=model_file)
    input_type = interpreter.get_input_details()[0]['dtype']
    output_type = interpreter.get_output_details()[0]['dtype']
    print(f'input: {input_type} output: {output_type}')


get_io_types(tflite_model)
get_file_sizes(tflite_model_file)

get_io_types(tflite_model_float)
get_file_sizes(tflite_model_float_file)

get_io_types(tflite_model_int)
get_file_sizes(tflite_model_int_file)


def get_test_image():
    url = 'https://i.tribune.com.pk/media/images/Floods1656337686-1/Floods1656337686-1.jpg'
    IMG_SIZE = (64, 64)

    response = requests.get(url)
    bytes_im = io.BytesIO(response.content)
    img = np.array(Image.open(bytes_im))[:, :, :3]

    img = img/255.0
    img = cv2.resize(img, IMG_SIZE)
    return img


def run_tflite_model(test_image, tflite_file):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        print("entered quant")
        input_scale, input_zero_point = input_details["quantization"]
        test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(
        test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    return output


test_image = get_test_image()
output = run_tflite_model(test_image, tflite_model_int_file)


def plot_image(img, mask):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

# plot_image(test_image, output)
