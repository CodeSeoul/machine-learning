import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Blur(blur_limit=3, p=0.5),
    # A.RandomRotate90(p=1) ,
    # A.Rotate(limit=90,p=0.5) ,
    # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)

])

def modify_mask(mask):
    mask = np.expand_dims(mask, axis=2)
    t_mask = np.zeros(mask.shape)
    np.place(t_mask[:, :, 0], mask[:, :, 0] >= 0.5, 1)
    return t_mask


def map_function(img, mask, img_size, training):
    img, mask = plt.imread(img.decode())[:, :, :3], plt.imread(mask.decode())
    img = cv2.resize(img, img_size)
    mask = modify_mask(cv2.resize(mask, img_size))

    img = img/255.0
    if training == True:
        transformed = transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

    return img.astype(np.float32), mask.astype(np.float32)


def create_dataset(data, bs, img_size, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((data['Image'], data['Mask']))
    dataset = dataset.shuffle(100)
    dataset = dataset.map(lambda img, mask: tf.numpy_function(
        map_function, [img, mask, img_size, training], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(bs)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
