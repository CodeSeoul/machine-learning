import os
import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from time import time
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from loss import dice_coef, dice_coef_loss
from dataset import create_dataset
from keras.callbacks import EarlyStopping


def train(args):
    df = pd.read_csv(args.dpath + 'metadata.csv')
    df['Image'] = df['Image'].map(lambda x: args.dpath + 'Image/' + x)
    df['Mask'] = df['Mask'].map(lambda x: args.dpath + 'Mask/' + x)

    df_train, df_test = train_test_split(df, test_size=0.1)

    train_dataset = create_dataset(
        df_train, args.batch_size, (args.img_size, args.img_size), training=True)
    test_dataset = create_dataset(
        df_test, args.batch_size, (args.img_size, args.img_size), training=False)

    model = sm.Unet(args.encoder,
                    input_shape=(args.img_size, args.img_size, 3),
                    classes=1,
                    activation='sigmoid',
                    encoder_weights='imagenet')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[sm.metrics.iou_score],
    )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit(
        train_dataset, validation_data=test_dataset, epochs=args.epochs, callbacks=[es])

    if args.save:
        os.makedirs(args.spath, exist_ok=True)
        model.save(os.path.join(args.spath, 'tf_full.h5'))
        hist_df = pd.DataFrame(history.history)
        with open(os.path.join(args.spath, 'history.csv'), mode='w') as f:
            hist_df.to_csv(f)


if __name__ == '__main__':
    basedir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpath", type=str,
                        default=os.path.join(basedir, 'input/'))
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--spath", type=str,
                        default=os.path.join(basedir, 'result/'))
    parser.add_argument("--encoder", type=str, default='mobilenetv2')
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    train(args)
