import tensorflow.keras as keras
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import sklearn.preprocessing
import argparse
import networks
import re
import dataload

parser = argparse.ArgumentParser("Preprocessing of images in /data/raw/.")


def main(args):
    if re.match(".*/src$", os.getcwd()):
        os.chdir("../")  # change directory to root directory
    df = pd.read_csv("./data/annotations.csv")

    le = sklearn.preprocessing.LabelEncoder() 
    labels = le.fit_transform(df["name"])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    ids = df["id"].astype(int).to_numpy()

    dg = dataload.DataLoader(ids, labels, args.batch_size)
    model = networks.InceptionNet(filters=32, dim_output=len(df["name"].unique()))

    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    opt = keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.fit_generator(dg, steps_per_epoch=None, epochs=args.epochs, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=6, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    model.summray()


if __name__ == "__main__":
    parser.add_argument("-e", "--epochs", default="10", type=int, help="Number of epochs")
    parser.add_argument("-l", "--learning-rate", default="0.0001", type=float, help="Learning rate")
    parser.add_argument("-b", "--batch-size", default="32", type=int, help="Batch size")

    parsed_args = parser.parse_args()
    main(parsed_args)

