import tensorflow.keras as keras
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import sklearn.preprocessing
import argparse
import re
import datetime

import networks
import dataload

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser("Preprocessing of images in /data/raw/.")


def main(args):
    if re.match(".*/src$", os.getcwd()):
        os.chdir("../")  # change directory to root directory
    df = pd.read_csv("./data/annotations.csv")
    df = df.sample(frac =1) # shuffle dataframe
    df["id"] = df["id"].apply(lambda x: str(x) + ".png")

  
    le = sklearn.preprocessing.LabelEncoder() 
    labels = le.fit_transform(df["name"])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df[['name']])
    
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory='./data/processed',
        x_col="id",
        y_col="name",
        batch_size=32,
        class_mode="sparse")

    model = networks.network_factory("Inception",
            filters=32,
            dim_output=len(df["name"].unique()))

    model.compile(loss="sparse_categorical_crossentropy", 
            optimizer="adam", 
            metrics=["accuracy"])
    
    opt = keras.optimizers.Adam(learning_rate=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False)

    callbacks = [keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", str(datetime.datetime.now())),
        histogram_freq=1,
        profile_batch=0)]

    model.fit_generator(train_generator,
            steps_per_epoch=None,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=None,
            validation_steps=None,
            validation_freq=1,
            class_weight=None,
            max_queue_size=10,
            workers=6,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0)

    model.summary()


if __name__ == "__main__":
    parser.add_argument("-e", "--epochs", default="10", type=int, help="Sets number of epochs")
    parser.add_argument("-l", "--learning-rate", default="0.0001", type=float, help="Sets learning rate")
    parser.add_argument("-b", "--batch-size", default="32", type=int, help="Sets batch size")
    parser.add_argument("-n", "--network", default="Inception", type=str, help="Type of network")
    parser.add_argument("-s", "--split", default="0.8", type=float, help="Split ratio of dataset")

    parsed_args = parser.parse_args()
    main(parsed_args)

