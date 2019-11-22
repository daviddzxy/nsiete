import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import os
import sklearn.preprocessing
import argparse
import re
import datetime

import networks

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main(args):
    if re.match(".*/src$", os.getcwd()):
        os.chdir("../")  #  change directory to root directory
    
    date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    df = pd.read_csv("./data/annotations.csv")

    if args.dog_breeds is not None:
        df = df[df["name"].isin(args.dog_breeds)]

    df = df.sample(frac=1)  # shuffle dataframe
    df["id"] = df["id"].apply(lambda x: str(x) + ".png")

    #  stratify keeps the ratio of classes in train and test
    train, test = train_test_split(
            df,
            train_size=0.2,
            random_state=0,
            stratify=df[['name']])
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory='./data/processed',
        x_col="id",  # image filename
        y_col="name",
        batch_size=args.batch_size,
        class_mode="sparse")

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory='./data/processed',
        x_col="id",
        y_col="name",
        batch_size=args.batch_size,
        class_mode="sparse")

    model = networks.network_factory(
            args.network,
            filters=32,
            dim_output=len(df["name"].unique()))

    opt = keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False)

    model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])
    
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=os.path.join("logs", date),
        histogram_freq=1,
        profile_batch=0)]

    model.fit_generator(
            train_generator,
            steps_per_epoch=None,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=test_generator,
            validation_steps=None,
            validation_freq=1,
            class_weight=None,
            max_queue_size=10,
            workers=6,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0)

    model.summary()

    model.save("./models/" + args.network + "_" + date + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script.")
    parser.add_argument("-e", "--epochs", default="10", type=int, help="Sets number of epochs.")
    parser.add_argument("-l", "--learning-rate", default="0.0001", type=float, help="Sets learning rate.")
    parser.add_argument("-b", "--batch-size", default="4", type=int, help="Sets batch size.")
    parser.add_argument("-n", "--network", default="Inception", type=str, help="Type of network.")
    parser.add_argument("-s", "--split", default="0.8", type=float, help="Portion of dataset used for training.")
    parser.add_argument("-w", "--workaround", action="store_true", help="Turn on workaround for Error \"Cudnn could "
                                                                          "not create handle\" because of low memory. "
                                                                          "Run only if you train the model on low "
                                                                          "spec GPU. Workaround is turned off by "
                                                                          "default, to turn it on set the -w argument")
    parser.add_argument("-d", "--dog-breeds", nargs="*", help="List of dog breeds to train on the neural network. Use the names from column names from annotaions.csv ")
    parsed_args = parser.parse_args()

    # Workaround for could not create cudnn handle because of low memory.
    if parsed_args.workaround:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(parsed_args)

