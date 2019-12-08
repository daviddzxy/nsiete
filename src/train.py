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
    if re.match(".*/src$", os.getcwd()) or re.match(".*/notebooks$", os.getcwd()):
        os.chdir("../")  #  change directory to root directory

    date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    df_train = pd.read_csv("./data/train.csv")
    df_valid = pd.read_csv("./data/valid.csv")

    # select dog breeds to train on
    if args.dog_breeds is not None:
        df_train = df_train[df_train["name"].isin(args.dog_breeds)]
        df_valid = df_valid[df_valid["name"].isin(args.dog_breeds)]

    df_train["id"] = df_train["id"].apply(lambda x: str(x) + ".png")
    df_valid["id"] = df_valid["id"].apply(lambda x: str(x) + ".png")

    if args.augmentation:
        train_datagen = ImageDataGenerator(rescale=1./255,
                horizontal_flip=True,
                rotation_range=30,
                zoom_range=0.15,
                shear_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                fill_mode="nearest")
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory="./data/processed",
        x_col="id",  # image filename
        y_col="name",
        batch_size=args.batch_size,
        target_size=(299, 299),
        class_mode="sparse")

    valid_generator = test_datagen.flow_from_dataframe(
        dataframe=df_valid,
        directory="./data/processed",
        x_col="id",
        y_col="name",
        batch_size=args.batch_size,
        target_size=(299, 299),
        class_mode="sparse")

    model = networks.network_factory(
            args.network,
            filters=32,
            dim_output=len(df_train["name"].unique()))

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
        keras.callbacks.TensorBoard(log_dir=os.path.join("logs",  date + '_' + args.network),
        histogram_freq=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1, restore_best_weights=True)]


    model.fit_generator(
            train_generator,
            steps_per_epoch=None,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_generator,
            validation_steps=None,
            validation_freq=1,
            class_weight=None,
            max_queue_size=10,
            workers=6,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0)

    model.summary()

    model.save_weights("./model_weights/" + date + "_" + args.network + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", default="10", type=int, help="Sets number of epochs.")
    parser.add_argument("-l", "--learning-rate", default="0.0001", type=float, help="Sets learning rate.")
    parser.add_argument("-b", "--batch-size", default="32", type=int, help="Sets batch size.")
    parser.add_argument("-n", "--network", default="Inception", type=str, choices=["Inception", "InceptionV3", "BaseConv", "InceptionResNet"], help="Type of network.")
    parser.add_argument("-w", "--workaround", action="store_true", help="Turn on workaround for Error \"Cudnn could "
                                                                          "not create handle\" because of low memory. "
                                                                          "Run only if you train the model on low "
                                                                          "spec GPU. Workaround is turned off by "
                                                                          "default, to turn it on set the -w argument")
    parser.add_argument("-d", "--dog-breeds", nargs="*", help="List of dog breeds to train on the neural network. Use the names from column names from annotaions.csv. If not specified train on all breeds. ")
    parser.add_argument("-a", "--augmentation", action="store_true", help="Allow augmentation.")
    parsed_args = parser.parse_args()

    # Workaround for could not create cudnn handle because of low memory.
    if parsed_args.workaround:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(parsed_args)

