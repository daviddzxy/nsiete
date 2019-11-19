import tensorflow.keras as keras
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import sklearn.preprocessing

import networks
import dataload


# only for testing puproses 
# tbd
df = pd.read_csv('../data/annotations.csv')
df = df.loc[(df["name"] == 'Chihuahua') | (df["name"] == 'African_hunting_dog')]


le = sklearn.preprocessing.LabelEncoder()
labels = le.fit_transform(df["name"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
ids = df["id"].astype(int).to_numpy()

dg = dataload.DataLoader(ids, labels, 32)
model = networks.InceptionNet(32, dim_output=2)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.fit_generator(dg, steps_per_epoch=None, epochs=30, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=6, use_multiprocessing=False, shuffle=True, initial_epoch=0)
model.summray()
