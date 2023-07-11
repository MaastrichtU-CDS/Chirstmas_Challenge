import sys
import os
import pandas as pd
import numpy as np
from time import time
from keras import applications, optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,PReLU, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
import tensorflow as tf

class _model_():
    def __init__(self):
        self.size = 512
        self.lr = 0.01
        self.momentum = 0
        self.batch_size = 32
        self.epochs = 1

    def model_load(self,updatedWeightfile):
        model = load_model(updatedWeightfile)
        return model

    def get_model(self):
        # model = Sequential()  # Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power).
        # model.add(Conv2D(32, (5, 5), input_shape=(self.size, self.size,1)))  # Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes.
        # model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
        # model.add(PReLU())
        # # model.add(BatchNormalization())
        # model.add(Conv2D(64, (3, 3)))
        # model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
        # model.add(PReLU())
        # # model.add(BatchNormalization())

        # model.add(Conv2D(128, (3, 3)))
        # model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
        # model.add(PReLU())
        # # model.add(BatchNormalization())

        # model.add(Flatten())
        # model.add(Dense(256))
        # model.add(PReLU())
        # model.add(Dense(128))
        # model.add(PReLU())
        # model.add(Dropout(0.50))
        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))
        # model.compile(loss='binary_crossentropy',
        #               # optimizer=optimizers.Adam(),
        #               optimizer=optimizers.SGD(lr=self.lr, momentum=self.momentum),
        #               metrics=['accuracy', 'mse'])
        model=load_model('initial_model.h5')
        return model
    def validation_data(self,intest_dir,batch):
        validation_datagen = ImageDataGenerator()
        val_generator = validation_datagen.flow_from_directory(
            intest_dir,
            target_size=(self.size,self.size),
            batch_size=batch,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )
        return val_generator, val_generator.classes
    def train_model(self, model, train_generator, test_generator):
        # tensorboard = TensorBoard(log_dir=rootdir + outcometype + logtitle, write_graph=False)
        # checkpoints=ModelCheckpoint(bestweights, monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, min_lr=0.001)
        history = model.fit_generator(
            train_generator,
            class_weight={0: 1, 1: 1},
            # steps_per_epoch= 154 // batch_size,
            steps_per_epoch=1,
            epochs=1,
            validation_data=test_generator,
            validation_steps=1,
            verbose=2
        )
        model_weights = model.get_weights()
        # train_acc=history.history['acc']
        # val_loss=history.history['val_loss']
        # val_acc=history.history['val_acc']
        # print(model_weights)
        return history, model_weights, model
    def get_data(self, train_dir, test_dir,image_number):
        train_datagen = ImageDataGenerator()
        validation_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            train_dir,  # this is the target directory
            target_size=(self.size, self.size),
            batch_size=image_number,
            class_mode='binary',
            color_mode='grayscale',
            shuffle= False
        )

        test_generator = validation_datagen.flow_from_directory(
            test_dir,  # this is the target directory
            target_size=(self.size, self.size),
            batch_size=10,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False)

        return train_generator, test_generator




