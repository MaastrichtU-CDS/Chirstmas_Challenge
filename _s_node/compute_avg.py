import time
import os
import numpy as np
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
import re
import tensorflow
from app import app
from pathlib import Path
size=512
lr=0.01
momentum=0.5

# def Average_weights(weights1,weights2):
# 	collection=[]
# 	collection.append(weights1)
# 	collection.append(weights2)
# 	weights = [model.get_weights() for model in collection]
# 	new_weights = list()

# 	for weights_list_tuple in zip(*weights):
# 		new_weights.append(
# 			[numpy.array(weights_).mean(axis=0)\
# 				for weights_ in zip(*weights_list_tuple)])
# 	return new_weights

def Average_weights(model1,model2,p1,p2):
    weights1=model1.get_weights()
    weights2=model2.get_weights()
    weights1_new=list(p1*(np.array(weights1)))
    weights2_new=list(p2*(np.array(weights2)))
    collection=[]
    collection.append(weights1_new)
    collection.append(weights2_new)
    new_weights = list()
    for weights_list_tuple in zip(*collection):
        new_weights.append(
            [np.array(weights_).sum(axis=0)\
                 for weights_ in zip(*weights_list_tuple)])
    return new_weights

def compute_avg(file1,file2,iteration):
    #Filelist = [file1,file2]
    print("Star Point 1")
    # file11 = Path(os.path.join(app.config['UPLOAD_FOLDER'],'1'+'_'+str(iteration)+'iteration.h5'))
    # print(file11)
    # file22 = Path(os.path.join(app.config['UPLOAD_FOLDER'],'2'+'_'+str(iteration)+'iteration.h5'))
    # print(file22)
    #Filelist2 = [file11,file22]
    print("Start Point 2")

    print("Entered the loop")
    K.clear_session()
    model1 = load_model(str(file1))
    model2 = load_model(str(file2))
    averaged_weights = Average_weights(model1, model2,0.5260416667,0.4739583333)
    K.clear_session()
    print('weights')
    model = Sequential()  # Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power).
    model.add(Conv2D(32, (5, 5), input_shape=(size, size,
                                                  1)))  # Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes.
    model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
    model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
    model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
    model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(PReLU())
    model.add(Dense(128))
    model.add(PReLU())
    model.add(Dropout(0.50))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.set_weights(averaged_weights)
    model.compile(loss='binary_crossentropy',
                     # optimizer=optimizers.Adam(),
                     optimizer=optimizers.SGD(lr=lr, momentum=momentum),
                      metrics=['accuracy', 'mse'])
    print('model_finished')
        #model.save(app.config['DOWNLOAD_FOLDER'] + str(iteration) + 'iteration.h5')  # save the averaged weights to s-node/app/output
    print('wirtingdone')
    filename = os.path.join(Path(app.config['DOWNLOAD_FOLDER'],str(iteration)+'iteration.h5'))
    model.save(str(filename))
    print('wirtingdone')
    K.clear_session()
    return filename
