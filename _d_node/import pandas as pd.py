import pandas as pd
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
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
from numpy.random import seed
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import roc_auc_score,classification_report
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
import sys
import yaml
import matplotlib.pyplot as plt
import os
import numpy
outcometype='DM'
logtitle="2ndrun_2019"
size=512
# rootdir=config['Setting']['rootdir']
# imagedir='/home/ubuntu/IMAGES_DM_CENTER/'

# bestweights=imagedir+ outcometype + logtitle + ".h5"
lr=0.01
momentum=0.5  # momentum=0
batch_size=32 #shoule be the number of training images on each nodes.
epochs=1
node1_train='/Users/zhangchong/Downloads/Maastro/cds/2nodes/node1/TRAIN'
node1_test='/Users/zhangchong/Downloads/Maastro/cds/2nodes/node1/TEST'
node2_train='/Users/zhangchong/Downloads/Maastro/cds/2nodes/node2/TRAIN'
node2_test='/Users/zhangchong/Downloads/Maastro/cds/2nodes/node2/TEST'

initial_model=load_model('/Users/zhangchong/Downloads/Maastro/cds/models/initial_model.h5')
initial_weights=initial_model.get_weights()

class Histories(Callback):

	def __init__(self, validation_generator = None, train_generator = None):
		super(Histories, self).__init__()
		self.validation_generator = validation_generator
		self.train_generator = train_generator

	def on_train_begin(self, logs={}):
		self.aucs = []
		self.trainingaucs = []
		self.losses = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		valid_steps = np.ceil(self.validation_generator.samples/self.validation_generator.batch_size)
		true_classes = self.validation_generator.classes
		predictions = self.model.predict_generator(generator = self.validation_generator, steps = valid_steps,workers=1)
		roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))

		self.aucs.append(round(roc_auc,3))
		print('Validation AUCS')
		print(self.aucs)

		valid_steps = np.ceil(self.train_generator.samples/self.train_generator.batch_size)
		true_classes = self.train_generator.classes
		predictions = self.model.predict_generator(generator = self.train_generator, steps = valid_steps,workers=1)
		roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))

		self.trainingaucs.append(round(roc_auc,3))
		print('Training AUCS')
		print(self.trainingaucs)

		return self.trainingaucs, self.aucs
def get_model(local_lr):
					
	model = Sequential() #Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power). 

	model.add(Conv2D(32, (5, 5), input_shape=(size, size, 1)),) #Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes.
	model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
	model.add(PReLU()) 
	#model.add(BatchNormalization())

	model.add(Conv2D(64, (3, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
	model.add(PReLU())
	#model.add(BatchNormalization())


	model.add(Conv2D(128, (3, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
	model.add(PReLU())
	#model.add(BatchNormalization())

	model.add(Flatten()) 
	model.add(Dense(256))
	model.add(PReLU())
	model.add(Dense(128))
	model.add(PReLU())
	model.add(Dropout(0.50))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	try:
		model.set_weights(averaged_weights)
		print('read averaged model')
	except:
		print('initial weights')
		model.set_weights(initial_weights)
		pass
	model.compile(loss='binary_crossentropy',
			#optimizer=optimizers.Adam(),
			optimizer=optimizers.Adam(lr=local_lr),
			metrics=['accuracy','mse'])
	return model

def get_data(train_dir,test_dir,image_number):
	train_datagen = ImageDataGenerator()
	validation_datagen = ImageDataGenerator()
	
	train_generator = train_datagen.flow_from_directory(
		train_dir,  # this is the target directory
		target_size=(size, size),  
		batch_size=image_number,
		class_mode='binary',
		color_mode = 'grayscale'
		)  

	test_generator = validation_datagen.flow_from_directory(
		test_dir,  # this is the target directory
		target_size=(size, size),  
		batch_size=10,
		class_mode='binary',
		color_mode = 'grayscale',
		shuffle = False)  


	return train_generator, test_generator

def validation_data(intest_dir,batch):
	validation_datagen = ImageDataGenerator()
	val_generator=validation_datagen.flow_from_directory(
		intest_dir,
		target_size=(size, size),  
		batch_size=batch,
		class_mode='binary',
		color_mode = 'grayscale',
		shuffle = False
	)
	return val_generator, val_generator.classes

def train_model(model,train_generator, test_generator, ratio, samples):
	# tensorboard = TensorBoard(log_dir=rootdir + outcometype + logtitle, write_graph=False)
	# checkpoints=ModelCheckpoint(bestweights, monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1,min_lr=0.001)

	history = model.fit_generator(
		train_generator,
		class_weight = {0 : 1, 1: 1},
		# steps_per_epoch= 154 // batch_size,
		steps_per_epoch=samples//batch_size,
		epochs=1,
		validation_data=test_generator,
		validation_steps= 1,
		verbose = 2
		)
	model_weights=model.get_weights()

	# train_acc=history.history['acc']
	val_loss=history.history['val_loss']
	val_acc=history.history['val_acc']
	# print(model_weights)
	return history, model_weights

def Average_weights(weights1,weights2):
	collection=[]
	collection.append(weights1)
	collection.append(weights2)
	weights = collection
	new_weights = list()

	for weights_list_tuple in zip(*weights):
		new_weights.append(
			[numpy.array(weights_).mean(axis=0)\
				for weights_ in zip(*weights_list_tuple)])
	return new_weights


def History_to_csv(history):
	hist_df=pd.DataFrame(history.history)
	return hist_df

def Save_history(node_csv,node):
	with open('/home/ubuntu/localrun/'+node+'testhistory.csv', mode='w') as f:
		node_csv.to_csv(f)
	
def AUC_cal():
	pass

def tuning_learning_rate(history, best_val_loss, patience, local_lr, node_name, lr_reducing_factor=0.5, max_patience=20):
	val_losses = history.history['val_loss']
	
	min_lr = 0.0001
	
	for tmp_val_loss in val_losses:
		if tmp_val_loss < best_val_loss:
			best_val_loss=tmp_val_loss
			patience=0
			print(node_name + ' ######Best validation loss updated to ' + str(best_val_loss) + '######')
		else:
			patience +=1
			print(node_name + ' ######Validation loss doesn\'t change, ' + 'val_loss: ' + str(tmp_val_loss) + ', patience: ' + str(patience) + ' ######')
	
	if not patience < max_patience:
		tmp_lr = local_lr
		local_lr=local_lr * lr_reducing_factor
		if local_lr < min_lr:
			local_lr = tmp_lr
		patience = 0
		print(node_name + ' ######Local learning rate reduced to ' + str(local_lr) + ' ######')
	else:
		local_lr=local_lr
	
	return(local_lr, patience, best_val_loss)
	

if __name__ == "__main__":
# 	intest_generator,intest_generator_class=validation_data(test_dir,106)
# 	alltrain_generator,all_train_generator_class=validation_data(train_dir,192)
	L=list()
	L1=list()
	# Loss1=list()
	# Loss2=list()
	# train_aucs=list()
	# test_aucs=list()
	# tuning learning rate
	local_lr_1 = lr
	local_lr_2 = lr
	local_lr_3 = lr

	patience_1 = 0
	patience_2 = 0
	patience_3 = 0
	
	lowest_val_loss_1 = 1
	lowest_val_loss_2 = 1
	lowest_val_loss_3 = 1

	Saving_Flag = True

	lr_reducing_factor = 0.5
	train_generator_1,test_generator1=get_data(node1_train,node1_test,101) # loop外
	train_generator_2,test_generator2=get_data(node2_train,node2_test,91)
	csvs1=pd.DataFrame()
	csvs2=pd.DataFrame()
	for i in range(10):
		print(i)
		model1=get_model(local_lr_1)
		model2=get_model(local_lr_2)
		history1,model_weights1=train_model(model1,train_generator_1,test_generator1,1,101)
		history2,model_weights2=train_model(model2,train_generator_2,test_generator2,1,91)
  
  
		## evaluate val_loss to reduce learning rate
		# local node 1
		# local_lr_1, patience_1, lowest_val_loss_1 = tuning_learning_rate(history1, lowest_val_loss_1, 
		# 																 patience_1, local_lr_1,'node1')
		
		# local_lr_2, patience_2, lowest_val_loss_2 = tuning_learning_rate(history2, lowest_val_loss_2, 
		# 																 patience_2, local_lr_2,'node2')
		
		## -----------------------------------------
		# history3,model_weights3=train_model(model,train_generator_3,test_generator3)
		averaged_weights=Average_weights(model_weights1,model_weights2)
		csv1=History_to_csv(history1)
		csv2=History_to_csv(history2)
		csvs1=pd.concat([csvs1,csv1])
		csvs2=pd.concat([csvs2,csv2])
# 		model = Sequential() #Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power). 

# 		model.add(Conv2D(32, (5, 5), input_shape=(size, size, 1))) #Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes.
# 		model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
# 		model.add(PReLU()) 
# 		#model.add(BatchNormalization())

# 		model.add(Conv2D(64, (3, 3)))
# 		model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
# 		model.add(PReLU())
# 		#model.add(BatchNormalization())


# 		model.add(Conv2D(128, (3, 3)))
# 		model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
# 		model.add(PReLU())
# 		#model.add(BatchNormalization())

# 		model.add(Flatten()) 
# 		model.add(Dense(256))
# 		model.add(PReLU())
# 		model.add(Dense(128))
# 		model.add(PReLU())
# 		model.add(Dropout(0.50))
# 		model.add(Dense(1))
# 		model.add(Activation('sigmoid'))
# 		model.set_weights(averaged_weights)
# 		model.compile(loss='binary_crossentropy',
# 			#optimizer=optimizers.Adam(),
# 			optimizer=optimizers.SGD(lr=lr,momentum=momentum),
# 			metrics=['accuracy','mse'])
# 		model.save('/home/ubuntu/localrun/' + str(i)+'localrun_final_averaged.h5')
  
# 		evaluation=model.evaluate_generator(generator=intest_generator,steps=1)
# 		evaluation_train=model.evaluate_generator(generator=alltrain_generator,steps=1)
  
  
# 		L.append(evaluation)
# 		L1.append(evaluation_train)

		# pred_train=model.predict_generator(generator=alltrain_generator,steps=1)
		# pred_test=model.predict_generator(generator=intest_generator,steps=1)
		# auc_train=roc_auc_score(y_true = all_train_generator_class, y_score = np.ravel(pred_train))
		# auc_test=roc_auc_score(y_true = intest_generator_class, y_score = np.ravel(pred_test))
		# train_aucs.append(auc_train)
# 		# test_aucs.append(auc_test)
# 	Save_history(csvs1,'node1')
# 	Save_history(csvs2,'node2')
# 	evaluation_df=pd.DataFrame(list(L))
# 	evaluation_df_train=pd.DataFrame(list(L1))
# 	# aucs_train_df=pd.DataFrame(train_aucs)
# 	# aucs_test_df=pd.DataFrame(test_aucs)
# 	evaluation_df.to_csv('/home/ubuntu/localrun/evaluation.csv')
# 	evaluation_df_train.to_csv('/home/ubuntu/localrun/evaluation_train.csv')
# 	# aucs_train_df.to_csv('/home/ubuntu/localrun/aucs_train_df.csv')
# 	# aucs_train_df.to_csv('/home/ubuntu/localrun/aucs_test_df.csv')
