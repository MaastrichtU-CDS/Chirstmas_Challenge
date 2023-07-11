import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
#import requests
from _requests_ import _requests_
from _model_ import _model_
from _xnatDataQuery import _xnatDataQuery
import time
#from _requests_ import _requests_
from pathlib import Path
from PIL import ImageFile
import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
image_number=91

if __name__ == "__main__":

    # variable_definitions
    url_download ="http://127.0.0.1:5000/file_download"
    url_upload = "http://127.0.0.1:5000/file-upload"
    train_dir =Path(os.path.join(os.getcwd(),'TRAIN'))
    test_dir=Path(os.path.join(os.getcwd(),'TEST'))

    if not os.path.exists('inputWeight'):
        os.mkdir('inputWeight')

    if not os.path.exists('outputWeight'):
        os.mkdir('outputWeight')

    inputWeight = Path(os.path.join(os.getcwd(),'inputWeight'))
    outputWeight =Path(os.path.join(os.getcwd(),'outputWeight'))

    #new object for data request
    # xnatData = _xnatDataQuery()
    requests = _requests_()
    my_model = _model_()

    # xnatData.downloadXNATdata(str(train_dir),str(test_dir))

    text_file = open("input.txt", "r")
    lines = text_file.readlines()
    for ln in lines:
        node_id = int(ln)
    text_file.close()

    # intest_generator,intest_generator_class=validation_data(test_dir,106)
    # alltrain_generator,all_train_generator_class=validation_data(train_dir,192)
    L = list()
    L1 = list()

    train_generator1, test_generator1 = my_model.get_data(train_dir,test_dir,image_number)  #get_data(node_train, node_test)
    csvs = pd.DataFrame()

    for i in range(10):
        iteration = i
        print("iteration",iteration)
        if i ==0:
            model = my_model.get_model()
        else:
            print("Hello")
            #get updated weight file from s-node
            updatedWeightFile_sNode = _requests_._download_file(iteration-1,url_download,inputWeight)
            print(type(updatedWeightFile_sNode))
            print(updatedWeightFile_sNode)
            if os.path.exists(str(updatedWeightFile_sNode)) :
                model = my_model.model_load(updatedWeightFile_sNode)

        history, model_weights, model = my_model.train_model(model, train_generator1, test_generator1)
        filename = os.path.join(Path(outputWeight,str(node_id)+'_'+str(iteration) + 'iteration.h5'))
        model.save(str(filename),include_optimizer=False)
        _requests_.upload_file(filename,iteration,url_upload,node_id)

