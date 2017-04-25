from AUNC_DeepLearning.QM_train_model_0401 import Utils
import os
import scipy.io as sio
from keras.models import Sequential,Model,model_from_json
from keras.layers import Convolution2D, MaxPooling2D,Conv1D,Conv3D,MaxPooling1D,MaxPooling3D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,concatenate
from keras.layers import Input,Lambda,ConvLSTM2D
from keras.optimizers import RMSprop
import os
import cv2
import h5py
import datetime
import shutil
from collections import Counter
from keras.applications.vgg16 import VGG16
from keras import optimizers
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import platform
from keras import backend as K
from sklearn.preprocessing import robust_scale,scale
import random
from keras.datasets import mnist


class AUNC():
    def __init__(self,work_path,data_name):
        self.work_path=work_path
        self.data_path=os.path.join(self.work_path,data_name)
        self.utils=Utils()
        self._init_model()
        self._3d_shape=(46,55,46)
        self._4d_shape=(46,55,46,1)
        self._time_steps=10

    def __str__(self):
        print_str = "*****data info*****\n\nclass:{}\nfiles_len:{}\n"
        return print_str.format(self.true_class,self.train_files_len)

    def _init_model(self):
        self.true_class = self.utils.get_files_name(self.data_path)
        self.nb_classes = len(self.true_class)
        self.class_str2num,self.class_num2str=self.utils.build_trueclass_strnum(self.true_class)
        self.train_files_len = list(map(lambda x: self.utils.get_files_len(os.path.join(self.data_path, x)), self.true_class))

        if self.nb_classes == 2:
            self._loss = 'binary_crossentropy'
            self._class_mode = 'binary'
            self._activation = 'sigmoid'
            self._dense_final = 1
        else:
            self._loss = 'categorical_crossentropy'
            self._class_mode = 'categorical'
            self._activation = 'softmax'
            self._dense_final = self.nb_classes

    def _load_aunc_mat(self,path):
        x=sio.loadmat(path)['X'] #(32792,170)
        # x=robust_scale(x,axis=1)
        x = scale(x, axis=1)
        return x.transpose() #(170,32792)

    def _load_aunc_mat_4d(self,path):
        """
        (46,55,46,170) ->(self._time_steps,46,55,46,1)
        :param path:
        :return:
        """
        x = sio.loadmat(path)['data_4d'] #
        x = x[:,:,:,:self._time_steps]
        ##scaler
        x = np.clip(x,-1000,1000)
        x = x/1000.0

        x = np.expand_dims(x,axis=-1) #(46,55,46,timestep,1)
        x = np.transpose(x,axes=[3,0,1,2,4]) #(timestep,46,55,46,1)
        return x

    def get_Xy(self,dim_flag='2d'):
        data = []
        for class_i in self.true_class:
            if dim_flag=='4d':
                tmp=self.utils.get_files_array(self._load_aunc_mat_4d,os.path.join(self.data_path,class_i))
            else:
                tmp = self.utils.get_files_array(self._load_aunc_mat, os.path.join(self.data_path, class_i))
            data += tmp
        data = np.array(data)
        print('data shape:{}'.format(data.shape))

        label = self.utils.get_label_from_file_len(self.train_files_len)

        return data, label

    def _cnn_1d_model(self,input_shape,optimazor):
        d1_cnn = Sequential()
        d1_cnn.add(Conv1D(64,4,input_shape=input_shape))
        d1_cnn.add(BatchNormalization())
        d1_cnn.add(Activation('relu'))
        d1_cnn.add(MaxPooling1D(2))

        d1_cnn.add(Conv1D(64, 4))
        d1_cnn.add(BatchNormalization())
        d1_cnn.add(Activation('relu'))
        d1_cnn.add(MaxPooling1D(2))

        d1_cnn.add(Flatten())
        d1_cnn.add(Dense(1000))
        d1_cnn.add(BatchNormalization())
        d1_cnn.add(Activation('relu'))
        d1_cnn.add(Dense(self._dense_final))
        d1_cnn.add(Activation(self._activation))

        d1_cnn.compile(loss=self._loss,
                      optimizer=optimazor,  # rmsprop
                      metrics=['accuracy'])

        return d1_cnn

    def _cnn_3d_model(self,input_shape,optimazor):
        d3_cnn = Sequential()
        d3_cnn.add(Conv3D(64, 4,4,4,input_shape=input_shape))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2,2,2))) #3d pooling only used in theano

        d3_cnn.add(Conv3D(64, 4, 4, 4))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2,2,2)))

        d3_cnn.add(Flatten())
        d3_cnn.add(Dense(512))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(Dense(self._dense_final))
        d3_cnn.add(Activation(self._activation))

        d3_cnn.compile(loss=self._loss,
                       optimizer=optimazor,  # rmsprop
                       metrics=['accuracy'])
        return d3_cnn

    def _cnn_3d_feature(self,input_shape):
        d3_cnn = Sequential()
        d3_cnn.add(Conv3D(64, 4, 4, 4, input_shape=input_shape))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2, 2, 2)))

        d3_cnn.add(Conv3D(64, 4, 4, 4))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2, 2, 2)))

        d3_cnn.add(Flatten())

        return d3_cnn

    def _get_cnn3d_fea(self):
        """
        input_shape(None,self._time_step,1,46,55,46)
        :return:
        """
        d3_cnn_feature_network=self._cnn_3d_feature(self._4d_shape)
        cnn3d_fea=[]
        input_tensor=[]

        for i in range(self._time_steps):
            tmp_input=Input(self._4d_shape)
            input_tensor.append(tmp_input)

            tmp_fea=d3_cnn_feature_network(tmp_input)
            cnn3d_fea.append(tmp_fea)

        merged_fea=concatenate(cnn3d_fea,axis=-1)

        return input_tensor,merged_fea


    def multi_time_3dcnn(self):
        input_tensor,merged_fea=self._get_cnn3d_fea()

        fea=Dense(512)(merged_fea)
        fea=BatchNormalization()(fea)
        fea=Activation('relu')(fea)
        fea=Dense(self._dense_final)(fea)
        fea=Activation(self._activation)(fea)

        model=Model(inputs=input_tensor,outputs=fea)

        model.compile(loss=self._loss,
                       optimizer='rmsprop',  # rmsprop
                       metrics=['accuracy'])

        return model

    def get_data4d_multi_3nn_Xy(self):
        """
        (None,time_step,46,55,46,1) -> [(:,time_0,:,:,:,:),(:,time_1,:,:,:,:),...]
        :return:
        """
        X,y=self.get_Xy(dim_flag='4d')
        X_data = []
        for i in range(self._time_steps):
            X_data.append(X[:, i, :, :, :, :])

        return X_data,y






    def model_fit(self,model,X_train,y_train,batch_size=16,epoch=20,verbose=1):
        model_history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  verbose=verbose,
                  validation_split=0.1,
                  shuffle=True)
        return model_history






if __name__=='__main__':
    Test = AUNC('/Users/l_mahome/Documents/py3_my_project/AUNC_DeepLearning','data')
    # print(Test)
    # X,y = Test.get_Xy()
    # print(y)
    # print(X[0,:,11])
    # print(X[2, :, 11])
    # print(sum(X[0, :, 11]))
    # # print(sum(X[0, 0, :]))
    # a=X[0,:,11]
    # print(sum(a))

    D3=AUNC('/Users/l_mahome/Documents/py3_my_project/AUNC_DeepLearning','data_4d')
    model=D3.multi_time_3dcnn()
    X,y=D3.get_data4d_multi_3nn_Xy()
    print(np.max(X[0]),np.min(X[0]))

    # D3.model_fit(model,X,y,epoch=2)


