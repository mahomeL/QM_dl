from AUNC_DeepLearning.QM_train_model_0401 import Utils
import os
import scipy.io as sio
from keras.models import Sequential,Model,model_from_json
from keras.layers import Convolution2D, MaxPooling2D,Conv1D,Conv3D,MaxPooling1D,MaxPooling3D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,concatenate
from keras.layers import Input,Lambda,ConvLSTM2D
from keras.optimizers import RMSprop
import os
# import cv2
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
    def __init__(self,train_path,test_path,time_range_start,time_step=3,batch_size=8,nb_epoch=20,random_state=2017):
        self.train_path=train_path
        self.test_path=test_path

        self.utils=Utils()
        np.random.seed(random_state)

        self._3d_shape=(46,55,46)
        self._4d_shape=(46,55,46,1)
        self.time_steps=time_step #set time_step input for model
        self.time_range_start=time_range_start  #each num time_range_start is a new sample for the 170

        self.batch_size=batch_size
        self.nb_epoch = nb_epoch

        self._init_model()

    def __str__(self):
        print_str = "*****data info*****\n\nclass:{}\nfiles_len:{}\n"
        return print_str.format(self.true_class,self.train_files_len)

    def _init_model(self):
        self.true_class = self.utils.get_files_name(self.train_path)
        self.nb_classes = len(self.true_class)
        self.class_str2num,self.class_num2str=self.utils.build_trueclass_strnum(self.true_class)

        self.train_files_len = self.utils.get_traintest_files_len(self.train_path,self.true_class)
        self.test_files_len = self.utils.get_traintest_files_len(self.test_path,self.true_class)

        self.train_samples_len = [i * len(self.time_range_start) for i in self.train_files_len]
        self.test_samples_len = [i * len(self.time_range_start) for i in self.test_files_len]

        self.train_files_name=list(map(lambda x:self.utils.get_files_name(os.path.join(self.train_path,x)),self.true_class))
        self.test_files_name=list(map(lambda x:self.utils.get_files_name(os.path.join(self.test_path,x)),self.true_class))

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
        (46,55,46,170) ->(self.time_steps,46,55,46,1)
        :param path:
        :return:
        """
        x = sio.loadmat(path)['data_4d']
        x_out=[]
        for start in self.time_range_start:
            tmp_x = x[:,:,:,start:start+self.time_steps]
            ##scaler
            tmp_x=np.clip(tmp_x,-1000,1000)
            tmp_x = tmp_x/1000.0
            tmp_x = np.expand_dims(tmp_x,axis=-1) #(46,55,46,timestep,1)
            tmp_x = np.transpose(tmp_x,axes=[3,0,1,2,4]) #(timestep,46,55,46,1)
            x_out.append(tmp_x)
        x_out = np.array(x_out)
        return x_out

    def get_Xy(self,file_path,dim_flag='2d'):
        """

        :param file_path: train_path or test_path
        :param dim_flag:
        :return:
        """
        data = []
        for class_i in self.true_class:
            if dim_flag=='4d':
                tmp=self.utils.get_files_array(self._load_aunc_mat_4d,os.path.join(file_path,class_i))
            else:
                tmp = self.utils.get_files_array(self._load_aunc_mat, os.path.join(file_path, class_i))
            data += tmp
        data = np.array(data)

        #because time_step feature will be a sample so should reshape
        files_len=self.utils.get_traintest_files_len(file_path,self.true_class) #
        data = data.reshape((sum(files_len)*len(self.time_range_start),self.time_steps,46,55,46,1))#
        print('data shape:{}'.format(data.shape))
        new_len = [i*len(self.time_range_start) for i in files_len] #

        label = self.utils.get_label_from_file_len(new_len)

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
        d3_cnn.add(Conv3D(64, (4,4,4),input_shape=input_shape))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2,2,2))) #3d pooling only used in theano

        d3_cnn.add(Conv3D(64, (4, 4, 4)))
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
        d3_cnn.add(Conv3D(64, (4, 4, 4), input_shape=input_shape))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2, 2, 2)))

        d3_cnn.add(Conv3D(64, (4, 4, 4)))
        d3_cnn.add(BatchNormalization())
        d3_cnn.add(Activation('relu'))
        d3_cnn.add(MaxPooling3D((2, 2, 2)))

        d3_cnn.add(Conv3D(128, (8, 8, 8)))
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

        for i in range(self.time_steps):
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
        for i in range(self.time_steps):
            X_data.append(X[:, i, :, :, :, :])

        return X_data,y


    def model_fit(self,model,X_train,y_train,batch_size=8,epoch=20,verbose=1):
        model_history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  verbose=verbose,
                  validation_split=0.2,
                  shuffle=True)
        print(model.summary())
        return model_history

    def _batch_generator(self,train_test_flag='train'):
        if train_test_flag=='train':
            while 1:
                _tmp_idx = np.random.choice(np.arange(sum(self.train_samples_len)),size=self.batch_size,replace=False)

                x_out = []
                y_out = []
                for idx in _tmp_idx:
                    # print(idx)
                    if idx<self.train_samples_len[0]:
                        _class='AU'
                    else:
                        _class='NC'
                        idx = idx-self.train_samples_len[0]

                    path = os.path.join(self.train_path,_class)

                    _sub=idx//len(self.time_range_start)
                    _time_start=idx%len(self.time_range_start)
                    # print(_sub,_time_start)

                    path=os.path.join(path,self.train_files_name[self.class_str2num[_class]][_sub])
                    x = sio.loadmat(path)['data_4d']
                    # print(path)

                    tmp_x = x[:, :, :, self.time_range_start[_time_start]:self.time_range_start[_time_start] + self.time_steps]
                    ##scaler
                    tmp_x = np.clip(tmp_x, -1000, 1000)
                    tmp_x = tmp_x / 1000.0
                    tmp_x = np.expand_dims(tmp_x, axis=-1)  # (46,55,46,timestep,1)
                    tmp_x = np.transpose(tmp_x, axes=[3, 0, 1, 2, 4])  # (timestep,46,55,46,1)
                    x_out.append(tmp_x) #(batcnsize,timestep,46,55,46,1)
                    y_out.append(self.class_str2num[_class] )

                x_out = np.array(x_out)
                # x_out: [[batchsize,46,55,46,1],[batchsize,46,55,46,1],...,]total self.time_steps
                tmp = []
                for t in range(self.time_steps):
                    tmp.append(x_out[:,t, :, :, :, :])

                yield (tmp,np.array(y_out))
        elif train_test_flag=='test':
            while 1:
                _tmp_idx = np.random.choice(np.arange(sum(self.test_samples_len)), size=self.batch_size, replace=False)

                x_out = []
                y_out = []
                for idx in _tmp_idx:
                    if idx < self.test_samples_len[0]:
                        _class = 'AU'
                    else:
                        _class = 'NC'
                        idx = idx - self.train_samples_len[0]

                    path = os.path.join(self.test_path, _class)

                    _sub = idx // len(self.time_range_start)
                    _time_start = idx % len(self.time_range_start)

                    path = os.path.join(path, self.test_files_name[self.class_str2num[_class]][_sub])
                    x = sio.loadmat(path)['data_4d']

                    tmp_x = x[:, :, :,self.time_range_start[_time_start]:self.time_range_start[_time_start] + self.time_steps]
                    ##scaler
                    tmp_x = np.clip(tmp_x, -1000, 1000)
                    tmp_x = tmp_x / 1000.0
                    tmp_x = np.expand_dims(tmp_x, axis=-1)  # (46,55,46,timestep,1)
                    tmp_x = np.transpose(tmp_x, axes=[3, 0, 1, 2, 4])  # (timestep,46,55,46,1)

                    x_out.append(tmp_x)
                    y_out.append(self.class_str2num[_class])

                x_out = np.array(x_out)
                # x_out: [[batchsize,46,55,46,1],[batchsize,46,55,46,1],...,]total self.time_steps
                tmp = []
                for t in range(self.time_steps):
                    tmp.append(x_out[:, t, :, :, :, :])

                yield (tmp, np.array(y_out))
        else:
            raise(BaseException)

    def model_fit_generator(self,model):
        model.fit_generator(generator=self._batch_generator('train'),
                            steps_per_epoch=sum(self.train_samples_len)//self.batch_size,
                            epochs=self.nb_epoch,
                            validation_data=self._batch_generator('test'),
                            validation_steps=sum(self.test_samples_len)//self.batch_size,
                            )

        # for e in range(self.nb_epoch):
        #     print('Epoch {}'.format(e))
        #     batches=0
        #     test_gene = self._batch_generator('test')
        #     for X_batch,y_batch in self._batch_generator('train'):
        #         _loss_tr = model.train_on_batch(X_batch,y_batch)
        #         for test_xy in test_gene:
        #             _loss_te = model.test_on_batch(test_xy[0],test_xy[1])
        #             break
        #         print('train : {},{}'.format(_loss_tr[0],_loss_tr[1]))
        #         print('test : {},{}'.format(_loss_te[0], _loss_te[1]))
        #         batches+=1
        #         if batches>=sum(self.train_samples_len)//self.batch_size:
        #             break


    def _write_Xy_h5file(self,X,y,filename=None):
        if filename is None:
            filename='AUNC4d_TimeStep{}.h5'.format(self.time_steps)
        xyfile=h5py.File(filename,'w')
        xyfile.create_dataset('X',data=X)
        xyfile.create_dataset('y',data=y)

    def _read_Xy_h5file(self,filename):
        xyfile = h5py.File(filename,'r')
        X=xyfile['X']
        y=xyfile['y']
        return X,y


if __name__=='__main__':
    # Test = AUNC('/Users/l_mahome/Documents/py3_my_project/AUNC_DeepLearning','data')
    # print(Test)
    # X,y = Test.get_Xy()
    # print(y)
    # print(X[0,:,11])
    # print(X[2, :, 11])
    # print(sum(X[0, :, 11]))
    # # print(sum(X[0, 0, :]))
    # a=X[0,:,11]
    # print(sum(a))
    time_step = 5
    time_range_start=range(0,120,time_step)
    D3=AUNC('/media/s/Data/LIAOLuo/AUNC_DeepLearning/AUNC_oridata_mat_3d_46_55_46/train',
            '/media/s/Data/LIAOLuo/AUNC_DeepLearning/AUNC_oridata_mat_3d_46_55_46/train',
            time_range_start=list(time_range_start),
            time_step=time_step,
            batch_size=8
            )
    print(D3)
    print(' total samples:{} iter per opoch :{}'.format(sum(D3.train_samples_len),sum(D3.train_samples_len)//D3.batch_size))
    model=D3.multi_time_3dcnn()
    # X,y=D3.get_data4d_multi_3nn_Xy()
    # #
    # # X_data = []
    # # for i in range(10):
    # #     X_data.append(X[i,:, :, :, :, :])
    # D3.model_fit(model,X,y,epoch=6)

    D3.model_fit_generator(model=model)



