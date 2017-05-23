# -*- coding: utf-8 -*-

#Lmahome

from keras.models import Sequential,Model,model_from_json
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
import os
import cv2
import scipy.io as sio
import h5py
import datetime
import shutil
from collections import Counter
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import platform
from keras import backend as K
from PIL import Image

class Pre_uils():
    """
    initial environment before deep-learning
    """
    def __init__(self):
        self.utils = Utils()

    def delete_dotfile_in_linux(self,path):
        """
        delete '.*.JPG jpg' hide file by recursive in linux 
        :param path: pictures path 
        :return: 
        """
        if platform.system()=='Linux':
            to_cmd = 'find ./ -name ".*.JPG" -print'
            to_cmd2 = 'find ./ -name ".*.jpg" -print'
            print('system has delete (.*.jpg):{}'.format(os.system(to_cmd)))
            print('system has delete (.*.jpg):{}'.format(os.system(to_cmd2)))
            to_cmd = 'find ./ -name ".*.JPG" -delete'
            to_cmd2 = 'find ./ -name ".*.jpg" -delete'
            os.system(to_cmd)
            os.system(to_cmd2)

    def resize_save_pics(self,path,save_path,width=350,height=350):
        """
        resize pictures from path and save it in new path
        train:
            class_1:pic_1,pic_2,...
            class_2:
            ...
        :param path: path to 'train' dir
        :param width:
        :param height:
        :return:
        """
        files = self.utils.get_files_name(path)
        print('find {} dirs'.format(len(files)))
        for class_i in files:
            class_path = os.path.join(path, class_i)
            if not os.path.isdir(class_path):
                raise(IOError,'attention dir struct')

            save_class_path = os.path.join(save_path,class_i)
            if not os.path.exists(save_class_path):
                os.makedirs(save_class_path)

            print('{} have {} pictures to be resized'.format(class_i,len(self.utils.get_files_name(class_path))))
            for pic in self.utils.get_files_name(class_path):
                try:
                    img = cv2.imread(os.path.join(class_path,pic))
                except:
                    raise (IOError, 'load pictures error')

                img = cv2.resize(img, (height, width))

                cv2.imwrite(os.path.join(save_class_path,pic),img)


class Utils():
    """
    main utils deal with directory and pictures
    """
    def __init__(self):
        pass

    def _get_matlab_data(self,path,key):
        return sio.loadmat(path)[key]

    def _get_pic_data(self,path,width,height):
        img = load_img(path, target_size=(height, width))
        img = img_to_array(img)
        return img

    def filter_file_from_dot(self,file_list):
        """
        filter the file startwith dot when use 'listdir' 
        :param file_list: a list have many filenames
        :return: filtered file list
        """
        file_menu = [file_i for file_i in file_list if not file_i.startswith('.')]
        return file_menu

    def random_select_pics_from_path(self,source_path, dest_path,validation_ratio, copy_cut_flag=''):
        """
        random select pics from source_path(have nb_classes dir) .this function used split train/validation/test
        copy(cut) pics to desti_path
        :param source_path: source
        :param dest_path: destination
        :param validation_ratio:the ratio of cut/copy file 
        :param cut_copy_flag: cut or copy ,if '' will do nothing
        :return:
        """
        file_menu = os.listdir(source_path)
        file_menu.sort()
        file_menu = self.filter_file_from_dot(file_menu)

        if isinstance(validation_ratio,float):
            validation_ratio = [validation_ratio]*len(file_menu)
        elif isinstance(validation_ratio,list) and len(validation_ratio)==len(file_menu):
            pass
        else:raise(AttributeError,' validation_ratio should be float or list')

        for i,menu_i in enumerate(file_menu):
            sour_menu_path = os.path.join(source_path,menu_i)
            sour_pics_name = os.listdir(sour_menu_path)
            sour_pics_name.sort()
            sour_pics_name = self.filter_file_from_dot(sour_pics_name)

            sour_pics_len = len(sour_pics_name)
            print('current file name:{}'.format(menu_i))
            print('sour_pics_len : {}'.format(sour_pics_len))
            validation_len = int(sour_pics_len * validation_ratio[i])
            if sour_pics_len>0 and validation_len==0:
                validation_len=1
            print('validation_len: {}'.format(validation_len))
            assert validation_len >= 1
            validation_pics_name = np.random.choice(sour_pics_name,
                                                    validation_len,
                                                    replace=False)

            dest_menu_path = os.path.join(dest_path,menu_i)
            if not os.path.exists(dest_menu_path):
                os.makedirs(dest_menu_path)

            if not copy_cut_flag:pass
            elif copy_cut_flag.lower()=='cut':
                for i_pic_name in validation_pics_name:
                    shutil.move(os.path.join(sour_menu_path, i_pic_name),
                                os.path.join(dest_menu_path,i_pic_name))
            elif copy_cut_flag.lower()=='copy':
                for i_pic_name in validation_pics_name:
                    shutil.copyfile(os.path.join(sour_menu_path, i_pic_name),
                                os.path.join(dest_menu_path,i_pic_name))
            else:raise(BaseException,'invalid copy_cut_flag')


    def get_files_name(self,file_path):
        """
        get files name except dot-file in listdir
        :param file_path: listdir path
        :return: 
        """
        file_menu = os.listdir(file_path)
        file_menu.sort()
        file_menu = self.filter_file_from_dot(file_menu)
        return file_menu


    def get_files_len(self,file_path):
        """
        get files length except dot-file in listdir
        :param file_path: listdir path
        :return: 
        """
        file_menu = os.listdir(file_path)
        file_menu.sort()
        file_menu = self.filter_file_from_dot(file_menu)
        return len(file_menu)

    def get_traintest_files_len(self,train_test_path,true_class):
        files_len = list(map(lambda x: self.get_files_len(os.path.join(train_test_path, x)),true_class))
        return files_len

    def build_trueclass_strnum(self,true_class):
        class_str2num = dict([(_str, _num) for _num, _str in enumerate(true_class)])
        class_num2str = dict([(_num, _str) for _num, _str in enumerate(true_class)])
        return class_str2num,class_num2str

    def get_resized_pics_array(self,pic_path,width,height,no_rescale_minusmean=False):
        """
        resized pictures and append them in a list.
        :param pic_path: pictures path
        :param width:  
        :param height: 
        :param no_rescale_minusmean: if false,will minus mean and rescale
        :return: 
        """
        files = self.get_files_name(pic_path)
        pic_array=[]
        for pic in files:
            try:
                # img = cv2.imread(os.path.join(pic_path, pic))
                img = load_img(os.path.join(pic_path, pic),target_size=(height, width))
                img = img_to_array(img)

            except:
                raise(IOError,'load pictures error')

            #img = img.astype(np.float32, copy=False)
            if not no_rescale_minusmean:
                mean_pixel = [103.939, 116.779, 123.68]
                for c in range(3):
                    img[:, :, c] -= mean_pixel[c]
                img *= (1.0 / 255)

            pic_array.append(img)
        return pic_array

    def get_files_array(self,func,path):
        """

        :param path: point to a files-dir
        :param func: function that load a file(also process file) and return as a numpy array
        :return: files-array in dir
        """
        files = self.get_files_name(path)
        data = []
        for file in files:
            try:
                data_i_array = func(os.path.join(path,file))
            except:
                raise (IOError, 'load files error')
            data.append(data_i_array)
        return data

    def get_resized_pics_generator(self,pic_path,width,height):
        """
        resized pictures and minus mean ,rescale ,expand dims and yield them as generator
        :param pic_path: pictures path
        :param width: 
        :param height: 
        :return: 
        """
        files = self.get_files_name(pic_path)
        for pic in files:
            try:
                img = load_img(os.path.join(pic_path, pic), target_size=(height, width))
                img = img_to_array(img)
            except:
                raise(IOError,'load pictures error')

            mean_pixel = [103.939, 116.779, 123.68]
            for c in range(3):
                img[:, :, c] -= mean_pixel[c]
            img *= (1.0/255)

            img = np.expand_dims(img, axis=0)
            yield img


    def get_pic_augment_generator(self,clf_flag='model_bfp'):
        """
        build picture-augment
        :return: 
        """
        def minus_mean(img):
            img = img.astype(np.float32, copy=False)
            mean_pixel = [103.939, 116.779, 123.68]
            for c in range(3):
                img[:, :, c] -= mean_pixel[c]
            return img

        test_datagen = ImageDataGenerator(rescale=1.0/255, preprocessing_function=minus_mean)  #
        if clf_flag=='model_bfp' or clf_flag=='products_large5':
            train_datagen = ImageDataGenerator(
                rescale=1.0/255,
                samplewise_center=False,
                shear_range=0,
                rotation_range=5,
                width_shift_range=0.1,
                height_shift_range=0.05,
                zoom_range=0,
                fill_mode='nearest',
                vertical_flip=False,
                horizontal_flip=True,
                preprocessing_function = minus_mean,
            )
        elif clf_flag=='clothes_bf':
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                shear_range=0,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.1,
                zoom_range=0.1,
                fill_mode='nearest',
                vertical_flip=False,
                horizontal_flip=False,
                preprocessing_function=minus_mean,
            )
        else:
            raise(BaseException,'clf_info is not correct')


        return train_datagen,test_datagen

    def test_pic_augment(self,clf_flag,one_pic_path,save_path='',produce_num=20):
        """
        produce some pictures to test pictures-augment
        :param one_pic_path: one pictures path
        :param save_path: new pictures output path
        :param produce_num: new pictures number
        :return: 
        """
        img = load_img(one_pic_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        if not save_path:
            save_path = os.path.split(one_pic_path)[0]

        i = 0
        datagen = self.get_pic_augment_generator(clf_flag=clf_flag)[0]
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_path, save_prefix='AUG', save_format='jpeg'):
            i += 1
            if i > 20:
                break


    def get_callback(self,model_name_saveweight,stop_monitor='val_acc',save_monitor='val_acc',patience=8,save_weights_flag=True):
        """
        callback when fit model 
        :param model_name_saveweight: save model weights h5 file
        :param stop_monitor: 
        :param save_monitor: 
        :param patience: 
        :param save_weights_flag: save_weights_only
        :return: 
        """
        early_stopping = EarlyStopping(monitor=stop_monitor, patience=patience, verbose=1)
        save_best = ModelCheckpoint(model_name_saveweight+'.h5',
                                    verbose=1, monitor=save_monitor,
                                    save_best_only=True,
                                    save_weights_only=save_weights_flag
                                    )
        csv_logger = CSVLogger(model_name_saveweight+'.log', append=True)
        return early_stopping,save_best,csv_logger


    def get_model_name(self,model_function):
        '''
        #e.g: get_model_alex -> alex
        :param model_function:function that build model structure and compile it 
        :return:
        '''
        if isinstance(model_function,str):
            _model_name = model_function
            return _model_name
        else:
            _model_name = model_function.__name__
            _model_name = _model_name.split('_')
            if 'get' in _model_name:
                _model_name.remove('get')
            if 'model' in _model_name:
                _model_name.remove('model')
            _model_name = "_".join(_model_name)
            return _model_name


    def get_label_from_file_len(self,file_len):
        """
        build labels according to files length
        :param file_len: 
        :return: 
        """
        assert(isinstance(file_len,list) and len(file_len)>=2)
        _label = []
        for i, i_len in enumerate(file_len):
            _label += [i] * i_len
        _label = np.array(_label)

        if len(file_len)>2:
            _label = np_utils.to_categorical(_label)
        return _label


    def result_write2csv(self,in_order=False):
        #TODO  utils ,in order / correct + error
        pass


    def move_pics_from_result(self,pics_name,source_path,dest_path,is_copy=True):
        """
        move pictures by it's name from source to destination
        :param pics_name: 
        :param source_path: 
        :param dest_path: 
        :param is_copy: 
        :return: 
        """
        for _pic in pics_name:
            if is_copy:
                shutil.copyfile(os.path.join(source_path,_pic),
                                os.path.join(dest_path,_pic))
            else:
                shutil.move(os.path.join(source_path,_pic),
                                os.path.join(dest_path,_pic))

    def get_now_time_str(self):
        """
        get current string format time
        :return: 
        """
        _now = datetime.datetime.now()
        _now = _now.strftime('%m-%d-%H-%M-%S')
        return _now

    def get_model_layer_info_generator(self,model):
        """
        get model layer name 
        :param model: 
        :return: 
        """
        for i,layer in enumerate(model.layers):
            if hasattr(layer, "trainable"):
                yield (i,layer.name,layer.trainable)
            else:
                continue

    def get_weights_h5file_keys(self,h5_path):
        """
        get h5 file keys 
        :param h5_path: 
        :return: 
        """
        weights_bottleneck = h5py.File(h5_path)
        dense_keys = [key for key in list(weights_bottleneck.keys()) if key.startswith('dense')]
        dense_keys.sort()

        dense_weights = []
        for dense_i in dense_keys:
            _tmp_w_b = list(weights_bottleneck[dense_i].keys())
            assert 'W:' in _tmp_w_b[0] and 'b:' in _tmp_w_b[1]
            _tmp_weight = [weights_bottleneck[dense_i][_tmp_w_b[0]],weights_bottleneck[dense_i][_tmp_w_b[1]]]
            dense_weights.append(_tmp_weight)
        return dense_weights


    """for debug"""
    def _test_bug_flow_and_flow_dir(self,QM):
        """
        to find bug:result is different from flow(x,y) and flow_from_dir()
        :return: 
        """
        X_train, y_train = QM.get_X_y_train_valid_data('mypic', is_save=False, no_rescale_minusmean=False)

        def minus_mean(img):
            img = img.astype(np.float32, copy=False)
            mean_pixel = [103.939, 116.779, 123.68]
            for c in range(3):
                img[:, :, c] -= mean_pixel[c]
            return img


        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            preprocessing_function=minus_mean,
        )
        train_generator_2 = train_datagen.flow_from_directory(
            cur_path+'/mypic',
            shuffle=False,  ######for bug
            target_size=(350, 350),
            batch_size=1,
            class_mode='binary')

        print('****')
        a = range(200,350,10)
        print(X_train[0,a,a,:])
        print(y_train)
        print(X_train[1,a,a,:])
        print('\n\n')
        for i,pic in enumerate(train_generator_2):
            print(i)
            print(pic[0].shape)
            print(pic[0][0,a,a,:])
            if i==1:
                break

    def _test_model_evaluate(self,QM,model,train_datagen,train_generator):
        """
        test evaluate_generator from X_train,flow(x,y) and flow_directory
        :param QM: 
        :param model: 
        :param train_datagen: 
        :param train_generator: 
        :return: 
        """
        X_train, y_train = QM.get_X_y_train_valid_data(QM.train,is_save=False,no_rescale_minusmean=False)
        X_train_flow, y_train_flow = QM.get_X_y_train_valid_data(QM.train, is_save=False, no_rescale_minusmean=True)

        print('evaluate:{}'.format(model.evaluate(X_train,y_train)))
        print('evaluate generator array :{}'.format(model.evaluate_generator(train_datagen.flow(X_train_flow, y_train_flow, batch_size=QM.batch_size),
                                                                            val_samples=sum(QM.train_files_len))))
        print('evaluate generator:{}'.format(model.evaluate_generator(train_generator, val_samples=sum(QM.train_files_len))))

        train_generator_2 = train_datagen.flow_from_directory(
            QM.train_path,
            shuffle=False, ######for bug
            target_size=(QM.width, QM.height),
            batch_size=1,
            class_mode=QM._class_mode)
        print('evaluate generator_2:{}'.format(
            model.evaluate_generator(train_generator_2, val_samples=sum(QM.train_files_len))))

        # test for bug
        print('X_train {}'.format(X_train[0,:,:,:]))
        print('y_train {}'.format(y_train[0]))

        for pic_xy in train_datagen.flow(X_train_flow, y_train_flow,batch_size=1,shuffle=False):
            print('flow xy:{}'.format(pic_xy))
            print('shape:{}'.format(len(pic_xy)))
            break

        for pic_xy in train_generator:
            print('flow xy_from_dir:{}'.format(pic_xy))
            print('shape:{}'.format(len(pic_xy)))
            break

    def _test_freeze_layer(self,QM,model):
        """
        to decide freeze i-th layer
        :param QM: 
        :param model: 
        :return: 
        """
        all_layer = self.get_model_layer_info_generator(model)

        if QM.base_model == 'VGG16':
            for i in all_layer:
                print(i)
        elif QM.base_model == 'VGG19':
            for i in all_layer:
                print(i)
        elif QM.base_model == 'InceptionV3':
            for i in all_layer:
                print(i)
        elif QM.base_model == 'ResNet50':
            for i in all_layer:
                print(i)
        elif QM.base_model == 'Xception':
            for i in all_layer:
                print(i)
        else:
            raise (BaseException, 'self base_model is not offer')




class QM_ClfModel():
    """train classify model"""
    def __init__(self,work_path,train,valid,clf_info,save_info,datatry_info,
                 epoch=20,batch_size=64,dense_last2 = 256,
                 width=350,height=350,
                 use_pic_flow_dir = True,use_pic_flow_array = False,use_pic_array=False,
                 random_state=2017):
        """
        
        :param work_path: path to dir having 'train/validation/test'
        :param train: train dir name be used to train model
        :param valid: valid dir name be used to valid model
        :param save_info: save info be used in saving name
        :param datatry_info: data and number of times try 
        :param epoch: 
        :param batch_size: 
        :param dense_last2: dense before last one dense layor
        :param width: pictures width
        :param height:  pictures height
        :param use_pic_flow_dir:  flag to use flow from directory
        :param use_pic_flow_array: flag to use flow(X,y)
        :param use_pic_array: flag to use X_train,y_train
        :param random_state: random seed 
        """

        self.train = train
        self.valid = valid
        self.work_path = work_path
        self.train_path = os.path.join(self.work_path,train)
        self.valid_path = os.path.join(self.work_path,valid)
        os.chdir(self.work_path)


        self.width = width
        self.height = height
        self.optimazor = 'rmsprop'
        self.batch_size = batch_size
        self.dense_last2 = dense_last2
        self.epoch = epoch
        self.droupout = 0.5
        self.freeze = 0
        self.use_batchnorm=False

        self.base_model = 'VGG16'
        self.base_model_list = ['VGG16','VGG19','ResNet50','InceptionV3'] #'Xception' only tf
        self.fine_tune = 'FineTune'

        self.save_info = save_info
        self.datatry_info = datatry_info
        self.save_prefix = os.path.join(self.work_path,self.save_info + '_' + self.datatry_info + '_')
        self.utils = Utils()
        self.clf_info = clf_info

        self.use_pic_flow_dir = use_pic_flow_dir
        self.use_pic_flow_array=use_pic_flow_array
        self.use_pic_array = use_pic_array
        if not sum([self.use_pic_flow_dir,self.use_pic_flow_array,self.use_pic_array])==1:
            raise(BaseException,'use_pic: among of them must have only one True')

        self.train_datagen = []
        self.valid_datagen = []
        self.error_pic_ifno = []
        self.callbacks = None

        self.seed = random_state
        np.random.seed(self.seed)

        self.initial_model()

    def initial_model(self):
        """
        initial model,such as nb_classes,classes_name,files len in train dir
        :return: 
        """

        _train_class = self.utils.get_files_name(self.train_path)
        _valid_class = self.utils.get_files_name(self.valid_path)
        assert len(_valid_class)>=2 and len(_train_class) >= 2 \
               and len(_valid_class)==len(_train_class),'Error:class labels must more than 1.'

        self.nb_classes = len(_train_class)

        self.true_class_str2num = dict([[k,i] for i,k in enumerate(_train_class)])
        self.true_class_num2str = {v:k for k,v in self.true_class_str2num.items()}
        self.true_class = list(self.true_class_str2num.keys())
        self.true_class.sort()

        self.train_files_len = list(map(lambda x:self.utils.get_files_len(os.path.join(self.train_path,x)),self.true_class))
        self.valid_files_len = list(map(lambda x:self.utils.get_files_len(os.path.join(self.valid_path,x)),self.true_class))

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


    def set_base_model_update_freeze_layer(self,base_model='VGG16',train_more_layer=False):
        """
        update freeze layer according it's base model when fine-tune
        :param base_model: 
        :param train_more_layer: 
        :return: 
        """
        self.base_model=base_model
        if train_more_layer:
            if self.base_model == 'VGG16':
                self.freeze = 11
            elif self.base_model == 'InceptionV3':
                self.freeze = 150
            elif self.base_model == 'VGG19':
                self.freeze = 12
            elif self.base_model == 'Xception':
                self.freeze = 106
            elif self.base_model == 'ResNet50':
                self.freeze = 128
        else:
            if self.base_model == 'VGG16':
                self.freeze = 15
            elif self.base_model == 'InceptionV3':
                self.freeze = 172
            elif self.base_model == 'VGG19':
                self.freeze = 17
            elif self.base_model == 'Xception':
                self.freeze = 116
            elif self.base_model == 'ResNet50':
                self.freeze = 148




    def _get_X_y(self,train_valid_test_path,no_rescale_minusmean=False):
        """
        build X,y according path
        :param train_valid_test_path: 
        :param no_rescale_minusmean: 
        :return: 
        """
        data = []

        for class_i in self.true_class:
            tmp = self.utils.get_resized_pics_array(os.path.join(train_valid_test_path,class_i),
                                                    self.width,self.height,no_rescale_minusmean=no_rescale_minusmean)
            data += tmp
        data = np.array(data)
        print('data shape:{}'.format(data.shape))

        files_len_list = list(map(lambda x: self.utils.get_files_len(os.path.join(train_valid_test_path,x)),self.true_class))
        print('files len:{}'.format(files_len_list))
        label = self.utils.get_label_from_file_len(files_len_list)

        return data,label


    def get_X_y_train_valid_data(self,train_valid_test,is_save = True,no_rescale_minusmean=False):
        """
        build X_train,y_train X_valid,y_valid
        :param train_valid_test: 
        :param is_save: 
        :param no_rescale_minusmean: 
        :return: 
        """
        x_save_name = self.save_prefix+ 'X.' + train_valid_test
        y_save_name = self.save_prefix + 'y.' + train_valid_test

        _tmp_path = os.path.join(self.work_path, train_valid_test)
        x_save_name = os.path.join(_tmp_path,x_save_name)
        y_save_name = os.path.join(_tmp_path,y_save_name)

        if os.path.exists(x_save_name) and os.path.exists(y_save_name):
            _X =  np.load(open(x_save_name))
            _y = np.load(open(y_save_name))
        else:
            _X,_y= self._get_X_y(_tmp_path,no_rescale_minusmean=no_rescale_minusmean)
            if is_save:
                print('writing {} data...'.format(train_valid_test))
                np.save(x_save_name, _X)
                np.save(y_save_name, _y)
        return _X,_y

    def get_model_alex(self):
        """
        alex network
        :return: 
        """
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(self.width, self.height,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(self.dense_last2))
        model.add(Activation('relu'))
        if self.use_batchnorm:
            model.add(BatchNormalization())
        else:
            model.add(Dropout(self.droupout))
        model.add(Dense(self.dense_last2))
        model.add(Activation('relu'))
        if self.use_batchnorm:
            model.add(BatchNormalization())
        else:
            model.add(Dropout(self.droupout))
        model.add(Dense(self._dense_final))
        model.add(Activation(self._activation))

        model.compile(loss=self._loss,
                      optimizer=self.optimazor,  # rmsprop
                      metrics=['accuracy'])

        model_name = 'alex'
        return model,model_name


    def model_fit(self,model,model_name,class_weight=None,use_callback=True,save_weights_flag=True,save_structure_flag=True,
                  stop_monitor = 'loss',save_monitor='val_acc',verbose=1):
        """
        fit model and data stream and call back
        :param model: 
        :param model_name: 
        :param class_weight: 
        :param use_callback: 
        :param save_weights_flag: 
        :param save_structure_flag: 
        :param stop_monitor: 
        :param save_monitor: 
        :param verbose: 
        :return: 
        """
        _model_name = model_name

        if use_callback:
            early_stopping,save_best,csvlog = self.utils.get_callback(model_name_saveweight=self.save_prefix+_model_name,
                                                                      save_weights_flag=save_weights_flag,
                                    stop_monitor=stop_monitor,save_monitor=save_monitor)
            self.callbacks=[early_stopping,save_best,csvlog]

        if class_weight:
            if not isinstance(class_weight,dict):
                raise(BaseException,'class weight like:{0:0.3, 1:0.2, 2:0.5}')

        if save_structure_flag:
            json_str = model.to_json()
            with open(os.path.join(self.work_path,self.save_prefix+_model_name + '.json'), 'w') as f:
                f.write(json_str)

        train_datagen, test_datagen = self.utils.get_pic_augment_generator(clf_flag=self.clf_info)
        if self.use_pic_flow_dir:
            print('\n***use_pic_flow_dir***')
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=(self.width, self.height),
                batch_size=self.batch_size,
                class_mode=self._class_mode)
            validation_generator = test_datagen.flow_from_directory(
                self.valid_path,
                target_size=(self.width, self.height),
                batch_size=self.batch_size,
                class_mode=self._class_mode)
            model.fit_generator(
                train_generator,
                samples_per_epoch=sum(self.train_files_len),
                nb_epoch=self.epoch,
                validation_data=validation_generator,
                nb_val_samples=sum(self.valid_files_len),
                class_weight=class_weight,
                verbose=verbose,
                callbacks=self.callbacks)  #
        elif self.use_pic_flow_array:
            print('\n***use_pic_flow_array***')
            X_train, y_train = self.get_X_y_train_valid_data(self.train)
            X_valid, y_valid = self.get_X_y_train_valid_data(self.valid)
            model.fit_generator(train_datagen.flow(X_train, y_train,batch_size=self.batch_size),
                                samples_per_epoch=sum(self.train_files_len),
                                nb_epoch=self.epoch,
                                verbose=verbose,
                                class_weight=class_weight,
                                callbacks=self.callbacks,
                                validation_data=test_datagen.flow(X_valid, y_valid, batch_size=self.batch_size),
                                nb_val_samples=sum(self.valid_files_len))
        elif self.use_pic_array:
            print('\n***use_pic_array***')
            X_train, y_train = self.get_X_y_train_valid_data(self.train)
            X_valid, y_valid = self.get_X_y_train_valid_data(self.valid)
            model.fit(X_train,y_train,
                      batch_size=self.batch_size,
                      nb_epoch=self.epoch,
                      verbose=verbose,
                      class_weight=class_weight,
                      callbacks=self.callbacks,
                      validation_data=[X_valid,y_valid])
        else:
            raise(BaseException)


        return model

    def _get_bnft_feature(self):
        """
        base model bottleneck feature
        :return: 
        """
        model = eval(self.base_model+"(weights='imagenet', include_top=False)")

        train_datagen,test_datagen = self.utils.get_pic_augment_generator(clf_flag=self.clf_info)

        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(self.width, self.height),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)
        validation_generator = test_datagen.flow_from_directory(
            self.valid_path,
            target_size=(self.width, self.height),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)

        model_name = self.utils.get_model_name(self.base_model)
        print('begin bottleneck feature train...')
        bottleneck_features_train = model.predict_generator(train_generator, sum(self.train_files_len))
        np.save(self.save_prefix+model_name+'.'+self.train, bottleneck_features_train)

        print('begin bottleneck feature validation...')
        bottleneck_features_validation = model.predict_generator(validation_generator, sum(self.valid_files_len))
        np.save(self.save_prefix+model_name+'.'+self.valid, bottleneck_features_validation)

    def _get_model_head(self,input_shape):
        """
        build model base on base_model
        :param input_shape: 
        :return: 
        """
        model = Sequential()
        model.add(Flatten(input_shape=input_shape)) # input_shape:train_data.shape[1:])
        model.add(Dense(self.dense_last2, activation='relu'))
        if self.use_batchnorm:
            model.add(BatchNormalization())
        else:
            model.add(Dropout(self.droupout))
        model.add(Dense(self.dense_last2, activation='relu'))
        if self.use_batchnorm:
            model.add(BatchNormalization())
        else:
            model.add(Dropout(self.droupout))
        model.add(Dense(self._dense_final, activation=self._activation))

        model.compile(optimizer=self.optimazor,  # rmsprop
                      loss=self._loss,
                      metrics=['accuracy'])

        return model

    def _get_model_fine_tune(self,_name):
        """
        build base_model and model head
        :param _name: 
        :return: 
        """
        base_model = eval(self.base_model + ("(weights='imagenet', include_top=False, input_shape=(self.width, self.height,3))"))

        head_h5_weights_path = os.path.join(self.work_path, self.save_prefix + _name + '.h5')
        print('load head weights:{}'.format(head_h5_weights_path))
        head_dense_weights = self.utils.get_weights_h5file_keys(head_h5_weights_path)

        top_clf = base_model.output
        top_clf = Flatten()(top_clf)
        top_clf = Dense(self.dense_last2, activation='relu',
                        weights=head_dense_weights[0])(top_clf)
        if self.use_batchnorm:
            top_clf = BatchNormalization()(top_clf)
        else:
            top_clf = Dropout(self.droupout)(top_clf)

        top_clf = Dense(self.dense_last2, activation='relu',
                        weights=head_dense_weights[1])(top_clf)
        if self.use_batchnorm:
            top_clf = BatchNormalization()(top_clf)
        else:
            top_clf = Dropout(self.droupout)(top_clf)
        top_clf = Dense(self._dense_final, activation=self._activation, weights=head_dense_weights[2])(top_clf)
        model = Model(input=base_model.input, output=top_clf)

        for layer in model.layers[:self.freeze]:
            if hasattr(layer,"trainable"):
                layer.trainable = False
            else:continue

        model.compile(loss=self._loss,
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        return model

    def get_model_fine_tune(self,class_weight = None,stop_monitor = 'loss',save_monitor='val_acc',build_bnft_fea=True):
        """
        fine tune a model base on based_model
        :param class_weight: 
        :param stop_monitor: 
        :param save_monitor: 
        :param build_bnft_fea: 
        :return: 
        """
        if build_bnft_fea:
            self._get_bnft_feature()
        model_name = self.utils.get_model_name(self.base_model)
        bottleneck_features_train = np.load(open(self.save_prefix + model_name + '.' + self.train+'.npy', 'rb'))
        bottleneck_features_validation = np.load(open(self.save_prefix + model_name + '.' + self.valid+'.npy', 'rb'))
        train_label = self.utils.get_label_from_file_len(self.train_files_len)
        valid_label = self.utils.get_label_from_file_len(self.valid_files_len)

        model_head = self._get_model_head(input_shape=bottleneck_features_train.shape[1:])
        _model_name = self.base_model+'head'
        early_stopping, save_best, csvlog = self.utils.get_callback(
            model_name_saveweight=self.save_prefix + _model_name,
            save_weights_flag=True,
            stop_monitor=stop_monitor,
            save_monitor=save_monitor)

        model_head.fit(bottleneck_features_train, train_label,
                  verbose=1, shuffle=True,
                  callbacks=[early_stopping, save_best, csvlog],
                  nb_epoch=self.epoch, batch_size=self.batch_size,
                  class_weight=class_weight,
                  validation_data=[bottleneck_features_validation,valid_label])

        model = self._get_model_fine_tune(_name = _model_name)
        model_name = model_name + self.fine_tune
        model.save_weights(os.path.join(self.work_path,self.save_prefix+model_name+'_headbase.h5')) #save head_weight and base_model weight before fine-tune

        return model,model_name
        

    def model_predict(self,trained_model,pic_path,true_class_eval='',
                      model_struct_json='',model_weights_h5='',
                      write2csv_name='',copycut_error_pic='copy',
                      verbose=True):
        """
        classify pictures by trained model
        :param trained_model: 
        :param pic_path: 
        :param true_class_eval: 
        :param model_struct_json: 
        :param model_weights_h5: 
        :param write2csv_name: 
        :param copycut_error_pic: 
        :param verbose: 
        :return: 
        """
        print('\n******\npic_path: {}'.format(pic_path))

        error_dir_out_info = ''
        if trained_model:
            model = trained_model
        elif model_struct_json and model_weights_h5:
            try:
                model = model_from_json(open(model_struct_json).read())
                model.load_weights(model_weights_h5)
                error_dir_out_info = [i for i in model_struct_json.split('_') if '.' in i][0]
                error_dir_out_info = error_dir_out_info[:-5] #no json
            except:
                raise(IOError,'error when load model from json/h5 file')
        else:
            raise(IOError,'No model')

        #TODO: from generator ,from dir,from array
        pic_gene = self.utils.get_resized_pics_generator(pic_path,self.width,self.height)
        pics_len = self.utils.get_files_len(pic_path)
        pics_name = self.utils.get_files_name(pic_path)
        pred_class = []
        pred_prob = []

        if true_class_eval:self.error_pic_ifno=[]

        for i,img in enumerate(pic_gene):
            tmp_pred_prob = model.predict(img)[0]
            tmp_pred_prob = np.array(tmp_pred_prob).tolist()
            if self.nb_classes==2:
                tmp_pred_prob = [1-tmp_pred_prob[0],tmp_pred_prob[0]]
            tmp_pred_prob = list(map(lambda x: round(x, 4), tmp_pred_prob))
            tmp_pred_class = np.argmax(tmp_pred_prob)
            _tmp_res = [self.true_class_num2str[tmp_pred_class]]+tmp_pred_prob

            if true_class_eval:
                if tmp_pred_class != self.true_class_str2num[true_class_eval]:
                    self.error_pic_ifno.append([pics_name[i]] + _tmp_res)

            pred_prob.append(tmp_pred_prob)
            pred_class.append(self.true_class_num2str[tmp_pred_class])

            if verbose:
                print ('{}/{} ===> {}'.format(i, pics_len, _tmp_res))

        if write2csv_name:
            columns = ['pic_name','predict_class']+[k+'_prob' for k in self.true_class]
            print(columns)
            _csv_data = []
            for i in range(pics_len):
                _csv_data.append([pics_name[i],pred_class[i]]+pred_prob[i])

            pd_res = pd.DataFrame(data=_csv_data,columns=columns)
            if not write2csv_name.endswith('.csv'):
                write2csv_name += '.csv'
            write2csv_name = self.save_prefix + write2csv_name
            print('writing to csv...')
            pd_res.to_csv(os.path.join(self.work_path, write2csv_name), header=True)

        if true_class_eval:
            clf_summary = Counter(pred_class)
            # print('error pic:{}'.format(self.error_pic_ifno))
            print('classify result:\t{}'.format(clf_summary))
            acc = round(float(clf_summary[true_class_eval])/pics_len,4)
            print('accuracy:\t{}'.format(acc))

        if copycut_error_pic.lower() in ['copy','cut']:
            if error_dir_out_info:
                dir_name = self.save_prefix + error_dir_out_info +'_err_clf_pics'
            else:
                dir_name = self.save_prefix + error_dir_out_info + '_err_clf_pics'
            error_pic_path = os.path.join(pic_path, dir_name)
            error_pic_path = os.path.join(error_pic_path, true_class_eval)
            try:
                os.makedirs(error_pic_path)
            except OSError:pass
            err_pic_name = (pic_info[0] for pic_info in self.error_pic_ifno)
            is_copy = True if copycut_error_pic.lower()=='copy' else False
            if err_pic_name:
                print('{} error pics to dir...'.format(copycut_error_pic.lower()))
                self.utils.move_pics_from_result(err_pic_name,pic_path,error_pic_path,is_copy=is_copy)



if __name__ == '__main__':
    print ('*******begin******')


    """Back2FrPr"""
    # 1
    # test done
    # cur_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/TestImage/back3frpr_resized'
    # resized_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/DL_Image/back3frpr_resized'
    # pre_utils = Pre_uils()
    # pre_utils.delete_dotfile_in_linux(cur_path)   # test done
    # pre_utils.resize_save_pics(cur_path ,
    #                            resized_path,) # test done


    # 2
    # cur_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/DL_Image/back3frpr_resized'
    # test done
    # UT = Utils()
    # # UT.random_select_pics_from_path(cur_path + '/train', dest_path=cur_path + '/test', copy_cut_flag='cut',
    # #                                 validation_ratio=[0.1,0.2,0.1])
    #
    # UT.test_pic_augment(cur_path+'/train/back/_28701108H4H+28709001B6H (4).JPG',save_path=cur_path)


    # 3
    # QM = QM_ClfModel(cur_path, 'train', 'validation', 'Back2FrPr', '0404try', random_state=20170404)
    # QM.epoch=1

    ## test done
    # ## QM.optimazor = 'adam'
    # QM.batch_size = 32
    # QM.dense_last2 = 512
    # class_weight = {0:0.7,
    #                 1:0.3}
    # # QM.use_pic_flow_dir=False
    # # QM.use_pic_array=True
    # # QM.use_pic_flow_array=False
    #
    # model,model_name = QM.get_model_alex()
    # QM.model_fit(model,model_name=model_name,class_weight=None,save_monitor='acc',stop_monitor='loss',use_callback=True,verbose=1)


    # for base_m in ['VGG19']:
    #     if base_m=='VGG16':
    #         continue
    #     QM.epoch = 80
    #     QM.set_base_model_update_freeze_layer(base_m)
    #     QM.batch_size = 32
    #     QM.dense_last2 = 256
    #     model,model_name = QM.get_model_fine_tune(build_bnft_fea=False,class_weight=None,save_monitor='val_acc',stop_monitor='loss') #cause of this function have a model_fit in it
    #     QM.model_fit(model,model_name = model_name,save_monitor='val_acc',stop_monitor='loss',class_weight=None)





    # test done
    # print(list(QM.true_class))
    # # _tmp_cla = 'frpr'
    # list(map(lambda _tmp_cla:QM.model_predict(trained_model=None,pic_path=cur_path+'/train/'+_tmp_cla,true_class_eval=_tmp_cla,
    #                  model_struct_json= cur_path + '/Back2FrPr_0403try_alex.json',
    #                  model_weights_h5=cur_path + '/Back2FrPr_0403try_alex.h5',write2csv_name=_tmp_cla,
    #                  copymove_error_pic='copy',verbose=False),list(QM.true_class)))

    # _tmp_cla = 'profile'
    # QM.model_predict(trained_model=None,pic_path=cur_path+'/train/'+_tmp_cla,true_class_eval=_tmp_cla,
    #                  model_struct_json= cur_path + '/Back3FrPr_0401try_alex.json',
    #                  model_weights_h5=cur_path + '/Back3FrPr_0401try_alex.h5',write2csv_name=_tmp_cla,
    #                  copymove_error_pic='copy',verbose=True)



    """Products5Large"""
    # 1
    # cur_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/TestImage/products_classify'
    # resized_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/DL_Image/products_classify_resized'
    # pre_utils = Pre_uils()
    # # pre_utils.delete_dotfile_in_linux(cur_path)   # test done
    # pre_utils.resize_save_pics(cur_path ,
    #                            resized_path) # test done

    # 2
    # cur_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/DL_Image/products_classify_resized'

    # test done
    # UT = Utils()
    # # UT.random_select_pics_from_path(cur_path + '/train', dest_path=cur_path + '/test', copy_cut_flag='cut',
    # #                                 validation_ratio=[0.1,0.1,0.3,0.1,0.1,0.1,0.1,0.1])
    #
    # UT.test_pic_augment(cur_path+'/train/back/_28701108H4H+28709001B6H (4).JPG',save_path=cur_path)

    # 3
    # QM = QM_ClfModel(cur_path, 'train', 'validation', 'Products5Large', '0406try', random_state=20170406)
    # # QM.epoch=1

    ## test done
    # ## QM.optimazor = 'adam'
    # QM.batch_size = 32
    # QM.dense_last2 = 512
    # model,model_name = QM.get_model_alex()
    # QM.model_fit(model,model_name=model_name,class_weight=None,save_monitor='acc',stop_monitor='loss',use_callback=True,verbose=1)

    # for base_m in ['VGG19']:
    #     QM.epoch = 80
    #     QM.set_base_model_update_freeze_layer(base_m,freeze_more=True)
    #     QM.batch_size = 32  # vg16 32
    #     if base_m=='VGG16':
    #         QM.dense_last2 = 512
    #     else:
    #         QM.dense_last2 = 256  # vg16 512
    #     model, model_name = QM.get_model_fine_tune(build_bnft_fea=True, class_weight=None, save_monitor='val_loss',
    #                                                stop_monitor='loss')  # cause of this function have a model_fit in it
    #     QM.model_fit(model, model_name=model_name, save_monitor='val_loss', stop_monitor='loss', class_weight=None)


    # 4
    # print(list(QM.true_class))
    # # _tmp_cla = 'frpr'
    # list(map(lambda _tmp_cla:QM.model_predict(trained_model=None,pic_path=cur_path+'/validation/'+_tmp_cla,true_class_eval=_tmp_cla,
    #                  model_struct_json= cur_path + '/Products5Large_0404try_VGG16FineTune.json',
    #                  model_weights_h5=cur_path + '/Products5Large_0404try_VGG16FineTune.h5',write2csv_name=_tmp_cla,
    #                  copycut_error_pic='cut',verbose=False),list(QM.true_class)))

    # _tmp_cla = 'profile'
    # QM.model_predict(trained_model=None,pic_path=cur_path+'/train/'+_tmp_cla,true_class_eval=_tmp_cla,
    #                  model_struct_json= cur_path + '/Back3FrPr_0401try_alex.json',
    #                  model_weights_h5=cur_path + '/Back3FrPr_0401try_alex.h5',write2csv_name=_tmp_cla,
    #                  copymove_error_pic='copy',verbose=True)



    """clothes_bkfr"""
    # 1
    # test done
    width = 700
    height = 700
    # cur_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/TestImage/products_classify/clothes_bkfr'
    # resized_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/DL_Image/clothes_backfront_resized'
    # pre_utils = Pre_uils()
    # pre_utils.delete_dotfile_in_linux(cur_path)   # test done
    # pre_utils.resize_save_pics(cur_path , resized_path , width=width,height=height) # test done

    #TODO: data augment!!!
    # 2
    cur_path = r'/media/qingmu/d6073071-33e5-42a8-9261-f90d5213a49b/DL_Image/clothes_backfront_resized'
    # test done
    # UT = Utils()
    # # UT.random_select_pics_from_path(cur_path + '/train', dest_path=cur_path + '/validation', copy_cut_flag='cut',
    # #                                 validation_ratio=[0.1,0.1])
    #
    # UT.test_pic_augment(clf_flag='clothes_bf',one_pic_path=cur_path+'/train/clothes_front/_31723N-1.jpg',save_path=cur_path)


    # 3
    QM = QM_ClfModel(cur_path, 'train', 'validation','clothes_bf', 'clothes_bkfr', '0409try', random_state=20170409)
    QM_bn = QM_ClfModel(cur_path, 'train', 'validation','clothes_bf', 'clothes_bkfr', '0410bntry', random_state=20170410)
    QM_bn.use_batchnorm=True
    #
    # ## test done
    # # ## QM.optimazor = 'adam'
    # QM.epoch=80
    # QM.batch_size = 32
    # QM.dense_last2 = 512
    # class_weight = {0:0.6,
    #                 1:0.4}
    # # QM.use_pic_flow_dir=False
    # # QM.use_pic_array=True
    # # QM.use_pic_flow_array=False
    #
    # model,model_name = QM.get_model_alex()
    # QM.model_fit(model,model_name=model_name,class_weight=class_weight,save_monitor='acc',stop_monitor='loss',use_callback=True,verbose=1)

    #
    # for base_m in ['VGG16']: #'InceptionV3','Xception','ResNet50'
        # QM.epoch = 80
        # QM.set_base_model_update_freeze_layer(base_m,train_more_layer=True)
        # # QM.optimazor='adam'
        # QM.batch_size = 32  #VGG16: 32 256 0.7,0.3
        # QM.dense_last2 = 256
        # class_weight = {0:0.6,
        #                 1:0.4}
        # model,model_name = QM.get_model_fine_tune(build_bnft_fea=False,class_weight=class_weight,save_monitor='acc',stop_monitor='loss') #cause of this function have a model_fit in it
        # QM.model_fit(model,model_name = model_name,save_monitor='acc',stop_monitor='loss',class_weight=class_weight)

        # QM_bn.epoch = 80
        # QM_bn.set_base_model_update_freeze_layer(base_m)
        # QM_bn.batch_size = 32  # 32
        # QM_bn.dense_last2 = 256  # 256
        # class_weight = {0: 1,
        #                 1: 1}
        # model, model_name = QM_bn.get_model_fine_tune(build_bnft_fea=True, class_weight=class_weight, save_monitor='acc',
        #                                            stop_monitor='loss')  # cause of this function have a model_fit in it
        # QM_bn.model_fit(model, model_name=model_name, save_monitor='acc', stop_monitor='loss', class_weight=class_weight)


    # test done
    # print(list(QM.true_class))
    # # _tmp_cla = 'frpr'
    # list(map(lambda _tmp_cla:QM.model_predict(trained_model=None,pic_path=cur_path+'/validation/'+_tmp_cla,true_class_eval=_tmp_cla,
    #                  model_struct_json= cur_path + '/0407/clothes_bkfr_0407bntry_VGG16head.json',
    #                  model_weights_h5=cur_path + '/0407/clothes_bkfr_0407bntry_VGG16head.h5',write2csv_name=_tmp_cla,
    #                  copycut_error_pic='copy',verbose=False),list(QM.true_class)))

    # print(list(QM_bn.true_class))
    # # _tmp_cla = 'frpr'
    list(map(lambda _tmp_cla: QM_bn.model_predict(trained_model=None, pic_path=cur_path + '/train/' + _tmp_cla,
                                               true_class_eval=_tmp_cla,
                                               model_struct_json=cur_path + '/clothes_bkfr_0409bntry_VGG16FineTune.json',
                                               model_weights_h5=cur_path + '/clothes_bkfr_0409bntry_VGG16FineTune.h5',
                                               write2csv_name=_tmp_cla,
                                               copycut_error_pic='copy', verbose=False), list(QM_bn.true_class)))

    # _tmp_cla = 'profile'
    # QM.model_predict(trained_model=None,pic_path=cur_path+'/train/'+_tmp_cla,true_class_eval=_tmp_cla,
    #                  model_struct_json= cur_path + '/Back3FrPr_0401try_alex.json',
    #                  model_weights_h5=cur_path + '/Back3FrPr_0401try_alex.h5',write2csv_name=_tmp_cla,
    #                  copymove_error_pic='copy',verbose=True)



    K.clear_session()









