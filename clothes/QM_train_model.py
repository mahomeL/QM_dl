# -*- coding: utf-8 -*-

#LL

from keras.models import Sequential,Model,model_from_json
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
import os
import cv2
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

class Utils():
    def __init__(self):
        pass

    def delete_file_in_linux(self,path):
        if platform.system()=='Linux':
            to_cmd = 'find ./ -name ".*.JPG" -print'
            to_cmd2 = 'find ./ -name ".*.jpg" -print'
            print('system has delete (.*.jpg):{}'.format(os.system(to_cmd)))
            print('system has delete (.*.jpg):{}'.format(os.system(to_cmd2)))
            to_cmd = 'find ./ -name ".*.JPG" -delete'
            to_cmd2 = 'find ./ -name ".*.jpg" -delete'
            os.system(to_cmd)
            os.system(to_cmd2)

    def filter_file_from_dot(self,file_list):
        file_menu = [file_i for file_i in file_list if not file_i.startswith('.')]
        return file_menu

    def random_select_pics_from_path(self,source_path, dest_path,validation_ratio, copy_cut_flag=''):
        """
        random select pics from source_path(have nb_classes dir)
        copy(cut) pics to desti_path
        :param source_path:
        :param dest_path:
        :param validation_ratio:
        :param cut_copy_flag:
        :return:
        """
        file_menu = os.listdir(source_path)
        file_menu = self.filter_file_from_dot(file_menu)

        if isinstance(validation_ratio,float):
            validation_ratio = [validation_ratio]*len(file_menu)
        elif isinstance(validation_ratio,list) and len(validation_ratio)==len(file_menu):
            pass
        else:raise(AttributeError,' validation_ratio should be float or list')

        for i,menu_i in enumerate(file_menu):
            sour_menu_path = os.path.join(source_path,menu_i)
            sour_pics_name = os.listdir(sour_menu_path)
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
                os.mkdir(dest_menu_path)

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
        file_menu = os.listdir(file_path)
        file_menu = self.filter_file_from_dot(file_menu)
        return file_menu


    def get_files_len(self,file_path):
        file_menu = os.listdir(file_path)
        file_menu = self.filter_file_from_dot(file_menu)
        return len(file_menu)


    def get_resized_pics_array(self,pic_path,width,height):
        files = self.get_files_name(pic_path)
        pic_array=[]
        for pic in files:
            try:
                img = cv2.imread(os.path.join(pic_path, pic))
            except:
                raise(IOError,'load pictures error')

            img = img.astype(np.float32, copy=False)
            mean_pixel = [103.939, 116.779, 123.68]
            for c in range(3):
                img[:, :, c] -= mean_pixel[c]

            img = cv2.resize(img, (height, width))
            # img = img.transpose((2, 0, 1))
            # img = np.expand_dims(img, axis=0)
            pic_array.append(img)
        return pic_array


    def get_resized_pics_generator(self,pic_path,width,height):
        files = self.get_files_name(pic_path)
        for pic in files:
            try:
                img = cv2.imread(os.path.join(pic_path, pic))
            except:
                raise(IOError,'load pictures error')

            img = img.astype(np.float32, copy=False)
            mean_pixel = [103.939, 116.779, 123.68]
            for c in range(3):
                img[:, :, c] -= mean_pixel[c]

            img = cv2.resize(img, (height, width))
            # img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            yield img


    def get_pic_augment_generator(self):

        def minus_mean(img):
            img = img.astype(np.float32, copy=False)
            mean_pixel = [103.939, 116.779, 123.68]
            for c in range(3):
                img[:, :, c] -= mean_pixel[c]
            return img

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            samplewise_center=False,
            shear_range=20,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.1,
            fill_mode='nearest',
            vertical_flip=False,
            horizontal_flip=True,
            preprocessing_function = minus_mean,
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255,preprocessing_function=minus_mean)
        return train_datagen,test_datagen

    def get_callback(self,model_name_saveweight,stop_monitor='val_acc',save_monitor='val_acc',patience=8,save_weights_flag=True):
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
        #get_model_alex -> alex
        :param model_function:
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
        #TODO utils
        for _pic in pics_name:
            if is_copy:
                shutil.copyfile(os.path.join(source_path,_pic),
                                os.path.join(dest_path,_pic))
            else:
                shutil.move(os.path.join(source_path,_pic),
                                os.path.join(dest_path,_pic))

    def get_now_time_str(self):
        _now = datetime.datetime.now()
        _now = _now.strftime('%m-%d-%H-%M-%S')
        return _now



class QM_ClfModel():
    """train classify model"""
    def __init__(self,work_path,train,valid,save_info,datatry_info,
                 epoch=20,batch_size=64,dense_last2 = 256,
                 width=350,height=350,
                 use_pic_flow_dir = True,use_pic_flow_array = False,use_pic_array=False,
                 random_state=2017):

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

        self.base_model = 'VGG16'
        self.base_model_list = ['VGG16','VGG19','ResNet50','InceptionV3'] #'Xception' only tf
        self.fine_tune = 'FineTune'

        self.save_info = save_info
        self.datatry_info = datatry_info
        self.save_prefix = os.path.join(self.work_path,self.save_info + '_' + self.datatry_info + '_')
        self.utils = Utils()

        self.use_pic_flow_dir = use_pic_flow_dir
        self.use_pic_flow_array=use_pic_array
        self.use_pic_array = use_pic_array
        if not sum([self.use_pic_flow_dir,self.use_pic_flow_array,self.use_pic_array])==1:
            raise(BaseException,'use_pic: among of them must have only one True')

        self.train_datagen = []
        self.valid_datagen = []
        self.error_pic_ifno = []
        self.callbacks = None

        self.seed = random_state
        np.random.seed(self.seed)




    def initial_model(self):

        _train_class = self.utils.get_files_name(self.train_path)
        _valid_class = self.utils.get_files_name(self.valid_path)
        assert len(_valid_class)>=2 and len(_train_class) >= 2 \
               and len(_valid_class)==len(_train_class),'Error:class labels must more than 1.'

        self.nb_classes = len(_train_class)
        self.true_class_str2num = dict([[k,i] for i,k in enumerate(_train_class)])
        self.true_class_num2str = {v:k for k,v in self.true_class_str2num.items()}
        self.true_class = self.true_class_str2num.keys()

        self.train_files_len = map(lambda x:len(self.utils.get_files_name(os.path.join(self.train_path,x))),_train_class)
        self.valid_files_len = map(lambda x:len(self.utils.get_files_name(os.path.join(self.valid_path,x))),_valid_class)

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

        if self.base_model=='VGG16':
            self.freeze=15
        elif self.base_model=='InceptionV3':
            self.base_model=172



    def _get_X_y(self,train_valid_test_path):
        data = []
        for class_i in self.true_class:
            tmp = self.utils.get_resized_pics_array(os.path.join(train_valid_test_path,class_i),
                                                    self.width,self.height)
            data += tmp
        data = np.array(data)


        files_len_list = map(lambda x: len(self.utils.get_files_name(os.path.join(train_valid_test_path,x))),self.true_class)
        # print('files len:{}'.format(files_len_list))
        label = self.utils.get_label_from_file_len(files_len_list)

        return data,label


    def get_X_y_train_valid_data(self,train_valid_test,is_save = True):
        assert(train_valid_test in ['train','validation','test'])
        x_save_name = self.save_prefix+ 'X.' + train_valid_test
        y_save_name = self.save_prefix + 'y.' + train_valid_test

        _tmp_path = os.path.join(self.work_path, train_valid_test)
        x_save_name = os.path.join(_tmp_path,x_save_name)
        y_save_name = os.path.join(_tmp_path,y_save_name)

        if os.path.exists(x_save_name) and os.path.exists(y_save_name):
            _X =  np.load(open(x_save_name))
            _y = np.load(open(y_save_name))
        else:
            _X,_y= self._get_X_y(_tmp_path)
            if is_save:
                print('writing {} data...'.format(train_valid_test))
                np.save(open(x_save_name, 'w'), _X)
                np.save(open(y_save_name, 'w'), _y)
        return _X,_y

    def get_model_alex(self):
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
        model.add(Dropout(self.droupout))
        model.add(Dense(self.dense_last2))
        model.add(Activation('relu'))
        model.add(Dropout(self.droupout))
        model.add(Dense(self._dense_final))
        model.add(Activation(self._activation))


        model.compile(loss=self._loss,
                      optimizer=self.optimazor,  # rmsprop
                      metrics=['accuracy'])

        model_name = 'alex'
        return model,model_name


    def model_fit(self,model,model_name,class_weight=None,use_callback=True,save_weights_flag=True,save_structure_flag=True,
                  stop_monitor = 'loss',save_monitor='val_acc'):

        _model_name = model_name

        if use_callback:
            early_stopping,save_best,csvlog = self.utils.get_callback(model_name_saveweight=self.save_prefix+_model_name,
                                                                      save_weights_flag=save_weights_flag,
                                    stop_monitor=stop_monitor,save_monitor=save_monitor)
            self.callbacks=[early_stopping,save_best,csvlog]

        if class_weight:
            if not isinstance(class_weight,dict):
                raise(BaseException,'class weight like:{0:0.3, 1:0.2, 2:0.5}')

        train_datagen, test_datagen = self.utils.get_pic_augment_generator()
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
                verbose=1,
                callbacks=self.callbacks)  #
        elif self.use_pic_flow_array:
            print('\n***use_pic_flow_array***')
            X_train, y_train = self.get_X_y_train_valid_data(self.train)
            X_valid, y_valid = self.get_X_y_train_valid_data(self.valid)
            model.fit_generator(train_datagen.flow(X_train, y_train,batch_size=self.batch_size),
                                samples_per_epoch=sum(self.train_files_len),
                                nb_epoch=self.epoch,
                                verbose=1,
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
                      verbose=1,
                      class_weight=class_weight,
                      callbacks=self.callbacks,
                      validation_data=[X_valid,y_valid])
        else:
            raise(BaseException)

        if save_structure_flag:
            json_str = model.to_json()
            with open(os.path.join(self.work_path,self.save_prefix+_model_name + '.json'), 'w') as f:
                f.write(json_str)

        return model

    def _get_bnft_feature(self):
        model = eval(self.base_model+"(weights='imagenet', include_top=False)")

        train_datagen,test_datagen = self.utils.get_pic_augment_generator()

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
        np.save(open(self.save_prefix+model_name+'.'+self.train, 'w'), bottleneck_features_train)

        print('begin bottleneck feature validation...')
        bottleneck_features_validation = model.predict_generator(validation_generator, sum(self.valid_files_len))
        np.save(open(self.save_prefix+model_name+'.'+self.valid, 'w'), bottleneck_features_validation)

    def _get_model_head(self,input_shape):

        model = Sequential()
        model.add(Flatten(input_shape=input_shape)) # input_shape:train_data.shape[1:])
        model.add(Dense(self.dense_last2, activation='relu'))
        model.add(Dropout(self.droupout))
        model.add(Dense(self.dense_last2, activation='relu'))
        model.add(Dropout(self.droupout))

        model.add(Dense(self._dense_final, activation=self._activation))

        model.compile(optimizer=self.optimazor,  # rmsprop
                      loss=self._loss,
                      metrics=['accuracy'])

        return model

    def _get_model_fine_tune(self,_name):
        base_model = eval(self.base_model + ("(weights='imagenet', include_top=False, input_shape=(self.width, self.height,3))"))

        weights_bottleneck = h5py.File(os.path.join(self.work_path,self.save_prefix + _name +'.h5'))

        weights_dense_256 = [weights_bottleneck['dense_1']['dense_1_W:0'], weights_bottleneck['dense_1']['dense_1_b:0']]
        weights_dense_256_2 = [weights_bottleneck['dense_2']['dense_2_W:0'], weights_bottleneck['dense_2']['dense_2_b:0']]
        weights_dense_1 = [weights_bottleneck['dense_3']['dense_3_W:0'], weights_bottleneck['dense_3']['dense_3_b:0']]

        top_clf = base_model.output
        top_clf = Flatten()(top_clf)
        top_clf = Dense(self.dense_last2, activation='relu',
                        weights=weights_dense_256)(top_clf)
        top_clf = Dropout(self.droupout)(top_clf)

        top_clf = Dense(self.dense_last2, activation='relu',
                        weights=weights_dense_256_2)(top_clf)
        top_clf = Dropout(self.droupout)(top_clf)

        top_clf = Dense(self._dense_final, activation=self._activation, weights=weights_dense_1)(top_clf)
        model = Model(input=base_model.input, output=top_clf)

        for layer in model.layers[:self.freeze]:  # train last two block
            layer.trainable = False

        model.compile(loss=self._loss,
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        return model

    def get_model_fine_tune(self,class_weight = None,stop_monitor = 'loss',save_monitor='val_acc'):
        self._get_bnft_feature()

        model_name = self.utils.get_model_name(self.base_model)
        bottleneck_features_train = np.load(open(self.save_prefix + model_name + '.' + self.train, 'r'))
        bottleneck_features_validation = np.load(open(self.save_prefix + model_name + '.' + self.valid, 'r'))
        train_label = self.utils.get_label_from_file_len(self.train_files_len)
        valid_label = self.utils.get_label_from_file_len(self.valid_files_len)

        model_head = self._get_model_head(input_shape=bottleneck_features_train.shape[1:])
        _model_name = 'head'
        early_stopping, save_best, csvlog = self.utils.get_callback(
            model_name_saveweight=self.save_prefix + _model_name,
            save_weights_flag=True,
            stop_monitor=stop_monitor,
            save_monitor=save_monitor)

        model_head.fit(bottleneck_features_train, train_label,
                  verbose=1, shuffle=True,
                  callbacks=[early_stopping, save_best, csvlog],
                  nb_epoch=self.epoch*2, batch_size=self.batch_size,
                  class_weight=class_weight,
                  validation_data=[bottleneck_features_validation,valid_label])

        model = self._get_model_fine_tune(_name = _model_name)
        model_name = model_name + self.fine_tune
        return model,model_name
        

    def model_predict(self,trained_model,pic_path,true_class_eval='',
                      model_struct_json='',model_weights_h5='',
                      write2csv_name='',copymove_error_pic='copy',
                      verbose=True):
        if trained_model:
            model = trained_model
        elif model_struct_json and model_weights_h5:
            try:
                model = model_from_json(open(model_struct_json).read())
                model.load_weights(model_weights_h5)
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
            tmp_pred_prob = map(lambda x: round(x, 4), tmp_pred_prob)
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
            columns = ['pic_name','predict_class']+[k+'_prob' for k in self.true_class_str2num.keys()]
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
            print('classify result:{}'.format(clf_summary))
            acc = round(float(clf_summary[true_class_eval])/pics_len,4)
            print('accuracy:{}'.format(acc))

        if copymove_error_pic.lower() in ['copy','move']:
            dir_name = self.save_prefix + 'err_clf_pics'

            error_pic_path = os.path.join(pic_path, dir_name)
            error_pic_path = os.path.join(error_pic_path, true_class_eval)
            try:
                os.makedirs(error_pic_path)
            except OSError:pass
            err_pic_name = (pic_info[0] for pic_info in self.error_pic_ifno)
            is_copy = True if copymove_error_pic.lower()=='copy' else False
            print('{} error pics to dir'.format(copymove_error_pic.lower()))
            self.utils.move_pics_from_result(err_pic_name,pic_path,error_pic_path,is_copy=is_copy)




if __name__ == '__main__':
    print ('*******begin******')
    cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes'

    # test done
    UT = Utils()
    UT.random_select_pics_from_path(cur_path + '/train', dest_path=cur_path + '/my_test_pic', copy_cut_flag='cut',
                                    validation_ratio=[0.1, 0.5])

    QM = QM_ClfModel(cur_path,'train','validation','backfrpr','0002try')
    QM.epoch=2

    # test done
    # QM.use_pic_flow_array = False
    # QM.use_pic_flow_dir = False
    # QM.use_pic_array = True

    QM.initial_model()
    QM.base_model='InceptionV3'


    # test done
    # model,model_name = QM.get_model_alex()
    # QM.model_fit(model,model_name=model_name)

    # test done
    # model,model_name = QM.get_model_fine_tune()
    # QM.model_fit(model,model_name = model_name)

    # test done
    # QM.model_predict(trained_model=None,pic_path=cur_path+'/train/frpr',true_class_eval='frpr',
    #                  model_struct_json= cur_path + '/backfrpr_0002try_VGG16FineTune.json',
    #                  model_weights_h5=cur_path + '/backfrpr_0002try_VGG16FineTune.h5',write2csv_name='test_csv',
    #                  copymove_error_pic='copy',verbose=True)









