# -*- coding: utf-8 -*-

#LL

from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
import os
import cv2
import h5py
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
K.set_image_dim_ordering('th')



class Utils():
    def __init__(self):
        pass

    def get_files_name(self,file_path):
        file_menu = os.listdir(file_path)
        while '.DS_Store' in file_menu:
            file_menu.remove('.DS_Store')
        return file_menu

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
            img = img.transpose((2, 0, 1))
            # img = np.expand_dims(img, axis=0)
            pic_array.append(img)
        return pic_array

    def get_pic_generator(self):

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
        csv_logger = CSVLogger(model_name_saveweight+',log', append=True)
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



class Train_Clf_Model():
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


        self.width = width
        self.height = height
        self.optimazor = 'rmsprop'
        self.batch_size = batch_size
        self.dense_last2 = dense_last2
        self.epoch = epoch
        self.droupout = 0.5
        self.freeze = 15

        self.base_model = 'VGG16'
        self.base_model_list = ['VGG16','VGG19','ResNet50','InceptionV3'] #'Xception' only tf

        self.save_info = save_info
        self.datatry_info = datatry_info
        self.save_prefix = os.path.join(self.work_path,self.save_info + '_' + self.datatry_info + '_')
        self.utils = Utils()

        self.use_pic_flow_dir = use_pic_flow_dir
        self.use_pic_flow_array=use_pic_array
        self.use_pic_array = use_pic_array
        assert(sum([self.use_pic_flow_dir,self.use_pic_flow_array,self.use_pic_array])==1,
               'use_pic: among of them must have only one True')

        self.train_datagen = []
        self.valid_datagen = []
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




    def _get_X_y(self,train_valid_test_path):
        data = []
        for class_i in self.true_class:
            tmp = self.utils.get_resized_pics_array(os.path.join(train_valid_test_path,class_i),
                                                    self.width,self.height)
            data += tmp
        self.X_train = np.array(data)

        label = []
        for i, files_len in enumerate(train_valid_test_path):
            label += [i] * files_len
        label = np.array(label)
        if self.nb_classes > 2:
            label = np_utils.to_categorical(label, nb_classes=self.nb_classes)

        return data,label


    def load_X_y_train_valid(self,train_valid_test,is_save = True):
        assert(train_valid_test in ['train','validation','test'])
        x_save_name = self.save_prefix+ 'X.' + train_valid_test
        y_save_name = self.save_prefix + 'y.' + train_valid_test

        _tmp_path = os.path.join(self.work_path, train_valid_test)
        x_save_name = os.path.join(_tmp_path,x_save_name)
        y_save_name = os.path.join(_tmp_path,y_save_name)

        if os.path.exists(x_save_name) and os.path.exists(y_save_name):
            self.X_train = np.load(open(x_save_name))
            self.y_train = np.load(open(y_save_name))
        else:
            self.X_train, self.y_train = self._get_X_y(train_valid_test)
            if is_save:
                np.save(open(x_save_name, 'w'), self.X_train)
                np.save(open(y_save_name, 'w'), self.y_train)


    def get_model_alex(self):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(3, self.width, self.height)))
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


        return model


    def model_fit(self,model_function,class_weight=None,use_callback=True,save_weights_flag=True,save_structure_flag=True):

        model = model_function()
        _model_name = self.utils.get_model_name(model_function=model_function)

        if use_callback:
            early_stopping,save_best,csvlog = self.utils.get_callback(model_name_saveweight=self.save_prefix+_model_name,
                                                                      save_weights_flag=save_weights_flag,
                                    stop_monitor='loss',save_monitor='val_acc')
            self.callbacks=[early_stopping,save_best,csvlog]


        if class_weight:
            assert(isinstance(class_weight,dict),'class weight like:{0:0.3, 1:0.2, 2:0.5}')

        train_datagen, test_datagen = self.utils.get_pic_generator()
        if self.use_pic_flow_dir:
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
        elif self.use_pic_flow_dir:
            X_train = np.load(open(self.save_prefix+'X.'+self.train))
            y_train = np.load(open(self.save_prefix+'y.'+self.train))
            X_valid = np.load(open(self.save_prefix+'X.'+self.valid))
            y_valid = np.load(open(self.save_prefix+'y.'+self.valid))

            model.fit_generator(train_datagen.flow(X_train, y_train,batch_size=self.batch_size),
                                samples_per_epoch=sum(self.train_files_len),
                                nb_epoch=self.epoch,
                                verbose=1,
                                class_weight=class_weight,
                                callbacks=self.callbacks,
                                validation_data=test_datagen.flow(X_valid, y_valid, batch_size=self.batch_size),
                                nb_val_samples=sum(self.valid_files_len))
        elif self.use_pic_array:
            X_train = np.load(open(self.save_prefix + 'X.' + self.train))
            y_train = np.load(open(self.save_prefix + 'y.' + self.train))
            X_valid = np.load(open(self.save_prefix + 'X.' + self.valid))
            y_valid = np.load(open(self.save_prefix + 'y.' + self.valid))
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
            with open(self.save_prefix+_model_name + '.json', 'w') as f:
                f.write(json_str)

        return model

    def _get_bnft_feature(self):
        model = eval(self.base_model+"(weights='imagenet', include_top=False)")

        train_datagen,test_datagen = self.utils.get_pic_generator()

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

    def get_model_head(self,input_shape):

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
        model.save_weights(self.save_prefix + 'head.h5py')
        return model

    def get_model_fine_tune(self,):
        base_model = eval(self.base_model + ("(weights='imagenet', include_top=False, input_shape=(3, self.width, self.height)"))
        weights_bottleneck = h5py.File(self.save_prefix + 'head.h5py')

        weights_dense_256 = [weights_bottleneck['dense_1']['dense_1_W'], weights_bottleneck['dense_1']['dense_1_b']]
        weights_dense_256_2 = [weights_bottleneck['dense_2']['dense_2_W'], weights_bottleneck['dense_2']['dense_2_b']]
        weights_dense_1 = [weights_bottleneck['dense_3']['dense_3_W'], weights_bottleneck['dense_3']['dense_3_b']]

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



if __name__ == '__main__':
    print ('*******begin******')
    cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes'
    clf_model = Train_Clf_Model(cur_path,'train','validation','backfrpr','0001try')
    clf_model.initial_model()
    clf_model.model_fit(clf_model.get_model_alex,)






