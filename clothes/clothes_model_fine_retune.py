# -*- coding: utf-8 -*-

#LL

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
import numpy as np
from keras import backend as K
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json

K.set_image_dim_ordering('th')


def get_files_len_from_path(path=r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes/train',
                            verbose = 0):
    # l u w
    file_len =[]

    file_menu = os.listdir(path)
    if '.DS_Store' in file_menu:
        file_menu.remove('.DS_Store')
    # print (file_menu)
    # 'lower_body''upper_body''whole_body'
    assert (file_menu[0]=='lower_body' and file_menu[1]=='upper_body'
            and file_menu[2] == 'whole_body'),'should have folder [lower_body,upper_body,whole_body'
    for menu_i in file_menu:
        sour_pics_name = os.listdir(path + r'/'+menu_i)
        if '.DS_Store' in sour_pics_name:
            sour_pics_name.remove('.DS_Store')
        sour_pics_len = len(sour_pics_name)
        file_len.append(sour_pics_len)
    if verbose:
        print('files_len:{}, total_sum:{}'.format(file_len,sum(file_len)))

    return file_len


def clothes_model_fine_retune(cur_path,load_info,save_info,
                              weig_path,art_path,
                              train_data_dir,validation_data_dir,
                              nb_train_samples,nb_validation_samples,
                              img_width=350,img_height=350,
                              save_weights_flag=True,nb_epoch=20):
    load_model_weights = cur_path + load_info+weig_path
    load_model_art= cur_path + load_info+art_path
    weihts_save_name = cur_path + save_info + weig_path

    model = model_from_json(open(load_model_art).read())
    model.load_weights(load_model_weights)

    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            # samplewise_center=True,
            shear_range=0.,
            rotation_range=0.1,
            fill_mode='nearest',
            vertical_flip=True,
            horizontal_flip=True
            )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical')

    early_moni = 'loss'
    save_best_moni = 'val_loss'
    early_patience = 4
    load_model_weights
    load_model_art= cur_path + load_info+art_path
    weihts_save_name = cur_path + save_info + weig_path
    print('************model info************\n')
    print('load weights path :{}\nload art path :{}\nsave weights path :{}'.format(load_model_weights,
                                                                                   load_model_art,
                                                                                   weihts_save_name))
    print('early stopping monitor :{}\nsave best monitor :{}'.format(early_moni,save_best_moni))
    print('early patience: {}'.format(early_patience))
    early_stopping = EarlyStopping(monitor=early_moni,patience=early_patience,verbose=1)
    save_best = ModelCheckpoint(weihts_save_name,
                                verbose=1,monitor=save_best_moni,
                                save_best_only=True,
                                save_weights_only=save_weights_flag
                                )
    # epoch to csv
    csv_logger = CSVLogger(weihts_save_name[:-2] + 'log',append=True)

    for layer in model.layers[:15]:
        layer.trainable = False

    print('after set, layer trainable ')
    for i, layer in enumerate(model.layers):
        print(i, layer.name,layer.trainable)

    model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=[save_best,early_stopping,csv_logger],
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)
    return model


def main():
    print ('begin')
    np.random.seed(20170308)

    cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/'

    os.chdir(cur_path)
    train_data_dir = cur_path + 'clothes/train'
    validation_data_dir = cur_path + 'clothes/validation'
    # dimensions of our images.
    img_width, img_height = 350, 350
    dense_last2 = 256

    train_file_len = get_files_len_from_path(path=train_data_dir)
    valid_file_len = get_files_len_from_path(path=validation_data_dir)
    samples_train_low = train_file_len[0]
    samples_train_up = train_file_len[1]
    samples_train_whole = train_file_len[2]
    nb_train_samples = sum(train_file_len)
    samples_valid_low = valid_file_len[0]
    samples_valid_up = valid_file_len[1]
    samples_valid_whole = valid_file_len[2]
    nb_validation_samples = sum(valid_file_len)

    nb_epoch = 20
    LL_IsDebug = False
    nb_classes = 3
    save_weights_flag = True

    load_info = '0305_1try_'

    save_info = '0308_1try_'

    weig_path = 'clothes_uplow_bnft_fine_tune_model.h5'
    art_path  = 'clothes_uplow_bnft_fine_tune_model_art.json'

    model = clothes_model_fine_retune(cur_path,load_info,save_info,weig_path,art_path,
                                      train_data_dir,validation_data_dir,
                                      nb_train_samples,nb_validation_samples)

if __name__ == '__main__':
    main()
