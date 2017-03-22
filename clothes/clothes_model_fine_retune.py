# -*- coding: utf-8 -*-

#LL


from keras import backend as K
import os
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

K.set_image_dim_ordering('th')
from clothes_bfp import load_train_valid_pic,_load_resized_pics


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


def clothes_model_fine_retune(cur_path,load_wei_info,load_ari_info,save_info,
                              weig_path,art_path,
                              train_data_dir,validation_data_dir,
                              nb_train_samples,nb_validation_samples,
                              img_width=350,img_height=350,
                              save_weights_flag=True,nb_epoch=20,
                              batch_size=128,use_dir_generator = True,
                              freeze_num=15):

    load_model_weights = cur_path + load_wei_info+weig_path
    # load_model_art= cur_path + load_info+art_path
    load_model_art = cur_path + load_ari_info + art_path
    weihts_save_name = cur_path + save_info + weig_path

    model = model_from_json(open(load_model_art).read())
    model.load_weights(load_model_weights)

    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         # samplewise_center=True,
    #         shear_range=0.,
    #         rotation_range=0.1,
    #         fill_mode='nearest',
    #         vertical_flip=True,
    #         horizontal_flip=True
    #         )
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # samplewise_center=True,
        shear_range=0,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        vertical_flip=False,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

    early_moni = 'acc'
    save_best_moni = 'val_acc'
    early_patience = 8

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

    for layer in model.layers[:freeze_num]:
        layer.trainable = False

    # print('after set, layer trainable ')
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name,layer.trainable)

    if use_dir_generator:
        model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch+20,
                verbose=1,
                callbacks=[save_best,early_stopping,csv_logger],
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples)
    else:
        X_train, y_train = load_train_valid_pic(train_data_dir)
        X_valid, y_valid = load_train_valid_pic(validation_data_dir)
        print('X_train shape\t:{}'.format(X_train.shape))
        print('X_valid shape\t:{}'.format(X_valid.shape))
        model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                            samples_per_epoch=nb_train_samples,
                            nb_epoch=nb_epoch + 20,
                            verbose=1,
                            callbacks=[save_best, early_stopping],
                            validation_data=test_datagen.flow(X_valid, y_valid, batch_size=batch_size),
                            nb_val_samples=nb_validation_samples)
    return model



if __name__ == '__main__':
    cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/'

    ###need be change by user
    save_info = '0321_2try_'
    wei_info = '0321_2try'
    ari_info = '0321_2try'
    classes_ab = 'bfp'
    np.random.seed(20170322)
    train_data_dir = cur_path + 'clothes/train'
    validation_data_dir = cur_path + 'clothes/validation'
    ######

    weig_path = 'clothes_' + classes_ab + '_bnft_fine_tune_model.h5'
    art_path = 'clothes_' + classes_ab + '_bnft_fine_tune_model_art.json'
    os.chdir(cur_path)
    train_file_len = get_files_len_from_path(path=train_data_dir)
    valid_file_len = get_files_len_from_path(path=validation_data_dir)
    nb_train_samples = sum(train_file_len)
    nb_validation_samples = sum(valid_file_len)

    #######
    clothes_model_fine_retune(cur_path=cur_path,load_wei_info=wei_info,load_ari_info=ari_info,
                              save_info=save_info, weig_path=weig_path,art_path=art_path,
                              train_data_dir=train_data_dir,
                              validation_data_dir=validation_data_dir,
                              nb_train_samples=nb_train_samples,
                              nb_validation_samples=nb_validation_samples,
                              batch_size=128,use_dir_generator=False,freeze_num=11)
