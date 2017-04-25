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


K.set_image_dim_ordering('th')
np.random.seed(20170305)


cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/'

os.chdir(cur_path)
train_data_dir = cur_path + 'clothes/train'
validation_data_dir = cur_path + 'clothes/validation'
# dimensions of our images.
img_width, img_height = 350, 350
dense_last2 = 256

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

save_info = '0305_1try_'

save_weights_flag = True

get_model_ori_weights_save_name = save_info+'clothes_uplow_model_ori_weights.h5'
get_model_ori_art_save_name = save_info+'clothes_uplow_model_ori_art.json'

get_v16_bnft_train_save_name = save_info+'clothes_uplow_model_16_bn_ft_train.npy'
get_v16_bnft_valid_save_name = save_info+'clothes_uplow_model_16_bn_ft_valid.npy'

bnft_clf_model_weights_save_name = save_info+'clothes_uplow_bnft_clf_model_weights.h5'
bnft_clf_model_art_save_name = save_info+'clothes_uplow_bnft_clf_model_art.json'

bnft_finetune_model_weights_save_name = save_info+'clothes_uplow_bnft_fine_tune_model.h5'
bnft_finetune_model_art_save_name = save_info+'clothes_uplow_bnft_fine_tune_model_art.json'




def get_model_ori(save_name = get_model_ori_weights_save_name,
                  save_weights_flag = save_weights_flag,
                  batch_size=32,art_name =get_model_ori_art_save_name ):

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2*dense_last2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2*dense_last2 ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', #rmsprop
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            # samplewise_center=True,
            shear_range=0.,
            rotation_range=0.2,
            fill_mode='nearest',
            vertical_flip=True,
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

    early_stopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1)
    save_best = ModelCheckpoint(save_name,
                                verbose=1, monitor='val_acc',
                                save_best_only=True,
                                save_weights_only=save_weights_flag
                                )
    print ('model input_shape:{}'.format(model.input_shape))
    # model.fit_generator(
    #         train_generator,
    #         samples_per_epoch=nb_train_samples,
    #         nb_epoch=nb_epoch + 10,
    #         validation_data=validation_generator,
    #         nb_val_samples=nb_validation_samples,
    #         verbose=1,
    #         callbacks=[save_best,early_stopping])#

    from keras.models import model_from_json
    # save
    if get_model_ori_art_save_name:
        json_str = model.to_json()
        with open(get_model_ori_art_save_name, 'w') as f:
            f.write(json_str)

    return model



def get_v16_bnft(save_train_bn_ft_name = get_v16_bnft_train_save_name,
                            save_valid_bn_ft_name=get_v16_bnft_valid_save_name):

    model = VGG16(weights='imagenet', include_top=False)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # samplewise_center=True,
        shear_range=0.,
        rotation_range=0.2,
        fill_mode='nearest',
        vertical_flip=True,
        horizontal_flip=True,
    )


    test_datagen = ImageDataGenerator(rescale=1. / 255)

    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
    print('begin bottleneck feature train...')
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples )
    np.save(open(save_train_bn_ft_name, 'w'), bottleneck_features_train)

    generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
    print('begin bottleneck feature validation...')
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)

    np.save(open(save_valid_bn_ft_name, 'w'), bottleneck_features_validation)



def get_bnft_clf_model(load_train_bn_ft_name=get_v16_bnft_train_save_name,
                       load_valid_bn_ft_name=get_v16_bnft_valid_save_name,
                       save_name = bnft_clf_model_weights_save_name,
                       save_art_name = bnft_clf_model_art_save_name,
                       save_weights_flag = save_weights_flag
                       ):

    train_data = np.load(open(load_train_bn_ft_name))
    print('train_data shape {}'.format(train_data.shape))
    train_labels = np.array([0] * samples_train_low + [1] * samples_train_up + [2] * samples_train_whole)
    train_labels = np_utils.to_categorical(train_labels,nb_classes)
    print('train_labels shape {}'.format(train_labels.shape))

    validation_data = np.load(open(load_valid_bn_ft_name))
    validation_labels = np.array([0] * samples_valid_low + [1] * samples_valid_up + [2] * samples_valid_whole)
    validation_labels = np_utils.to_categorical(validation_labels,nb_classes)
    print('valid_data shape {}'.format(validation_data.shape))
    print ('nb_classes {}'.format(nb_classes))
    model = Sequential()

    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(dense_last2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_last2, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', #rmsprop
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
    save_best = ModelCheckpoint(save_name,
                                verbose=1, monitor='val_acc',
                                save_best_only=True,
                                save_weights_only=save_weights_flag
                                )

    model.fit(train_data, train_labels,verbose=1,shuffle=True,
              callbacks=[early_stopping,save_best],
              nb_epoch=nb_epoch+30, batch_size=32, #50,32
              validation_data=(validation_data, validation_labels))

    json_str = model.to_json()
    with open(save_art_name, 'w') as f:
        f.write(json_str)

    # print (model.summary())
    # model.save_weights('clothes_uplow_bottleneck_fc_model.h5')
    # print('model output shape:{}'.format(model.output_shape))



def get_bnft_fine_tune_model(load_bnft_clf_name = bnft_clf_model_weights_save_name,
                             save_name = bnft_finetune_model_weights_save_name,
                             save_art_name=bnft_finetune_model_art_save_name,
                             save_weights_flag=save_weights_flag):
    base_model = VGG16(weights='imagenet',include_top=False,input_shape=(3,img_width, img_height))
    print('base model input shape {}'.format(base_model.input_shape))


    print ('base model output {}'.format(base_model.output_shape))

    from keras.models import Model
    import h5py
    weights_bottleneck = h5py.File(cur_path + load_bnft_clf_name)

    weights_dense_256 = [weights_bottleneck['dense_1']['dense_1_W'],weights_bottleneck['dense_1']['dense_1_b']]
    weights_dense_256_2 = [weights_bottleneck['dense_2']['dense_2_W'],weights_bottleneck['dense_2']['dense_2_b']]
    weights_dense_1 = [weights_bottleneck['dense_3']['dense_3_W'],weights_bottleneck['dense_3']['dense_3_b']]

    top_clf = base_model.output
    top_clf = Flatten()(top_clf)
    top_clf = Dense(dense_last2,activation='relu',
                    weights=weights_dense_256)(top_clf)
    top_clf = Dropout(0.5)(top_clf)

    top_clf = Dense(dense_last2, activation='relu',
                    weights=weights_dense_256_2)(top_clf)

    top_clf = Dropout(0.5)(top_clf)

    top_clf = Dense(nb_classes,activation='softmax',weights=weights_dense_1)(top_clf)

    model = Model(input=base_model.input , output=top_clf)

    # print('original layer trainable ')
    # for i,layer in enumerate(model.layers):
    #     print(i , layer.name,layer.trainable)

    for layer in model.layers[:15]:
        layer.trainable = False

    # print('after set, layer trainable ')
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name,layer.trainable)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            # samplewise_center=True,
            shear_range=0.,
            rotation_range=0.2,
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

    early_stopping = EarlyStopping(monitor='val_loss',patience=5,verbose=1)
    save_best = ModelCheckpoint(save_name,
                                verbose=1,monitor='val_loss',
                                save_best_only=True,
                                save_weights_only=save_weights_flag
                                )
    # epoch to csv
    csv_logger = CSVLogger(bnft_finetune_model_weights_save_name[:-2] + 'log',append=True)

    model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch+20,
            verbose=1,
            callbacks=[save_best,early_stopping,csv_logger],
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)

    json_str = model.to_json()
    with open(save_art_name, 'w') as f:
        f.write(json_str)



def save_load_model(model,is_save_only = True,is_load_only = True,
                    weights_name = '',art_name = ''):
    from keras.models import model_from_json
    # save
    if is_save_only:
        if  art_name:
            json_str = model.to_json()
            with open(art_name,'w') as f:
                f.write(json_str)
        if weights_name:
            model.save_weights(weights_name)

    # load
    if is_load_only:
        if art_name:
            model = model_from_json(open(art_name).read())
        if weights_name:
            model.load_weights(weights_name)
        return model


if __name__ == '__main__':
    print ('begin')
    # get_model_ori()
    # get_v16_bnft()
    #
    # get_bnft_clf_model()
    # get_bnft_fine_tune_model()