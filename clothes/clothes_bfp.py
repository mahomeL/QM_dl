# -*- coding: utf-8 -*-

#LL

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
import numpy as np
from keras import backend as K
import os
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


K.set_image_dim_ordering('th')
np.random.seed(20170320)

def get_files_len_from_path(pic_classes,path=r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/clothes/train',
                            verbose = 0):
    # l u w
    file_len =[]

    file_menu = os.listdir(path)
    if '.DS_Store' in file_menu:
        file_menu.remove('.DS_Store')
    # print (file_menu)
    # 'lower_body''upper_body''whole_body'
    assert (file_menu[0]==pic_classes[0] and file_menu[1]==pic_classes[1]
            and file_menu[2] == pic_classes[2]),'should have folder {}'.format(pic_classes)
    for menu_i in file_menu:
        sour_pics_name = os.listdir(path + r'/'+menu_i)
        if '.DS_Store' in sour_pics_name:
            sour_pics_name.remove('.DS_Store')
        sour_pics_len = len(sour_pics_name)
        file_len.append(sour_pics_len)
    if verbose:
        print('files_len:{}, total_sum:{}'.format(file_len,sum(file_len)))

    return file_len


def _load_resized_pics(pic_path):
    assert os.path.isdir(pic_path)
    files = os.listdir(pic_path)
    while '.DS_Store' in files:
        files.remove('.DS_Store')
    input_len = len(files)
    print ('input files len: {}'.format(input_len))

    pic_array = []
    for pic in files:
        try:
            img = cv2.imread(os.path.join(pic_path, pic))
        except:
            raise IOError

        img = img.astype(np.float32, copy=False)
        mean_pixel = [103.939, 116.779, 123.68]
        for c in range(3):
            # img[:, :, c] = img[:, :, c] / 255.0
            img[:, :, c] -= mean_pixel[c]

        img = cv2.resize(img, (350, 350))
        img = img.transpose((2, 0, 1))
        # img = np.expand_dims(img, axis=0)
        pic_array.append(img)
    return pic_array


def load_train_valid_pic(data_dir,is_write=False,save_name_x='bfp_x.train',save_name_y='bfp_y.train'):
    assert os.path.isdir(data_dir)
    files = os.listdir(data_dir)
    while '.DS_Store' in files:
        files.remove('.DS_Store')
    class_len = len(files)
    print (data_dir)
    print ('classes len : {}'.format(class_len))

    x_data = []
    y_data_len = []
    for class_i in files:
        tmp = _load_resized_pics(os.path.join(data_dir, class_i))
        x_data += tmp
        y_data_len.append(len(tmp))

    _tmp = []
    for i,cut_len in enumerate(y_data_len):
        _tmp += [i]*cut_len
    y_data = np.array(_tmp)

    y_data = np_utils.to_categorical(y_data, nb_classes=len(y_data_len))
    if is_write:
        np.save(open(save_name_x,'w'),np.array(x_data))
        np.save(open(save_name_y, 'w'), y_data)
    return np.array(x_data), y_data



def get_model_ori(save_name ,save_weights_flag,
                  art_name,batch_size,
                  use_dir_generator=True):
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

    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         # samplewise_center=True,
    #         shear_range=0.,
    #         rotation_range=0.2,
    #         fill_mode='nearest',
    #         vertical_flip=True,
    #         horizontal_flip=True,
    #         )

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
    if use_dir_generator:
        model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch + 10,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples,
                verbose=1,
                callbacks=[save_best,early_stopping])#
    else:
        X_train = np.load(open('bfp_x.train'))
        y_train = np.load(open('bfp_y.train'))
        X_valid = np.load(open('bfp_x.valid'))
        y_valid = np.load(open('bfp_y.valid'))
        model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                            samples_per_epoch=nb_train_samples,
                            nb_epoch=nb_epoch + 10,
                            verbose=1,
                            callbacks=[save_best, early_stopping],
                            validation_data=test_datagen.flow(X_valid, y_valid, batch_size=batch_size),
                            nb_val_samples=nb_validation_samples)

    from keras.models import model_from_json
    # save
    if get_model_ori_art_save_name:
        json_str = model.to_json()
        with open(get_model_ori_art_save_name, 'w') as f:
            f.write(json_str)

    return model



def get_v16_bnft(save_train_bn_ft_name,save_valid_bn_ft_name):

    model = VGG16(weights='imagenet', include_top=False)

    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     # samplewise_center=True,
    #     shear_range=0.,
    #     rotation_range=0.2,
    #     fill_mode='nearest',
    #     vertical_flip=True,
    #     horizontal_flip=True,
    # )


    test_datagen = ImageDataGenerator(rescale=1. / 255)

    generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('begin bottleneck feature train...')
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples )
    np.save(open(save_train_bn_ft_name, 'w'), bottleneck_features_train)

    generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('begin bottleneck feature validation...')
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)

    np.save(open(save_valid_bn_ft_name, 'w'), bottleneck_features_validation)



def get_bnft_clf_model(load_train_bn_ft_name,load_valid_bn_ft_name,
                       save_name ,save_art_name ,save_weights_flag
                       ):

    train_data = np.load(open(load_train_bn_ft_name))
    print('train_data shape {}'.format(train_data.shape))
    _tmp = []
    for i, cut_len in enumerate(train_file_len):
        _tmp += [i] * cut_len
    train_labels = np.array(_tmp)
    train_labels = np_utils.to_categorical(train_labels,nb_classes)
    print('train_labels shape {}'.format(train_labels.shape))

    validation_data = np.load(open(load_valid_bn_ft_name))
    _tmp = []
    for i, cut_len in enumerate(valid_file_len):
        _tmp += [i] * cut_len
    validation_labels = np.array(_tmp)
    validation_labels = np_utils.to_categorical(validation_labels, nb_classes)
    print('valid_data shape {}'.format(validation_data.shape))
    print ('nb_classes {}'.format(nb_classes))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(dense_last2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_last2, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adam', #rmsprop
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # early_stopping = EarlyStopping(monitor='loss', patience=8, verbose=1)
    save_best = ModelCheckpoint(save_name,
                                verbose=1, monitor='val_loss',
                                save_best_only=True,
                                save_weights_only=save_weights_flag
                                )
    model.fit(train_data, train_labels,verbose=1,shuffle=True,
              callbacks=[save_best],
              nb_epoch=nb_epoch+30, batch_size=batch_size, #50,32
              class_weight=class_weight,
              validation_data=(validation_data, validation_labels))

    json_str = model.to_json()
    with open(save_art_name, 'w') as f:
        f.write(json_str)

    # print (model.summary())
    # model.save_weights('clothes_uplow_bottleneck_fc_model.h5')
    # print('model output shape:{}'.format(model.output_shape))



def get_bnft_fine_tune_model(load_bnft_clf_name,save_name,save_art_name,
                             save_weights_flag,use_dir_generator=True):
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

    for layer in model.layers[:11]: #train last two block
        layer.trainable = False


    from keras import optimizers
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         # samplewise_center=True,
    #         shear_range=0.,
    #         rotation_range=0.2,
    #         fill_mode='nearest',
    #         vertical_flip=True,
    #         horizontal_flip=True
    #         )

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

    early_stopping = EarlyStopping(monitor='val_loss',patience=8,verbose=1)
    save_best = ModelCheckpoint(save_name,
                                verbose=1,monitor='val_acc',
                                save_best_only=True,
                                save_weights_only=save_weights_flag
                                )
    # epoch to csv
    csv_logger = CSVLogger(bnft_finetune_model_weights_save_name[:-2] + 'log',append=True)

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
        X_train = np.load(open('bfp_x.train'))
        y_train = np.load(open('bfp_y.train'))
        X_valid = np.load(open('bfp_x.valid'))
        y_valid = np.load(open('bfp_y.valid'))
        print('X_train shape\t:{}'.format(X_train.shape))
        print('X_valid shape\t:{}'.format(X_valid.shape))
        model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                            samples_per_epoch=nb_train_samples,
                            nb_epoch=nb_epoch + 20,
                            verbose=1,
                            callbacks=[save_best, early_stopping],
                            validation_data=test_datagen.flow(X_valid, y_valid, batch_size=batch_size),
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
    print ('*******begin******')

    cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/'

    os.chdir(cur_path)
    train_data_dir = cur_path + 'clothes/train'
    validation_data_dir = cur_path + 'clothes/validation'
    img_width, img_height = 350, 350
    dense_last2 = 256
    batch_size = 128

    pic_classes = ['back', 'front', 'profile']

    train_file_len = get_files_len_from_path(pic_classes, path=train_data_dir)
    valid_file_len = get_files_len_from_path(pic_classes, path=validation_data_dir)
    nb_train_samples = sum(train_file_len)
    nb_validation_samples = sum(valid_file_len)

    nb_epoch = 20
    LL_IsDebug = False
    nb_classes = 3

    save_info = '0321_2try_'

    save_weights_flag = True
    class_weight = {0: 0.35,  # b
                    1: 0.15,  # f
                    2: 0.5}  # p
    classes_ab = 'bfp'

    get_model_ori_weights_save_name = save_info + 'clothes_' + classes_ab + '_model_ori_weights.h5'
    get_model_ori_art_save_name = save_info + 'clothes_' + classes_ab + '_model_ori_art.json'

    get_v16_bnft_train_save_name = save_info + 'clothes_' + classes_ab + '_model_16_bn_ft_train.npy'
    get_v16_bnft_valid_save_name = save_info + 'clothes_' + classes_ab + '_model_16_bn_ft_valid.npy'

    bnft_clf_model_weights_save_name = save_info + 'clothes_' + classes_ab + '_bnft_clf_model_weights.h5'
    bnft_clf_model_art_save_name = save_info + 'clothes_' + classes_ab + '_bnft_clf_model_art.json'

    bnft_finetune_model_weights_save_name = save_info + 'clothes_' + classes_ab + '_bnft_fine_tune_model.h5'
    bnft_finetune_model_art_save_name = save_info + 'clothes_' + classes_ab + '_bnft_fine_tune_model_art.json'

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

    ## save x.train
    # X_train, y_train = load_train_valid_pic(train_data_dir,
    #                                         is_write=True,save_name_x='bfp_x.train',save_name_y='bfp_y.train')
    # X_valid,y_valid = load_train_valid_pic(validation_data_dir,is_write=True,save_name_x='bfp_x.valid',save_name_y='bfp_y.valid')
    # print('X_train shape\t:{}'.format(X_train.shape))
    # print('X_valid shape\t:{}'.format(X_valid.shape))

    ##
    use_generator = True

    get_model_ori(save_name = get_model_ori_weights_save_name,
                  save_weights_flag = save_weights_flag,
                  batch_size=batch_size,
                  art_name =get_model_ori_art_save_name,
                  use_dir_generator = use_generator)

    get_v16_bnft(save_train_bn_ft_name = get_v16_bnft_train_save_name,
                 save_valid_bn_ft_name=get_v16_bnft_valid_save_name)

    get_bnft_clf_model(load_train_bn_ft_name=get_v16_bnft_train_save_name,
                       load_valid_bn_ft_name=get_v16_bnft_valid_save_name,
                       save_name = bnft_clf_model_weights_save_name,
                       save_art_name = bnft_clf_model_art_save_name,
                       save_weights_flag = save_weights_flag)

    get_bnft_fine_tune_model(load_bnft_clf_name = bnft_clf_model_weights_save_name,
                             save_name = bnft_finetune_model_weights_save_name,
                             save_art_name=bnft_finetune_model_art_save_name,
                             save_weights_flag=save_weights_flag,
                             use_dir_generator = use_generator)
