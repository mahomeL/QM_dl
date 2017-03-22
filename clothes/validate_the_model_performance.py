# -*- coding: utf-8 -*-
from keras.models import model_from_json
import numpy as np
import os
from keras import backend as K
from collections import Counter
from tqdm import tqdm
from clothes_up_low_whole import get_files_len_from_path
K.set_image_dim_ordering('th')
cur_path = r'/Users/l_mahome/Documents/KAGGLE/open_vgg16_other/qingmu/'
import cv2

#ori_model
# model = model_from_json(open(cur_path + '0305_1try_clothes_uplow_model_ori_art.json').read())
# model.load_weights(cur_path + '0305_1try_clothes_uplow_model_ori_weights.h5')

#fine_tune_model
model = model_from_json(open(cur_path + '0305_1try_clothes_uplow_bnft_fine_tune_model_art.json').read())
model.load_weights(cur_path + '0309_2try_clothes_uplow_bnft_fine_tune_model.h5')



def get_img_test_model_perfm(pic_classes='upper_body',
                             cur_path=cur_path,train_or_valid = '',
                             verbose = 1 ,pic_begin=0):
   # assert ('train' in train_or_valid) or ('validation' in train_or_valid)
   file_path = cur_path + r'clothes/' + train_or_valid +r'/' + pic_classes
   files = os.listdir(file_path)
   if '.DS_Store' in files:
      files.remove('.DS_Store')
   tqdm.write('\n\nfile_path:  {}'.format(file_path))
   # print('\nfile_path:{}'.format(file_path))
   print ('pic_classes:  {}\npics_len:  {}'.format(pic_classes,len(files)))

   for pic in files[pic_begin:]:
      if verbose:
         print ('pic_name:{}'.format(pic))
      yield pic
      pic_name = file_path + r'/' + pic

      img = cv2.imread(pic_name)
      # print('img shape:{}'.format(img.shape))

      img = cv2.resize(img, (350, 350))
      # print('resized img shape:{}'.format(img.shape))
      img = img.astype(np.float32, copy=False)

      mean_pixel = [103.939, 116.799, 123.68]
      for c in range(3):
          img[:, :, c] = img[:, :, c] - mean_pixel[c]

      for c in range(3):
         img[:, :, c] = img[:, :, c]/255.0



      img = img.transpose((2, 0, 1))
      # print('transpose img shape:{}'.format(img.shape))
      img  = np.expand_dims(img, axis=0)
      # print('expand img shape:{}'.format(img.shape))
      yield img



##pred

def model_pred(pic_classes='lower_body', train_or_valid='validation', verbose=1):
    cur_files_len = get_files_len_from_path(path=cur_path + r'clothes/' + train_or_valid,
                                            verbose=0)
    pred_res_list = []
    img = get_img_test_model_perfm(pic_classes=pic_classes, train_or_valid=train_or_valid, verbose=verbose)
    # true_classes = dict([('lower_body',0),('upper_body',1),('whole_body',2)])
    true_classes = dict([('back', 0), ('front', 1), ('profile', 2)])
    true_classes_num2str = {v:k for k,v in true_classes.items()}
    error_pic = []

    pr_str= '{}\t:%.4f\n{}\t:%.4f\n{}\t:%.4f\n'.format(true_classes_num2str[0],
                                                    true_classes_num2str[1],
                                                    true_classes_num2str[2])
    if verbose:
        for i in range(1,cur_files_len[true_classes[pic_classes]]+1):
            try:
                pic_name = img.next()
                pred_res = model.predict(img.next())[0]
                pred_res_list.append(np.argmax(pred_res))
                if np.argmax(pred_res) != true_classes[pic_classes]:
                    error_pic.append((i,pic_name,'%.4f'%pred_res[true_classes[pic_classes]],np.argmax(pred_res),np.max(pred_res)))
                print('{}-th pic'.format(i))
                # print('lower body :%.4f\nupper body :%.4f\nwhole body :%.4f\n'%tuple(pred_res))
                print(pr_str % tuple(pred_res))
            except StopIteration:
                print('current stop-i:{}'.format(i))
                break
            finally:
                pass

    else:
        # print (cur_files_len[true_classes[pic_classes]]+1)
        for i in tqdm(range(1, cur_files_len[true_classes[pic_classes]]+1)):
            try:
                pic_name = img.next()
                pred_res = model.predict(img.next())[0]
                pred_res_list.append(np.argmax(pred_res))
                if np.argmax(pred_res) != true_classes[pic_classes]:
                    error_pic.append((i,pic_name,'%.4f'%pred_res[true_classes[pic_classes]],
                                      np.argmax(pred_res),np.max(pred_res)))
                # print('{}-th pic'.format(i))
                # print('Predict Result:\nlower body :%.4f\nupper body :%.4f\nwhole body :%.4f\n' % tuple(pred_res))
                # print(pr_str%tuple(pred_res))
            except StopIteration:
                print('current stop-i:{}'.format(i))
                break
            finally:
                pass

    pred_count = Counter(pred_res_list)
    print('\n*****RESULT COUNT*****')
    print('lower body :{}\nupper body :{}\nwhole body :{}'.format(pred_count[0],pred_count[1],pred_count[2]))
    print('\naccuracy :{:.3f}'.format(pred_count[true_classes[pic_classes]]/float(cur_files_len[true_classes[pic_classes]])))
    if len(error_pic):
        print('\nerror pic (index,name,prob,er-class,er-prob) is :\n{}'.format(error_pic))


##RESULT
_pic_classes='whole_body'
_verbose=0
_train_or_valid = 'test'
model_pred(pic_classes=_pic_classes,
           train_or_valid=_train_or_valid,
           verbose=_verbose)

